#!/usr/bin/env python3
"""
01_run_experiments.py
Dispatches all benchmark jobs across M2 Ultra / M1 MAX / M2 Pro.
Reads discovered_paths.json written by 00_discover.py.

Experiment types:
  A. MoE baseline         — 35B-A3B alone, Q4 + Q8
  B. MoE + SD             — 35B-A3B + draft 0.8B / 2B, sweep γ ∈ {4,8,16}
  C. Dense baselines      — 0.8B / 2B / 4B / 9B alone
  D. Dense + SD           — 4B / 9B as target + 0.8B draft
  E. Cross-arch           — Gemma-3-4B alone + with 0.8B draft (same-size Qwen as draft)
  F. Prompt-length sweep  — MoE Q4 + best draft, vary prompt tokens {64,256,512}

Each run logs raw llama.cpp stdout to results/raw/<job_id>.txt
Results are summarised into results/csv/runs.csv in real time.
"""

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BATCH_SIZES, DRAFT_LENGTHS, HOME, LLAMA_BIN, MACHINES,
    MODEL_ROOT, MODELS, N_PREDICT, N_PROMPT_TOKENS, N_REPEATS,
    PROMPTS, TEMPERATURE,
)

BASE   = Path(__file__).parent
PATHS  = json.loads((BASE / "discovered_paths.json").read_text())
BINS   = PATHS["bins"]
MFILES = PATHS["models"]
SSH_OK = PATHS["ssh_ok"]
RES    = Path(PATHS["results_dir"])

CSV_FILE = RES / "csv" / "runs.csv"
CSV_COLS = [
    "job_id", "timestamp", "machine", "chip",
    "target_model", "draft_model", "draft_len",
    "n_prompt", "n_predict", "repeat",
    "prompt_key",
    "tokens_per_sec", "prompt_eval_ms", "eval_ms_per_token",
    "draft_acceptance_rate",
    "speedup_vs_baseline",
    "raw_log",
    "status",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg): print(f"[{datetime.now():%H:%M:%S}] {msg}")

def find_model_info(model_id):
    return next((m for m in MODELS if m["id"] == model_id), None)

def get_machine(model_id):
    m = find_model_info(model_id)
    return m["machine"] if m else "m2ultra"

def ssh_prefix(machine):
    info = MACHINES[machine]
    if info["host"] == "local":
        return []
    user = info.get("name", "")
    ssh_target = f"{user}@{info['host']}" if user else info["host"]
    return ["ssh", ssh_target]

def run_cmd(cmd_list, machine, log_file):
    """Run cmd on machine, tee output to log_file, return stdout."""
    pfx = ssh_prefix(machine)
    full = pfx + cmd_list if pfx else cmd_list
    log_file.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(full, capture_output=True, text=True, timeout=600)
    combined = r.stdout + "\n" + r.stderr
    with open(log_file, "w") as fh:
        fh.write(combined)
    return combined, r.returncode

def parse_llama_bench_output(raw):
    """Extract tokens/sec and latency from llama-bench CSV stdout."""
    import io
    result = {"tokens_per_sec": None, "prompt_eval_ms": None, "eval_ms_per_token": None}
    lines = raw.splitlines()
    # Find the CSV header line (contains "avg_ts")
    header_idx = None
    for i, line in enumerate(lines):
        if "avg_ts" in line and "n_gen" in line:
            header_idx = i
            break
    if header_idx is None:
        return result
    # Use proper CSV reader to handle quoted fields with commas
    csv_text = "\n".join(lines[header_idx:header_idx + 10])  # header + a few data rows
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            n_gen = int(row.get("n_gen", "0") or "0")
            n_prompt = int(row.get("n_prompt", "0") or "0")
            avg_ts = float(row.get("avg_ts", "0") or "0")
            avg_ns = float(row.get("avg_ns", "0") or "0")
        except (ValueError, TypeError):
            continue
        if n_gen > 0 and avg_ts > 0:
            result["tokens_per_sec"] = avg_ts
            result["eval_ms_per_token"] = (avg_ns / 1e6) / n_gen
        elif n_prompt > 0 and avg_ns > 0:
            result["prompt_eval_ms"] = avg_ns / 1e6
    return result

def parse_speculative_output(raw):
    """Extract acceptance rate and speed from llama-speculative stdout."""
    result = {"tokens_per_sec": None, "draft_acceptance_rate": None, "eval_ms_per_token": None}
    # acceptance rate: "accept    = 85.000%"
    m = re.search(r"accept\s*=\s*([\d.]+)\s*%", raw)
    if m:
        result["draft_acceptance_rate"] = float(m.group(1)) / 100.0
    # Prefer the "decoded X tokens in Y seconds, speed: Z t/s" line (wall-clock)
    m_dec = re.search(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", raw)
    if m_dec:
        n_tok = int(m_dec.group(1))
        total_sec = float(m_dec.group(2))
        result["tokens_per_sec"] = float(m_dec.group(3))
        result["eval_ms_per_token"] = (total_sec * 1000.0) / n_tok if n_tok > 0 else None
    else:
        # Fallback: compute from n_predict and target total time
        m_np = re.search(r"n_predict\s*=\s*(\d+)", raw)
        n_predict = int(m_np.group(1)) if m_np else None
        target_section = raw.split("target:")[-1] if "target:" in raw else raw
        m_total = re.search(r"total time\s*=\s*([\d.]+)\s*ms", target_section)
        if m_total and n_predict and n_predict > 0:
            total_ms = float(m_total.group(1))
            result["tokens_per_sec"] = n_predict / (total_ms / 1000.0)
            result["eval_ms_per_token"] = total_ms / n_predict
    return result

# ── CSV writer ────────────────────────────────────────────────────────────────
def init_csv():
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="") as f:
            csv.DictWriter(f, CSV_COLS).writeheader()

def append_csv(row):
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, CSV_COLS, extrasaction="ignore")
        w.writerow(row)

# ── Single job builders ────────────────────────────────────────────────────────
def bench_cmd(model_file, n_prompt, n_predict):
    """llama-bench baseline (no draft)."""
    return [
        BINS["llama-bench"],
        "-m", model_file,
        "-p", str(n_prompt),
        "-n", str(n_predict),
        "-r", "1",
        "-o", "csv",
    ]

def speculative_cmd(target_file, draft_file, draft_len, prompt_text, n_predict):
    """llama-speculative run."""
    return [
        BINS["llama-speculative"],
        "-m",  target_file,
        "-md", draft_file,
        "--draft", str(draft_len),
        "-n", str(n_predict),
        "--temp", str(TEMPERATURE),
        "-p", prompt_text,
    ]

# ── Experiment suites ─────────────────────────────────────────────────────────
def suite_A_moe_baseline():
    """A: MoE 35B-A3B without speculative decoding."""
    log("=== Suite A: MoE baseline ===")
    for moe_id in ["moe_q4", "moe_q8"]:
        if moe_id not in MFILES: continue
        machine = get_machine(moe_id)
        for n_prompt in N_PROMPT_TOKENS:
            for rep in range(N_REPEATS):
                job_id = f"A_{moe_id}_np{n_prompt}_r{rep}"
                log(f"  {job_id}")
                cmd = bench_cmd(MFILES[moe_id], n_prompt, N_PREDICT)
                raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
                parsed = parse_llama_bench_output(raw)
                append_csv({
                    "job_id": job_id, "timestamp": datetime.now().isoformat(),
                    "machine": machine, "chip": MACHINES[machine]["chip"],
                    "target_model": moe_id, "draft_model": "none",
                    "draft_len": 0, "n_prompt": n_prompt, "n_predict": N_PREDICT,
                    "repeat": rep, "prompt_key": "bench",
                    **parsed,
                    "draft_acceptance_rate": None,
                    "speedup_vs_baseline": 1.0,
                    "raw_log": f"raw/{job_id}.txt",
                    "status": "ok" if rc == 0 else f"err_{rc}",
                })

def suite_B_moe_sd():
    """B: MoE + speculative decoding, sweep γ and draft model size."""
    log("=== Suite B: MoE + Speculative Decoding ===")
    for moe_id in ["moe_q4", "moe_q8"]:
        if moe_id not in MFILES: continue
        machine = get_machine(moe_id)
        for draft_id in ["draft_0.8b", "draft_2b"]:
            if draft_id not in MFILES: continue
            for gamma in DRAFT_LENGTHS:
                for pk, prompt in PROMPTS.items():
                    for rep in range(N_REPEATS):
                        job_id = f"B_{moe_id}_{draft_id}_g{gamma}_{pk}_r{rep}"
                        log(f"  {job_id}")
                        cmd = speculative_cmd(
                            MFILES[moe_id], MFILES[draft_id],
                            gamma, prompt, N_PREDICT
                        )
                        raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
                        parsed = parse_speculative_output(raw)
                        append_csv({
                            "job_id": job_id, "timestamp": datetime.now().isoformat(),
                            "machine": machine, "chip": MACHINES[machine]["chip"],
                            "target_model": moe_id, "draft_model": draft_id,
                            "draft_len": gamma, "n_prompt": len(prompt.split()),
                            "n_predict": N_PREDICT, "repeat": rep, "prompt_key": pk,
                            **parsed,
                            "prompt_eval_ms": None,
                            "speedup_vs_baseline": None,  # computed in analysis
                            "raw_log": f"raw/{job_id}.txt",
                            "status": "ok" if rc == 0 else f"err_{rc}",
                        })

def suite_C_dense_baseline():
    """C: Dense models baseline (no speculative decoding)."""
    log("=== Suite C: Dense baselines ===")
    dense_ids = ["dense_0.8b", "dense_2b", "dense_4b", "dense_9b", "gemma_4b_q4"]
    for mid in dense_ids:
        if mid not in MFILES: continue
        minfo = find_model_info(mid)
        machine = minfo["machine"]
        if not SSH_OK.get(machine, True):
            log(f"  Skipping {mid}: SSH to {machine} unavailable, running on m2ultra instead")
            machine = "m2ultra"
        for n_prompt in N_PROMPT_TOKENS:
            for rep in range(N_REPEATS):
                job_id = f"C_{mid}_np{n_prompt}_r{rep}"
                log(f"  {job_id}")
                cmd = bench_cmd(MFILES[mid], n_prompt, N_PREDICT)
                raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
                parsed = parse_llama_bench_output(raw)
                append_csv({
                    "job_id": job_id, "timestamp": datetime.now().isoformat(),
                    "machine": machine, "chip": MACHINES[machine]["chip"],
                    "target_model": mid, "draft_model": "none",
                    "draft_len": 0, "n_prompt": n_prompt, "n_predict": N_PREDICT,
                    "repeat": rep, "prompt_key": "bench",
                    **parsed,
                    "draft_acceptance_rate": None,
                    "speedup_vs_baseline": 1.0,
                    "raw_log": f"raw/{job_id}.txt",
                    "status": "ok" if rc == 0 else f"err_{rc}",
                })

def suite_D_dense_sd():
    """D: Dense target models + speculative decoding (for comparison with Suite B)."""
    log("=== Suite D: Dense + Speculative Decoding ===")
    targets = ["dense_4b", "dense_9b"]
    for target_id in targets:
        if target_id not in MFILES: continue
        minfo = find_model_info(target_id)
        machine = minfo["machine"]
        if not SSH_OK.get(machine, True):
            machine = "m2ultra"
        draft_id = "draft_0.8b"
        if draft_id not in MFILES: continue
        for gamma in DRAFT_LENGTHS:
            for pk, prompt in PROMPTS.items():
                for rep in range(N_REPEATS):
                    job_id = f"D_{target_id}_{draft_id}_g{gamma}_{pk}_r{rep}"
                    log(f"  {job_id}")
                    cmd = speculative_cmd(
                        MFILES[target_id], MFILES[draft_id],
                        gamma, prompt, N_PREDICT
                    )
                    raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
                    parsed = parse_speculative_output(raw)
                    append_csv({
                        "job_id": job_id, "timestamp": datetime.now().isoformat(),
                        "machine": machine, "chip": MACHINES[machine]["chip"],
                        "target_model": target_id, "draft_model": draft_id,
                        "draft_len": gamma, "n_prompt": len(prompt.split()),
                        "n_predict": N_PREDICT, "repeat": rep, "prompt_key": pk,
                        **parsed,
                        "prompt_eval_ms": None,
                        "speedup_vs_baseline": None,
                        "raw_log": f"raw/{job_id}.txt",
                        "status": "ok" if rc == 0 else f"err_{rc}",
                    })

def suite_F_prompt_sweep():
    """F: MoE Q4 + best draft, sweep prompt length to study memory-bound regime shift."""
    log("=== Suite F: Prompt-length sweep ===")
    moe_id   = "moe_q4"
    draft_id = "draft_0.8b"
    if moe_id not in MFILES or draft_id not in MFILES: return
    machine = get_machine(moe_id)
    # generate synthetic prompts of different token counts
    base_prompt = "Please write a detailed analysis of " + ("the following topic. " * 50)
    for n_prompt in [32, 64, 128, 256, 512, 1024]:
        # adjust prompt length
        prompt_words = base_prompt.split()[:n_prompt]
        prompt = " ".join(prompt_words)
        for rep in range(N_REPEATS):
            # baseline
            job_id = f"F_base_np{n_prompt}_r{rep}"
            cmd = bench_cmd(MFILES[moe_id], n_prompt, N_PREDICT)
            raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
            parsed = parse_llama_bench_output(raw)
            append_csv({
                "job_id": job_id, "timestamp": datetime.now().isoformat(),
                "machine": machine, "chip": MACHINES[machine]["chip"],
                "target_model": moe_id, "draft_model": "none",
                "draft_len": 0, "n_prompt": n_prompt, "n_predict": N_PREDICT,
                "repeat": rep, "prompt_key": "sweep",
                **parsed,
                "draft_acceptance_rate": None, "speedup_vs_baseline": 1.0,
                "raw_log": f"raw/{job_id}.txt",
                "status": "ok" if rc == 0 else f"err_{rc}",
            })
            # with SD
            job_id = f"F_sd_np{n_prompt}_r{rep}"
            cmd = speculative_cmd(MFILES[moe_id], MFILES[draft_id], 8, prompt, N_PREDICT)
            raw, rc = run_cmd(cmd, machine, RES / "raw" / f"{job_id}.txt")
            parsed = parse_speculative_output(raw)
            append_csv({
                "job_id": job_id, "timestamp": datetime.now().isoformat(),
                "machine": machine, "chip": MACHINES[machine]["chip"],
                "target_model": moe_id, "draft_model": draft_id,
                "draft_len": 8, "n_prompt": n_prompt, "n_predict": N_PREDICT,
                "repeat": rep, "prompt_key": "sweep",
                **parsed,
                "prompt_eval_ms": None, "speedup_vs_baseline": None,
                "raw_log": f"raw/{job_id}.txt",
                "status": "ok" if rc == 0 else f"err_{rc}",
            })

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", nargs="+",
        default=["A", "B", "C", "D", "F"],
        choices=["A", "B", "C", "D", "F"],
        help="Which experiment suites to run"
    )
    args = parser.parse_args()

    init_csv()
    log(f"Starting experiments. CSV: {CSV_FILE}")
    log(f"Suites: {args.suites}")

    suite_map = {
        "A": suite_A_moe_baseline,
        "B": suite_B_moe_sd,
        "C": suite_C_dense_baseline,
        "D": suite_D_dense_sd,
        "F": suite_F_prompt_sweep,
    }
    for s in args.suites:
        suite_map[s]()

    log(f"Done. Results in {CSV_FILE}")
    log("Next: python3 02_analyze.py")

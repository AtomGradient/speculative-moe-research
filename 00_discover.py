#!/usr/bin/env python3
"""
00_discover.py
Run this first. Verifies:
  - llama.cpp binaries exist and are executable
  - All model GGUF files are found (handles nested dirs)
  - SSH connections to remote machines work
  - Python env is accessible on remotes
Writes discovered paths to discovered_paths.json for use by later scripts.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import HOME, LLAMA_BIN, MACHINES, MODEL_ROOT, MODELS, PYTHON_ENV

RESULTS_DIR = Path(HOME) / "speculative-moe-research" / "results"
OUT_FILE = Path(__file__).parent / "discovered_paths.json"

# ANSI colours
GRN = "\033[92m"; RED = "\033[91m"; YLW = "\033[93m"; RST = "\033[0m"; BLD = "\033[1m"
ok  = lambda s: f"{GRN}✓{RST} {s}"
err = lambda s: f"{RED}✗{RST} {s}"
wrn = lambda s: f"{YLW}!{RST} {s}"

def section(title):
    print(f"\n{BLD}{'─'*60}{RST}")
    print(f"{BLD}  {title}{RST}")
    print(f"{BLD}{'─'*60}{RST}")

# ── 1. llama.cpp binaries ──────────────────────────────────────────────────────
section("llama.cpp binaries")
REQUIRED_BINS = ["llama-bench", "llama-cli", "llama-speculative"]
OPTIONAL_BINS = ["llama-server"]
found_bins = {}

for b in REQUIRED_BINS + OPTIONAL_BINS:
    for candidate in [
        Path(LLAMA_BIN) / "build" / "bin" / b,
        Path(LLAMA_BIN) / b,
        Path(LLAMA_BIN) / "build" / b,
    ]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            found_bins[b] = str(candidate)
            print(ok(f"{b}  →  {candidate}"))
            break
    else:
        if b in REQUIRED_BINS:
            print(err(f"{b}  NOT FOUND in {LLAMA_BIN}"))
        else:
            print(wrn(f"{b}  not found (optional)"))

# ── 2. Model GGUF files ────────────────────────────────────────────────────────
section("Model GGUF files")
found_models = {}

for m in MODELS:
    model_dir = Path(MODEL_ROOT) / m["dir"]
    if not model_dir.exists():
        print(err(f"{m['label']}  dir not found: {model_dir}"))
        continue
    gguf_files = sorted(model_dir.rglob("*.gguf"))
    if not gguf_files:
        print(err(f"{m['label']}  no .gguf in {model_dir}"))
        continue
    # prefer Q4_K or Q8_K main shard; fall back to first file
    chosen = gguf_files[0]
    for f in gguf_files:
        n = f.name.lower()
        if "00001" in n or ("q4" in n and "split" not in n) or ("q8" in n and "split" not in n):
            chosen = f
            break
    found_models[m["id"]] = str(chosen)
    size_gb = chosen.stat().st_size / 1e9
    print(ok(f"{m['label']:40s}  {chosen.name}  ({size_gb:.1f} GB)"))

# ── 3. SSH + remote env ────────────────────────────────────────────────────────
section("SSH connectivity & remote Python env")
ssh_ok = {}

for name, info in MACHINES.items():
    if info["host"] == "local":
        print(ok(f"{name} ({info['chip']})  local — skipping SSH"))
        ssh_ok[name] = True
        continue
    host = info["host"]
    user = info.get("name", "")
    ssh_target = f"{user}@{host}" if user else host
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         ssh_target, f"source {PYTHON_ENV} && python3 -c 'import sys; print(sys.version)'"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        ver = r.stdout.strip().split("\n")[-1]
        print(ok(f"{name} ({info['chip']})  SSH OK  python {ver}"))
        ssh_ok[name] = True
    else:
        print(err(f"{name} ({info['chip']})  SSH FAILED to {host}"))
        print(f"   stderr: {r.stderr.strip()[:120]}")
        ssh_ok[name] = False

# ── 4. Results dir ─────────────────────────────────────────────────────────────
section("Results directory")
for sub in ["raw", "csv", "plots", "logs"]:
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
print(ok(f"Results dir: {RESULTS_DIR}"))

# ── 5. Write discovered_paths.json ────────────────────────────────────────────
section("Summary")
payload = {
    "bins":       found_bins,
    "models":     found_models,
    "ssh_ok":     ssh_ok,
    "results_dir": str(RESULTS_DIR),
}
OUT_FILE.write_text(json.dumps(payload, indent=2))
print(ok(f"Wrote {OUT_FILE}"))

missing_bins   = [b for b in REQUIRED_BINS if b not in found_bins]
missing_models = [m["id"] for m in MODELS if m["id"] not in found_models]
ssh_failed     = [k for k, v in ssh_ok.items() if not v]

if missing_bins:
    print(err(f"Missing required binaries: {missing_bins}"))
    print(wrn("  Build llama.cpp:  cd ~/llama.cpp && cmake -B build && cmake --build build -j"))
if missing_models:
    print(wrn(f"Missing models: {missing_models}  (will be skipped)"))
if ssh_failed:
    print(wrn(f"SSH failed: {ssh_failed}  (those experiments will be skipped)"))

if not missing_bins:
    print(f"\n{GRN}{BLD}✓ Ready to run experiments.{RST}  Next: python3 01_run_experiments.py")
else:
    print(f"\n{RED}{BLD}✗ Fix missing binaries before proceeding.{RST}")
    sys.exit(1)

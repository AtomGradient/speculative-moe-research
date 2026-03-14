#!/usr/bin/env python3
"""
02_analyze.py
Reads results/csv/runs.csv, computes derived metrics, and generates:
  - results/csv/summary.csv    (clean aggregated table for the paper)
  - results/plots/fig1_speedup_curve.png
  - results/plots/fig2_acceptance_rate.png
  - results/plots/fig3_moe_vs_dense.png
  - results/plots/fig4_prompt_sweep.png

Run after 01_run_experiments.py completes (or partially, for interim results).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
CSV  = BASE / "results" / "csv" / "runs.csv"
PLOT = BASE / "results" / "plots"
PLOT.mkdir(parents=True, exist_ok=True)

PAPER_STYLE = {
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
}
plt.rcParams.update(PAPER_STYLE)

MODEL_LABELS = {
    "moe_q4":     "MoE 35B-A3B Q4\n(active 3B)",
    "moe_q8":     "MoE 35B-A3B Q8\n(active 3B)",
    "dense_0.8b": "Dense 0.8B",
    "dense_2b":   "Dense 2B",
    "dense_4b":   "Dense 4B",
    "dense_9b":   "Dense 9B",
    "gemma_4b_q4":"Gemma-3 4B Q4",
}
ACTIVE_PARAMS = {
    "moe_q4": 3.0, "moe_q8": 3.0,
    "dense_0.8b": 0.8, "dense_2b": 2.0,
    "dense_4b": 4.0, "dense_9b": 9.0,
    "gemma_4b_q4": 4.0,
}
COLORS = {
    "moe":   "#534AB7",  # purple — MoE
    "dense": "#1D9E75",  # teal   — Dense
    "gemma": "#D85A30",  # coral  — cross-arch
    "draft08": "#BA7517",
    "draft2b": "#888780",
}

# ── Load ───────────────────────────────────────────────────────────────────────
if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run 01_run_experiments.py first.")
    sys.exit(1)

df = pd.read_csv(CSV)
df = df[df["status"] == "ok"].copy()
df["active_params_b"] = df["target_model"].map(ACTIVE_PARAMS)
df["has_draft"] = df["draft_model"] != "none"

print(f"Loaded {len(df)} successful runs.")

# ── Compute speedup ────────────────────────────────────────────────────────────
# For each (target_model, n_prompt, prompt_key) group, baseline = mean tps without draft
baselines = (
    df[~df["has_draft"]]
    .groupby(["target_model", "n_prompt", "prompt_key"])["tokens_per_sec"]
    .mean()
    .rename("baseline_tps")
    .reset_index()
)
df = df.merge(baselines, on=["target_model", "n_prompt", "prompt_key"], how="left")
df["speedup"] = df["tokens_per_sec"] / df["baseline_tps"]

# ── Summary CSV ───────────────────────────────────────────────────────────────
agg = (
    df.groupby(["target_model", "draft_model", "draft_len", "n_prompt", "prompt_key"])
    .agg(
        mean_tps=("tokens_per_sec", "mean"),
        std_tps=("tokens_per_sec", "std"),
        mean_speedup=("speedup", "mean"),
        mean_acceptance=("draft_acceptance_rate", "mean"),
        n_runs=("job_id", "count"),
    )
    .reset_index()
)
agg["active_params_b"] = agg["target_model"].map(ACTIVE_PARAMS)
agg.to_csv(BASE / "results" / "csv" / "summary.csv", index=False)
print(f"Written summary.csv ({len(agg)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Speedup vs active parameter count  (main result)
# X = active params, Y = SD speedup ratio, separate lines per draft model
# MoE point highlighted with a different marker
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, draft_id, draft_label, col in [
    (axes[0], "draft_0.8b", "Draft 0.8B", COLORS["draft08"]),
    (axes[1], "draft_2b",   "Draft 2B",   COLORS["draft2b"]),
]:
    sub = agg[(agg["draft_model"] == draft_id) & (agg["draft_len"] == 8)]
    if sub.empty:
        ax.set_title(f"No data yet for {draft_label}")
        continue

    dense_pts = sub[~sub["target_model"].str.startswith("moe")]
    moe_pts   = sub[sub["target_model"].str.startswith("moe")]

    ax.errorbar(
        dense_pts["active_params_b"], dense_pts["mean_speedup"],
        yerr=dense_pts["std_tps"] / dense_pts["mean_tps"] * dense_pts["mean_speedup"],
        fmt="o-", color=COLORS["dense"], label="Dense", linewidth=1.5, markersize=6,
    )
    ax.scatter(
        moe_pts["active_params_b"], moe_pts["mean_speedup"],
        color=COLORS["moe"], marker="*", s=180, zorder=5, label="MoE (35B-A3B)",
    )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Active parameters (B)")
    ax.set_ylabel("Speedup ratio  (SD / no-SD)")
    ax.set_title(f"SD speedup — {draft_label}")
    ax.legend(frameon=False)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

fig.suptitle(
    "Fig 1: Does MoE's low active-parameter count reduce Speculative Decoding gains?\n"
    "★ = MoE point (35B total, 3B active)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(PLOT / "fig1_speedup_curve.png", bbox_inches="tight")
print("Saved fig1_speedup_curve.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Draft acceptance rate — MoE vs Dense at matched active params
# ═══════════════════════════════════════════════════════════════════════════════
sd_only = agg[agg["has_draft"] if "has_draft" in agg.columns
              else agg["draft_model"] != "none"].copy()
sd_only = agg[agg["draft_model"] != "none"]

fig, ax = plt.subplots(figsize=(8, 5))
for target_id, grp in sd_only.groupby("target_model"):
    grp = grp.sort_values("draft_len")
    color = COLORS["moe"] if "moe" in target_id else COLORS["dense"]
    marker = "*" if "moe" in target_id else "o"
    label = MODEL_LABELS.get(target_id, target_id)
    ax.plot(
        grp["draft_len"], grp["mean_acceptance"],
        marker=marker, color=color, label=label,
        linewidth=1.5, markersize=8 if marker == "*" else 5,
    )

ax.set_xlabel("Draft length γ")
ax.set_ylabel("Mean draft acceptance rate  α")
ax.set_title("Fig 2: Acceptance rate vs draft length\nMoE vs Dense targets")
ax.legend(frameon=False, fontsize=9)
ax.set_xticks(DRAFT_LENGTHS if agg["draft_len"].nunique() > 1 else [4, 8, 16])
fig.tight_layout()
fig.savefig(PLOT / "fig2_acceptance_rate.png", bbox_inches="tight")
print("Saved fig2_acceptance_rate.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: Raw throughput — MoE vs Dense (baseline, no SD)
# Bars sorted by active_params_b; MoE bar marked distinctly
# ═══════════════════════════════════════════════════════════════════════════════
baseline_agg = agg[(agg["draft_model"] == "none") & (agg["n_prompt"] == 64)]
baseline_agg = baseline_agg.sort_values("active_params_b")

if not baseline_agg.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs    = range(len(baseline_agg))
    colors = [COLORS["moe"] if "moe" in r else
              COLORS["gemma"] if "gemma" in r else
              COLORS["dense"]
              for r in baseline_agg["target_model"]]
    bars = ax.bar(xs, baseline_agg["mean_tps"], color=colors,
                  yerr=baseline_agg["std_tps"], capsize=4, width=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [MODEL_LABELS.get(r, r) for r in baseline_agg["target_model"]],
        rotation=30, ha="right", fontsize=9
    )
    ax.set_ylabel("Tokens / second")
    ax.set_title("Fig 3: Baseline throughput (no speculative decoding)\npurple = MoE, teal = dense")
    fig.tight_layout()
    fig.savefig(PLOT / "fig3_moe_vs_dense.png", bbox_inches="tight")
    print("Saved fig3_moe_vs_dense.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: SD speedup vs prompt length  (memory-bound regime analysis)
# ═══════════════════════════════════════════════════════════════════════════════
sweep = df[df["prompt_key"] == "sweep"].copy()
if not sweep.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    for has_sd, label, color, ls in [
        (False, "No SD (baseline)", "gray", "--"),
        (True,  "With SD (0.8B draft, γ=8)", COLORS["moe"], "-"),
    ]:
        grp = sweep[sweep["has_draft"] == has_sd].groupby("n_prompt")["tokens_per_sec"].mean()
        ax.plot(grp.index, grp.values, color=color, linestyle=ls,
                marker="o", label=label, linewidth=1.5)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Fig 4: Speculative Decoding gain vs prompt length\n(MoE 35B-A3B Q4)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT / "fig4_prompt_sweep.png", bbox_inches="tight")
    print("Saved fig4_prompt_sweep.png")
    plt.close()

# ── Print key numbers for the paper ──────────────────────────────────────────
print("\n" + "═"*60)
print("KEY RESULTS SUMMARY")
print("═"*60)
for mid in ["moe_q4", "dense_4b", "dense_9b"]:
    row = agg[(agg["target_model"] == mid) & (agg["draft_model"] == "draft_0.8b") &
              (agg["draft_len"] == 8) & (agg["n_prompt"] == 64)]
    if row.empty: continue
    r = row.iloc[0]
    print(f"{MODEL_LABELS.get(mid, mid):30s}  "
          f"speedup={r['mean_speedup']:.2f}x  "
          f"acceptance={r['mean_acceptance']:.2%}  "
          f"tps={r['mean_tps']:.1f}")
print("═"*60)
print(f"\nAll plots → {PLOT}")

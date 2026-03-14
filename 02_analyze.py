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

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
}
ACTIVE_PARAMS = {
    "moe_q4": 3.0, "moe_q8": 3.0,
    "dense_0.8b": 0.8, "dense_2b": 2.0,
    "dense_4b": 4.0, "dense_9b": 9.0,
}
COLORS = {
    "moe":     "#534AB7",  # purple — MoE
    "dense":   "#1D9E75",  # teal   — Dense
    "draft08": "#BA7517",  # amber  — 0.8B draft
    "draft2b": "#888780",  # gray   — 2B draft
    "sd":      "#E63946",  # red    — SD line
    "baseline":"#6B7280",  # gray   — baseline
}
DRAFT_LENGTHS = [4, 8, 16]

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
# Use mean baseline tps per target_model (averaging across prompt lengths)
baselines = (
    df[~df["has_draft"]]
    .groupby("target_model")["tokens_per_sec"]
    .mean()
    .rename("baseline_tps")
)
df = df.merge(baselines, on="target_model", how="left")
df["speedup"] = df["tokens_per_sec"] / df["baseline_tps"]

# ── Summary CSV ───────────────────────────────────────────────────────────────
# Aggregate: exclude sweep data for main summary
main_df = df[df["prompt_key"] != "sweep"]
agg = (
    main_df.groupby(["target_model", "draft_model", "draft_len"])
    .agg(
        mean_tps=("tokens_per_sec", "mean"),
        std_tps=("tokens_per_sec", "std"),
        mean_speedup=("speedup", "mean"),
        std_speedup=("speedup", "std"),
        mean_acceptance=("draft_acceptance_rate", "mean"),
        n_runs=("job_id", "count"),
    )
    .reset_index()
)
agg["active_params_b"] = agg["target_model"].map(ACTIVE_PARAMS)
agg.to_csv(BASE / "results" / "csv" / "summary.csv", index=False)
print(f"Written summary.csv ({len(agg)} rows)")

# Also make a per-prompt-key summary
agg_prompt = (
    main_df[main_df["has_draft"]]
    .groupby(["target_model", "draft_model", "draft_len", "prompt_key"])
    .agg(
        mean_tps=("tokens_per_sec", "mean"),
        mean_acceptance=("draft_acceptance_rate", "mean"),
        mean_speedup=("speedup", "mean"),
        n_runs=("job_id", "count"),
    )
    .reset_index()
)
agg_prompt.to_csv(BASE / "results" / "csv" / "summary_by_prompt.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Speedup vs active parameter count  (main result)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

# Get SD runs with γ=8 and 0.8B draft (best config for comparison)
sd_g8 = agg[(agg["draft_model"] == "draft_0.8b") & (agg["draft_len"] == 8)]
dense_sd = sd_g8[~sd_g8["target_model"].str.startswith("moe")]
moe_sd = sd_g8[sd_g8["target_model"].str.startswith("moe")]

if not dense_sd.empty:
    dense_sd = dense_sd.sort_values("active_params_b")
    ax.errorbar(
        dense_sd["active_params_b"], dense_sd["mean_speedup"],
        yerr=dense_sd["std_speedup"].fillna(0),
        fmt="o-", color=COLORS["dense"], label="Dense models",
        linewidth=2, markersize=8, capsize=4,
    )
if not moe_sd.empty:
    for _, row in moe_sd.iterrows():
        ax.scatter(
            row["active_params_b"], row["mean_speedup"],
            color=COLORS["moe"], marker="*", s=250, zorder=5,
            label=MODEL_LABELS.get(row["target_model"], row["target_model"]),
        )

ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="No speedup (1.0x)")
ax.set_xlabel("Active parameters (B)", fontsize=12)
ax.set_ylabel("Speedup ratio  (SD / baseline)", fontsize=12)
ax.set_title(
    "Speculative Decoding Speedup vs Active Parameter Count\n"
    "(Draft: Qwen3.5-0.8B, γ=8, M2 Ultra)",
    fontsize=12, fontweight="bold",
)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks([1, 2, 3, 4, 5, 9])
ax.set_xticklabels(["1B", "2B", "3B", "4B", "5B", "9B"])
fig.tight_layout()
fig.savefig(PLOT / "fig1_speedup_curve.png", bbox_inches="tight")
print("Saved fig1_speedup_curve.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Draft acceptance rate vs draft length — MoE vs Dense
# ═══════════════════════════════════════════════════════════════════════════════
sd_data = agg[agg["draft_model"] != "none"].copy()

fig, ax = plt.subplots(figsize=(8, 5.5))
for target_id in ["moe_q4", "moe_q8", "dense_4b", "dense_9b"]:
    for draft_id in ["draft_0.8b", "draft_2b"]:
        sub = sd_data[(sd_data["target_model"] == target_id) & (sd_data["draft_model"] == draft_id)]
        if sub.empty:
            continue
        sub = sub.sort_values("draft_len")
        is_moe = "moe" in target_id
        color = COLORS["moe"] if is_moe else COLORS["dense"]
        marker = "*" if is_moe else "o"
        ls = "-" if "0.8b" in draft_id else "--"
        draft_label = "0.8B" if "0.8b" in draft_id else "2B"
        label = f"{MODEL_LABELS.get(target_id, target_id).replace(chr(10), ' ')} + {draft_label}"
        ax.plot(
            sub["draft_len"], sub["mean_acceptance"],
            marker=marker, color=color, linestyle=ls, label=label,
            linewidth=1.5, markersize=10 if marker == "*" else 6,
        )

ax.set_xlabel("Draft length γ", fontsize=12)
ax.set_ylabel("Mean draft acceptance rate α", fontsize=12)
ax.set_title("Draft Acceptance Rate vs Draft Length\nMoE vs Dense Targets", fontsize=12, fontweight="bold")
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8, loc="upper right")
ax.set_xticks(DRAFT_LENGTHS)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(PLOT / "fig2_acceptance_rate.png", bbox_inches="tight")
print("Saved fig2_acceptance_rate.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: Raw throughput comparison — MoE vs Dense (baseline + with SD)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

models_order = ["dense_0.8b", "dense_2b", "moe_q4", "moe_q8", "dense_4b", "dense_9b"]
models_present = [m for m in models_order if m in agg["target_model"].values]

x = np.arange(len(models_present))
width = 0.25

# Baseline bars
baseline_data = agg[(agg["draft_model"] == "none")]
baseline_vals = [baseline_data[baseline_data["target_model"] == m]["mean_tps"].values[0]
                 if m in baseline_data["target_model"].values else 0 for m in models_present]
baseline_errs = [baseline_data[baseline_data["target_model"] == m]["std_tps"].values[0]
                 if m in baseline_data["target_model"].values else 0 for m in models_present]
bar_colors_base = [COLORS["moe"] if "moe" in m else COLORS["dense"] for m in models_present]
bars1 = ax.bar(x - width, baseline_vals, width, yerr=baseline_errs, capsize=3,
               color=bar_colors_base, alpha=0.7, label="No SD (baseline)", edgecolor="white")

# SD with 0.8B draft, γ=8
sd08_data = agg[(agg["draft_model"] == "draft_0.8b") & (agg["draft_len"] == 8)]
sd08_vals = [sd08_data[sd08_data["target_model"] == m]["mean_tps"].values[0]
             if m in sd08_data["target_model"].values else 0 for m in models_present]
sd08_errs = [sd08_data[sd08_data["target_model"] == m]["std_tps"].values[0]
             if m in sd08_data["target_model"].values else 0 for m in models_present]
bars2 = ax.bar(x, sd08_vals, width, yerr=sd08_errs, capsize=3,
               color=COLORS["draft08"], alpha=0.8, label="SD + 0.8B draft (γ=8)", edgecolor="white")

# SD with 2B draft, γ=8
sd2b_data = agg[(agg["draft_model"] == "draft_2b") & (agg["draft_len"] == 8)]
sd2b_vals = [sd2b_data[sd2b_data["target_model"] == m]["mean_tps"].values[0]
             if m in sd2b_data["target_model"].values else 0 for m in models_present]
sd2b_errs = [sd2b_data[sd2b_data["target_model"] == m]["std_tps"].values[0]
             if m in sd2b_data["target_model"].values else 0 for m in models_present]
bars3 = ax.bar(x + width, sd2b_vals, width, yerr=sd2b_errs, capsize=3,
               color=COLORS["draft2b"], alpha=0.8, label="SD + 2B draft (γ=8)", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models_present],
                   rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Tokens / second", fontsize=12)
ax.set_title("Throughput: Baseline vs Speculative Decoding\n(γ=8, M2 Ultra)", fontsize=12, fontweight="bold")
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
fig.tight_layout()
fig.savefig(PLOT / "fig3_moe_vs_dense.png", bbox_inches="tight")
print("Saved fig3_moe_vs_dense.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: SD speedup vs prompt length  (memory-bound regime analysis)
# ═══════════════════════════════════════════════════════════════════════════════
sweep = df[df["prompt_key"] == "sweep"].copy()
if not sweep.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: throughput
    ax = axes[0]
    for has_sd, label, color, ls, marker in [
        (False, "No SD (baseline)", COLORS["baseline"], "--", "s"),
        (True,  "SD (0.8B draft, γ=8)", COLORS["sd"], "-", "o"),
    ]:
        grp = sweep[sweep["has_draft"] == has_sd].groupby("n_prompt")["tokens_per_sec"]
        means = grp.mean()
        stds = grp.std()
        ax.errorbar(means.index, means.values, yerr=stds.values, color=color, linestyle=ls,
                    marker=marker, label=label, linewidth=1.5, capsize=3)
    ax.set_xlabel("Prompt length (tokens)", fontsize=11)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Throughput vs Prompt Length", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)

    # Right: speedup ratio
    ax = axes[1]
    base_sweep = sweep[~sweep["has_draft"]].groupby("n_prompt")["tokens_per_sec"].mean()
    sd_sweep = sweep[sweep["has_draft"]].groupby("n_prompt")["tokens_per_sec"].mean()
    common = base_sweep.index.intersection(sd_sweep.index)
    if len(common) > 0:
        speedup = sd_sweep[common] / base_sweep[common]
        ax.plot(speedup.index, speedup.values, color=COLORS["sd"], marker="o",
                linewidth=2, markersize=8, label="SD speedup")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Prompt length (tokens)", fontsize=11)
        ax.set_ylabel("Speedup (SD / baseline)", fontsize=11)
        ax.set_title("SD Speedup vs Prompt Length", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("MoE 35B-A3B Q4 — Prompt Length Sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT / "fig4_prompt_sweep.png", bbox_inches="tight")
    print("Saved fig4_prompt_sweep.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Speedup by draft length γ — detailed breakdown
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, draft_id, draft_label in [
    (axes[0], "draft_0.8b", "Draft: Qwen3.5-0.8B"),
    (axes[1], "draft_2b",   "Draft: Qwen3.5-2B"),
]:
    sub = agg[(agg["draft_model"] == draft_id) & (agg["draft_len"] > 0)]
    if sub.empty:
        ax.set_title(f"No data for {draft_label}")
        continue
    for target_id in sub["target_model"].unique():
        grp = sub[sub["target_model"] == target_id].sort_values("draft_len")
        is_moe = "moe" in target_id
        color = COLORS["moe"] if is_moe else COLORS["dense"]
        marker = "*" if is_moe else "o"
        label = MODEL_LABELS.get(target_id, target_id).replace("\n", " ")
        ax.plot(grp["draft_len"], grp["mean_speedup"], marker=marker, color=color,
                label=label, linewidth=1.5, markersize=10 if marker == "*" else 6)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Draft length γ", fontsize=11)
    ax.set_ylabel("Speedup (SD / baseline)", fontsize=11)
    ax.set_title(draft_label, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=8)
    ax.set_xticks(DRAFT_LENGTHS)

fig.suptitle("Speedup vs Draft Length by Target Model", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT / "fig5_speedup_by_gamma.png", bbox_inches="tight")
print("Saved fig5_speedup_by_gamma.png")
plt.close()

# ── Print key numbers for the paper ──────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY RESULTS SUMMARY")
print("=" * 70)

print("\n--- Baseline Throughput (no SD) ---")
for mid in ["dense_0.8b", "dense_2b", "moe_q4", "moe_q8", "dense_4b", "dense_9b"]:
    row = agg[(agg["target_model"] == mid) & (agg["draft_model"] == "none")]
    if row.empty:
        continue
    r = row.iloc[0]
    print(f"  {MODEL_LABELS.get(mid, mid).replace(chr(10), ' '):35s}  "
          f"tps={r['mean_tps']:.1f} ± {r['std_tps']:.1f}")

print("\n--- SD with 0.8B Draft (γ=8) ---")
for mid in ["moe_q4", "moe_q8", "dense_4b", "dense_9b"]:
    row = agg[(agg["target_model"] == mid) & (agg["draft_model"] == "draft_0.8b") &
              (agg["draft_len"] == 8)]
    if row.empty:
        continue
    r = row.iloc[0]
    print(f"  {MODEL_LABELS.get(mid, mid).replace(chr(10), ' '):35s}  "
          f"speedup={r['mean_speedup']:.2f}x  "
          f"acceptance={r['mean_acceptance']:.2%}  "
          f"tps={r['mean_tps']:.1f}")

print("\n--- SD with 2B Draft (γ=8) ---")
for mid in ["moe_q4", "moe_q8"]:
    row = agg[(agg["target_model"] == mid) & (agg["draft_model"] == "draft_2b") &
              (agg["draft_len"] == 8)]
    if row.empty:
        continue
    r = row.iloc[0]
    print(f"  {MODEL_LABELS.get(mid, mid).replace(chr(10), ' '):35s}  "
          f"speedup={r['mean_speedup']:.2f}x  "
          f"acceptance={r['mean_acceptance']:.2%}  "
          f"tps={r['mean_tps']:.1f}")

print("=" * 70)
print(f"\nAll plots → {PLOT}")

# MoE vs Speculative Decoding — Experiment Suite
## Research question
When MoE active-parameter count is low (e.g. 3B active out of 35B total),
does Speculative Decoding still provide meaningful speedup?

---

## Prerequisites (one-time, do first)

```bash
# 1. Activate the shared Python env
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate

# 2. Install analysis deps (fast, no GPU needed)
pip install -r ~/speculative-moe-research/requirements.txt

# 3. Build llama.cpp if not already built
cd ~/Documents/mlx-community/llama.cpp
cmake -B build -DLLAMA_METAL=ON && cmake --build build -j$(sysctl -n hw.logicalcpu)
```

---

## Execution order

### Step 0 — Discover & verify (run once)
```bash
cd ~/speculative-moe-research
python3 00_discover.py
```
Checks binaries, finds all .gguf files, tests SSH to m1max + m2pro.
Writes `discovered_paths.json`. Fix any errors before proceeding.

**If SSH hostnames differ**, edit `config.py`:
```python
MACHINES = {
    "m1max": {"host": "YOUR_M1MAX_HOSTNAME.local", ...},
    "m2pro": {"host": "YOUR_M2PRO_HOSTNAME.local", ...},
}
```

---

### Step 1 — Run experiments

**Quick smoke test (10 min) — run Suite A only:**
```bash
python3 01_run_experiments.py --suites A
```

**Full experiment matrix (estimated ~4-6 hours total):**
```bash
# Recommended: run suites in parallel across machines
# Suite A+B on M2 Ultra (MoE experiments):
python3 01_run_experiments.py --suites A B

# Suite C+D on remotes (dense baselines) — open new terminal:
python3 01_run_experiments.py --suites C D

# Suite F last (prompt sweep):
python3 01_run_experiments.py --suites F
```

Results accumulate in `results/csv/runs.csv` in real time.
Raw llama.cpp output saved to `results/raw/<job_id>.txt`.

---

### Step 2 — Analyze & plot
```bash
python3 02_analyze.py
```
Produces:
- `results/csv/summary.csv` — aggregated table for the paper
- `results/plots/fig1_speedup_curve.png` — SD speedup vs active params (main result)
- `results/plots/fig2_acceptance_rate.png` — acceptance rate vs γ
- `results/plots/fig3_moe_vs_dense.png` — raw throughput comparison
- `results/plots/fig4_prompt_sweep.png` — speedup vs prompt length

You can re-run `02_analyze.py` at any time on partial data.

---

## Experiment matrix

| Suite | Target | Draft | Variable | Machine | Est. time |
|-------|--------|-------|----------|---------|-----------|
| A | MoE 35B Q4+Q8 | — | prompt len | M2 Ultra | 30 min |
| B | MoE 35B Q4+Q8 | 0.8B / 2B | γ ∈ {4,8,16} × 4 prompts | M2 Ultra | 2-3 hr |
| C | Dense 0.8B–9B | — | prompt len | M1MAX/M2Pro | 40 min |
| D | Dense 4B / 9B | 0.8B | γ ∈ {4,8,16} | M1MAX/M2Pro | 1 hr |
| F | MoE Q4 | 0.8B | prompt tokens {32…1024} | M2 Ultra | 30 min |

---

## Key files
```
speculative-moe-research/
├── config.py               ← edit hostnames / model paths here
├── 00_discover.py          ← environment check
├── 01_run_experiments.py   ← main runner
├── 02_analyze.py           ← analysis + plots
├── requirements.txt
├── discovered_paths.json   ← written by 00_discover.py
└── results/
    ├── csv/
    │   ├── runs.csv        ← live data during experiments
    │   └── summary.csv     ← aggregated (written by analyze)
    ├── raw/                ← raw llama.cpp output per job
    └── plots/              ← paper figures
```

---

## Interpreting results

The central claim of our paper is tested by **Fig 1**:
- If the MoE ★ point falls **below** the dense trend line → MoE's sparse activation
  reduces SD gains (supports the hypothesis).
- If it falls **on or above** the line → SD is still valuable for MoE despite low
  active params (rejects the hypothesis; suggests the mechanisms are orthogonal).

**Fig 4** shows whether the memory-bandwidth regime shift (short vs long prompt)
changes the conclusion — critical for determining when SD is beneficial.

# Does Speculative Decoding Help Mixture-of-Experts?

> **Work in Progress** — This research is ongoing. Results and conclusions may be updated as additional optimizations are implemented and validated.

*By [AtomGradient](https://github.com/AtomGradient)*

---
## Publication

- **Paper**: [Download PDF](https://atomgradient.github.io/speculative-moe-research/paper.pdf)
- **Website**: [https://atomgradient.github.io/speculative-moe-research/](https://atomgradient.github.io/speculative-moe-research/)

---

## Research Question

When MoE active-parameter count is low (e.g., 3B active out of 35B total), does Speculative Decoding still provide meaningful speedup?

**Answer:** Yes — SD provides **1.18–1.30× speedup** on MoE models despite very low acceptance rates (<4%), driven by batch verification efficiency rather than draft token acceptance.

---

## Key Results

| Target Model | Draft | γ | Throughput | Speedup | Acceptance |
|-------------|-------|---|-----------|---------|------------|
| **MoE Q4 (35B-A3B)** | 0.8B | 8 | 65.2 tok/s | **1.18×** | 1.1% |
| **MoE Q4 (35B-A3B)** | 0.8B | 16 | 69.7 tok/s | **1.26×** | 0.2% |
| **MoE Q8 (35B-A3B)** | 0.8B | 16 | 64.8 tok/s | **1.30×** | 0.2% |
| Dense 4B | 0.8B | 16 | 75.1 tok/s | 1.12× | 0.4% |
| Dense 9B | 0.8B | 16 | 67.7 tok/s | **2.03×** | 0.4% |

**Baseline throughput (no SD):** MoE Q4: 55.3 tok/s | MoE Q8: 49.9 tok/s | Dense 4B: 67.2 tok/s | Dense 9B: 33.4 tok/s

### Key Findings

1. **SD works for MoE** — 1.18–1.30× speedup despite <4% draft acceptance, through batch verification efficiency
2. **Total params drive SD benefit** — speedup scales with memory bandwidth (total params), not active params
3. **Smaller draft is better** — 0.8B draft outperforms 2B due to lower compute overhead
4. **Larger γ is optimal** — γ=16 gives best speedup despite lowest acceptance (batch cost grows sub-linearly)

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Chip | Apple M2 Ultra |
| Memory | 192 GB Unified (LPDDR5) |
| Bandwidth | 800 GB/s |
| Framework | llama.cpp v8240/8280 (Metal) |

---

## Reproduce

```bash
# 1. Activate Python env
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Discover models and verify environment
python3 00_discover.py

# 4. Run all experiments (~40 min on M2 Ultra)
python3 01_run_experiments.py --suites A B C D F

# 5. Analyze and generate plots
python3 02_analyze.py
```

## Experiment Matrix

| Suite | Description | Runs |
|-------|-------------|------|
| A | MoE baseline (no SD) | 18 |
| B | MoE + SD (2 drafts × 3 γ × 4 prompts) | 144 |
| C | Dense baselines (no SD) | 36 |
| D | Dense + SD (4B/9B + 0.8B draft) | 72 |
| F | Prompt-length sweep (MoE Q4) | 36 |
| **Total** | | **306** |

## Key Files

```
speculative-moe-research/
├── config.py               ← model paths, machine config
├── 00_discover.py          ← environment check
├── 01_run_experiments.py   ← experiment runner
├── 02_analyze.py           ← analysis + plots
├── requirements.txt
├── docs/
│   ├── paper.tex           ← LaTeX source
│   ├── paper.pdf           ← compiled paper
│   └── index.html          ← GitHub Pages site (bilingual)
└── results/
    ├── csv/
    │   ├── runs.csv        ← raw experiment data (306 rows)
    │   └── summary.csv     ← aggregated summary
    ├── raw/                ← raw llama.cpp output per job
    └── plots/              ← paper figures (5 plots)
```

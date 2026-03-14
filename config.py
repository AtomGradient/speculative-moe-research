"""
Central config for the MoE vs Speculative Decoding experiment.
Edit MACHINES if SSH hostnames differ on your network.
"""

import os

HOME = os.path.expanduser("~")
LLAMA_BIN = os.path.join(HOME, "llama.cpp")
MODEL_ROOT = HOME
RESULTS_DIR = os.path.join(HOME, "speculative-moe-research", "results")
PYTHON_ENV = os.path.join(
    HOME, "Documents/mlx-community/3-11-mlx-community-env/bin/activate"
)

# ── Machines ──────────────────────────────────────────────────────────────────
# "local" means run directly; others are SSH targets.
# Adjust hostnames to match your network (try `hostname` on each machine).
MACHINES = {
    "m2ultra": {"host": "192.168.0.103","name":"alex", "ram_gb": 192, "chip": "M2 Ultra"},
    "m1max":   {"host": "192.168.0.109","name":"alex",  "ram_gb": 32, "chip": "M1 MAX"},
    "m2pro":   {"host": "192.168.0.107","name":"atomgradient",  "ram_gb": 32, "chip": "M2 Pro"},
}

# ── Models ────────────────────────────────────────────────────────────────────
# Each entry: display name, directory glob, role, target machine, ~RAM (GB)
MODELS = [
    # ── MoE target (primary subject of study) ──────────────────────────────
    {
        "id":      "moe_q4",
        "label":   "Qwen3.5-35B-A3B Q4",
        "dir":     "Qwen3.5-35B-A3B-GGUF-UD-Q4_K_XL",
        "role":    "moe_target",
        "machine": "m2ultra",
        "ram_est": 20,
        "active_params_b": 3.0,
        "total_params_b": 35.0,
    },
    {
        "id":      "moe_q8",
        "label":   "Qwen3.5-35B-A3B Q8",
        "dir":     "Qwen3.5-35B-A3B-GGUF-UD-Q8_K_XL",
        "role":    "moe_target",
        "machine": "m2ultra",
        "ram_est": 38,
        "active_params_b": 3.0,
        "total_params_b": 35.0,
    },

    # ── Draft models (speculative decoding) ────────────────────────────────
    {
        "id":      "draft_0.8b",
        "label":   "Qwen3.5-0.8B Q8 (draft)",
        "dir":     "Qwen3.5-0.8B-GGUF-UD-Q8_K_XL",
        "role":    "draft",
        "machine": "m2ultra",
        "ram_est": 1,
        "active_params_b": 0.8,
        "total_params_b": 0.8,
    },
    {
        "id":      "draft_2b",
        "label":   "Qwen3.5-2B Q8 (draft)",
        "dir":     "Qwen3.5-2B-GGUF-UD-Q8_K_L",
        "role":    "draft",
        "machine": "m2ultra",
        "ram_est": 2,
        "active_params_b": 2.0,
        "total_params_b": 2.0,
    },

    # ── Dense baselines (same-family, varying active params) ───────────────
    {
        "id":      "dense_0.8b",
        "label":   "Qwen3.5-0.8B bf16",
        "dir":     "Qwen3.5-0.8B",
        "role":    "dense_baseline",
        "machine": "m2pro",
        "ram_est": 2,
        "active_params_b": 0.8,
        "total_params_b": 0.8,
    },
    {
        "id":      "dense_2b",
        "label":   "Qwen3.5-2B bf16",
        "dir":     "Qwen3.5-2B-bf16",
        "role":    "dense_baseline",
        "machine": "m2pro",
        "ram_est": 4,
        "active_params_b": 2.0,
        "total_params_b": 2.0,
    },
    {
        "id":      "dense_4b",
        "label":   "Qwen3.5-4B bf16",
        "dir":     "Qwen3.5-4B-bf16",
        "role":    "dense_baseline",
        "machine": "m1max",
        "ram_est": 8,
        "active_params_b": 4.0,
        "total_params_b": 4.0,
    },
    {
        "id":      "dense_9b",
        "label":   "Qwen3.5-9B bf16",
        "dir":     "Qwen3.5-9B",
        "role":    "dense_baseline",
        "machine": "m1max",
        "ram_est": 18,
        "active_params_b": 9.0,
        "total_params_b": 9.0,
    },

    # ── Cross-arch baseline ────────────────────────────────────────────────
    {
        "id":      "gemma_4b_q4",
        "label":   "Gemma-3-4B Q4",
        "dir":     "gemma-3-4b-it-qat-4bit",
        "role":    "cross_arch",
        "machine": "m2pro",
        "ram_est": 3,
        "active_params_b": 4.0,
        "total_params_b": 4.0,
    },
]

# ── Experiment matrix ─────────────────────────────────────────────────────────
DRAFT_LENGTHS    = [4, 8, 16]          # γ values to sweep
N_PREDICT        = 256                 # tokens to generate per run
N_PROMPT_TOKENS  = [64, 256, 512]      # prompt length sweep
BATCH_SIZES      = [1]                 # llama.cpp local = batch 1 effectively
TEMPERATURE      = 0.0                 # greedy; deterministic for reproducibility
N_REPEATS        = 3                   # repeat each config for variance

# ── Test prompts (representative, fixed for reproducibility) ─────────────────
PROMPTS = {
    "short_factual": "What is the capital of France?",
    "code_gen": (
        "Write a Python function that implements binary search on a sorted list. "
        "Include type hints and a docstring."
    ),
    "long_reasoning": (
        "Explain the difference between mixture-of-experts and dense transformer "
        "architectures in detail, covering routing mechanisms, training dynamics, "
        "and inference characteristics. Then discuss when each is preferable."
    ),
    "chinese": (
        "请详细解释大型语言模型中的混合专家（MoE）架构，包括门控机制、专家路由策略"
        "以及与密集模型相比的优缺点。"
    ),
}

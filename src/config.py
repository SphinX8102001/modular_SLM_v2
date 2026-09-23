"""src/config.py — static constants, paths, and model registry (round 1)."""

from __future__ import annotations
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
RESULTS_MOCK_DIR: Path = ROOT_DIR / "results_mock"
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
ARTIFACTS_MOCK_DIR: Path = ROOT_DIR / "artifacts_mock"
ROUTER_STATE_DIR: Path = ROOT_DIR / "router_state"
ROUTER_STATE_FILE: Path = ROUTER_STATE_DIR / "router_state.json"

# ──────────────────────────────────────────────────────────────────────────────
# Embedding + clustering hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
N_CLUSTERS: int = 3
DEFAULT_SEED: int = 42
MIN_CLUSTER_WARN: int = 10

# ──────────────────────────────────────────────────────────────────────────────
# Execution & formatting constants
# ──────────────────────────────────────────────────────────────────────────────
EXEC_TIMEOUT_SECONDS: int = 5
NUMERIC_SUFFIX: str = "\n\nPut your final answer within \\boxed{}."

# ──────────────────────────────────────────────────────────────────────────────
# Specialist roles
# ──────────────────────────────────────────────────────────────────────────────
ROLES: list[str] = ["general", "math", "code"]

# ──────────────────────────────────────────────────────────────────────────────
# Generation settings per role
# repetition_penalty on code generation is a candidate ablation.
# ──────────────────────────────────────────────────────────────────────────────
GEN_SETTINGS: dict[str, dict[str, Any]] = {
    "general": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "do_sample": False,
    },
    "math": {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "do_sample": False,
    },
    "code": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "do_sample": False,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Model registry — two parameter scales
# Note: baseline is the general specialist; no separate baseline model entry.
# ──────────────────────────────────────────────────────────────────────────────
SIZE_CONFOUND_NOTE: str = (
    "No Qwen2.5-Math variant below 1.5B exists, so any math-routing gain at "
    "0.5b scale is partly a parameter-size advantage."
)

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "1.5b": {
        "general": "Qwen/Qwen2.5-1.5B-Instruct",
        "math":    "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "code":    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    },
    "0.5b": {
        "general": "Qwen/Qwen2.5-0.5B-Instruct",
        # SIZE CONFOUND: no Qwen2.5-Math variant below 1.5B exists, so any
        # math-routing gain at 0.5b scale is partly a parameter-size advantage.
        "math":    "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "code":    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# System prompts — single source of truth for calibration AND test time.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "general": (
        "You are a helpful assistant. Answer the question concisely and accurately. "
        "For factual questions, give a short direct answer."
    ),
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "code": (
        "You are an expert Python programmer. Return ONLY a Python code block "
        "with no explanation, no markdown prose outside the code block."
    ),
}

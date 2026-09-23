"""src/config.py — static constants and model registry (skeleton, round 0)."""

# ──────────────────────────────────────────────────────────────────────────────
# Embedding + clustering hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
N_CLUSTERS: int = 3
DEFAULT_SEED: int = 42

# ──────────────────────────────────────────────────────────────────────────────
# Specialist roles
# ──────────────────────────────────────────────────────────────────────────────
ROLES: list[str] = ["general", "math", "code"]

# ──────────────────────────────────────────────────────────────────────────────
# Model registry — two parameter scales
# ──────────────────────────────────────────────────────────────────────────────
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
# Final wording to be filled in a later round.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "general": "PLACEHOLDER_general_prompt",
    "math":    "PLACEHOLDER_math_prompt",
    "code":    "PLACEHOLDER_code_prompt",
}

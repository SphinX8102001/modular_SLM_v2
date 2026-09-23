"""
Central configuration for Modular SLM pipeline.

ALL system prompts and generation settings are defined HERE as the SINGLE
SOURCE OF TRUTH.  Both calibration code and test-time code import from this
module — nothing is duplicated.  Any change to a prompt automatically applies
to both phases.
"""
from __future__ import annotations

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Output directories — strict mock / real separation ────────────────────────
# Mock runs write ONLY to *_mock directories; real runs write ONLY to the
# non-mock directories.  run_pipeline.py enforces this at startup.
RESULTS_DIR        = ROOT_DIR / "results"
RESULTS_MOCK_DIR   = ROOT_DIR / "results_mock"
ARTIFACTS_DIR      = ROOT_DIR / "artifacts"
ARTIFACTS_MOCK_DIR = ROOT_DIR / "artifacts_mock"
DATA_DIR           = ROOT_DIR / "data"

# ── Router persistence ─────────────────────────────────────────────────────────
ROUTER_STATE_DIR  = ROOT_DIR / "router_state"
ROUTER_STATE_FILE = ROUTER_STATE_DIR / "router_state.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPTS — SINGLE SOURCE OF TRUTH
# Used identically at calibration time AND test time via a single shared
# code path (ModelRunner.generate).  Do NOT copy these strings elsewhere.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM_PROMPTS: dict[str, str] = {
    "general": (
        "You are a helpful assistant. Answer the question concisely and accurately. "
        "For factual questions, give a short direct answer."
    ),
    "math": (
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    "code": (
        "You are an expert Python programmer. Return ONLY a Python code block "
        "with no explanation, no markdown prose outside the code block."
    ),
}

ROLES: list[str] = list(SYSTEM_PROMPTS.keys())  # ["general", "math", "code"]

# ── Model IDs per scale ────────────────────────────────────────────────────────
# 1.5b scale → all three specialists are 1.5 B (architecturally fair).
# 0.5b scale → math specialist MUST be 1.5B because no Qwen2.5-Math variant
#              exists below 1.5B.  This is a KNOWN SIZE CONFOUND; results at
#              0.5b scale carry a note and "size_confound": true in metadata.
MODELS: dict[str, dict[str, str]] = {
    "1.5b": {
        "general":  "Qwen/Qwen2.5-1.5B-Instruct",
        "math":     "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "code":     "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "baseline": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "0.5b": {
        "general":  "Qwen/Qwen2.5-0.5B-Instruct",
        # ⚠ SIZE CONFOUND: no Qwen2.5-Math below 1.5B exists.
        "math":     "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "code":     "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "baseline": "Qwen/Qwen2.5-0.5B-Instruct",
    },
}

SIZE_CONFOUND_NOTE = (
    "KNOWN SIZE CONFOUND (0.5b scale): The math specialist uses "
    "Qwen2.5-Math-1.5B-Instruct (1.5 B parameters) because no Qwen2.5-Math "
    "variant exists below 1.5 B.  Math-routing gains at 0.5b scale partly "
    "reflect a parameter-size advantage, not routing efficacy alone."
)

# ── Generation settings (per role) ────────────────────────────────────────────
# temperature=0.0 (greedy, reproducible), repetition_penalty=1.05.
# max_new_tokens: 256 for general/code, 512 for math (chain-of-thought).
GEN_SETTINGS: dict[str, dict] = {
    "general": {
        "max_new_tokens":      256,
        "temperature":         0.0,
        "repetition_penalty":  1.05,
        "do_sample":           False,
    },
    "math": {
        "max_new_tokens":      512,
        "temperature":         0.0,
        "repetition_penalty":  1.05,
        "do_sample":           False,
    },
    "code": {
        "max_new_tokens":      256,
        "temperature":         0.0,
        "repetition_penalty":  1.05,
        "do_sample":           False,
    },
}

# ── Router / embedding settings ───────────────────────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384
N_CLUSTERS       = 3
MIN_CLUSTER_WARN = 10   # log warning if any cluster < this many val items

# ── Numeric eval_type: suffix appended to ALL queries at inference time ───────
# Applied for ALL roles (general / math / code) when eval_type == "numeric",
# so scoring format is consistent regardless of which specialist answers.
NUMERIC_SUFFIX = "\n\nPut your final answer within \\boxed{}."

# ── python_exec subprocess timeout ────────────────────────────────────────────
EXEC_TIMEOUT_SECONDS = 5

# ── Benchmark (--benchmark real) ──────────────────────────────────────────────
BENCHMARK_SEED   = 42
BENCHMARK_VAL_N  = 30    # val items per category
BENCHMARK_TEST_N = 100   # test items per category

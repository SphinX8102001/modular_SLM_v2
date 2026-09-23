"""src/evaluator.py — evaluation function stubs (skeleton, round 0.5).

Design decisions recorded here:
- Baseline == scores["general"]: no extra generation needed per item.
- Tie-break: a specialist wins a cluster only if its score is STRICTLY greater
  than the generalist's; ties go to the generalist.
- Val and test sets must be strictly disjoint by id, asserted at startup.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Internal extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last balanced \\boxed{...} in *text*, or None."""
    raise NotImplementedError("round_0")


def safe_eval_arithmetic(expr: str) -> float | None:
    """Safely evaluate a plain arithmetic expression string; return None on error."""
    raise NotImplementedError("round_0")


def extract_number(text: str) -> float | None:
    """Ladder: try \\boxed{}, then arithmetic eval, then last bare number in *text*."""
    raise NotImplementedError("round_0")


# ──────────────────────────────────────────────────────────────────────────────
# Verifiers
# ──────────────────────────────────────────────────────────────────────────────

def verify_contains(response: str, answer: list[str]) -> bool:
    """Return True if any string in *answer* appears as a whole word in *response*."""
    raise NotImplementedError("round_0")


def verify_numeric(
    response: str,
    answer: float,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """Extract the last number from *response* and check it equals *answer* within tolerances."""
    raise NotImplementedError("round_0")


def verify_mcq_letter(response: str, answer: str) -> bool:
    """Return True if *response* selects the correct MCQ letter (A/B/C/D)."""
    raise NotImplementedError("round_0")


def verify_python_exec(
    response: str,
    test_assertions: str,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Execute Python code block from *response* against *test_assertions*; return (passed, error_message)."""
    raise NotImplementedError("round_0")


# ──────────────────────────────────────────────────────────────────────────────
# Scoring + metrics
# ──────────────────────────────────────────────────────────────────────────────

def score_response(item: dict, response: str) -> float:
    """Dispatch to the correct verify_* function; return 1.0 (pass) or 0.0 (fail)."""
    raise NotImplementedError("round_0")


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion."""
    raise NotImplementedError("round_0")


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant counts *b* and *c*."""
    raise NotImplementedError("round_0")


def compute_metrics(records: list[dict]) -> dict:
    """Aggregate per-item records into accuracy, Wilson CI, and McNemar p-value.

    Each record is a dict with keys:
        item             : the original dataset item dict
        routed_specialist: str, the role name chosen by the router
        scores           : dict with keys "general", "math", "code" → float (0 or 1)
        baseline_score   : float — defined as scores["general"] (no extra generation)

    Returns a summary dict (schema finalised in round 1).
    """
    raise NotImplementedError("round_0")

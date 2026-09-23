"""src/evaluator.py — evaluation function stubs (skeleton, round 0)."""

from __future__ import annotations
from typing import Any


def verify_contains(response: str, expected: list[str]) -> bool:
    """Return True if any expected string appears as a whole word in *response*."""
    raise NotImplementedError("round_0")


def verify_numeric(response: str, expected: float, tol: float = 1e-6) -> bool:
    """Extract the last number from *response* and check it equals *expected* ± *tol*."""
    raise NotImplementedError("round_0")


def verify_mcq_letter(response: str, expected: str) -> bool:
    """Return True if *response* selects the correct MCQ letter (A/B/C/D)."""
    raise NotImplementedError("round_0")


def verify_python_exec(response: str, test_code: str) -> bool:
    """Execute the Python code block in *response* against *test_code* assertions."""
    raise NotImplementedError("round_0")


def score_response(item: dict, response: str) -> float:
    """Dispatch to the correct verify_* function and return 1.0 (pass) or 0.0 (fail)."""
    raise NotImplementedError("round_0")


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion."""
    raise NotImplementedError("round_0")


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant counts *b* and *c*."""
    raise NotImplementedError("round_0")


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item results into accuracy, Wilson CI, and McNemar p-value."""
    raise NotImplementedError("round_0")

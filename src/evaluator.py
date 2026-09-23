"""
Evaluation logic for Modular SLM.

Verifiers
---------
verify_contains   — word-boundary regex; checks \\boxed{} span first, then full text.
verify_numeric    — full 5-rung extraction ladder with safe AST arithmetic evaluator.
verify_mcq_letter — case-insensitive prefix + case-SENSITIVE [A-D] letter.
verify_python_exec— subprocess runner with sentinel guard; sys.exit / os._exit safe.

Metrics
-------
compute_metrics(records) -> dict
    Computes all 12+ metrics defined in the spec, including Wilson 95% CI,
    exact two-sided McNemar test, oracle accuracy, random router accuracy,
    routing regret, and head-to-head win/loss/tie table.

Note: no eval() or exec() is used anywhere in this module.
"""
from __future__ import annotations

import ast
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from math import comb
from typing import Optional

log = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers: LaTeX / boxed extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_all_boxed(text: str) -> list[str]:
    """
    Extract the contents of ALL balanced \\boxed{...} spans in *text*.

    Returns a list of content strings (innermost brace content), in order of
    occurrence.  Supports nested braces.
    """
    results = []
    marker  = r"\boxed{"
    m_len   = len(marker)
    idx     = 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        depth = 0
        start = pos + m_len
        end   = None
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    end = i
                    break
                depth -= 1
        if end is not None:
            results.append(text[start:end])
        idx = pos + 1
    return results


def extract_last_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST balanced \\boxed{...} in *text*, or None."""
    all_boxed = _extract_all_boxed(text)
    return all_boxed[-1] if all_boxed else None


def _normalize_latex_subscripts(s: str) -> str:
    """Convert LaTeX subscripts to plain text: H_2O -> H2O, CO_2 -> CO2."""
    return re.sub(r"_\{?([^{}_ ]+)\}?", r"\1", s)


def _strip_latex_formatting(s: str) -> str:
    """
    Remove common LaTeX formatting from a string for contains-checking.
    Handles \\text{}, \\mathrm{}, \\mathbf{}, subscripts, spaces.
    """
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|boldsymbol)\{([^{}]*)\}", r"\1", s)
    s = _normalize_latex_subscripts(s)
    s = s.replace("~", " ").replace("\\,", " ").replace("\\;", " ")
    return s.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safe AST-based arithmetic evaluator (no eval())
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _eval_ast_node(node) -> Optional[float]:
    """Recursively evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):                    # Python < 3.8 compat
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left  = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return None if right == 0 else left / right
        if isinstance(node.op, ast.Pow):
            try:
                return float(left ** right)
            except (OverflowError, ZeroDivisionError):
                return None
        if isinstance(node.op, ast.FloorDiv):
            return None if right == 0 else float(left // right)
        if isinstance(node.op, ast.Mod):
            return None if right == 0 else float(left % right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
    return None


# Regex to expand \frac / \dfrac / \tfrac{a}{b} -> (a)/(b)
_FRAC_RE = re.compile(
    r"\\[dt]?frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
)
# Regex for \text{...}
_TEXT_RE = re.compile(r"\\text\{([^{}]*)\}")
# Regex for unit words and currency symbols that should be stripped
_UNIT_STRIP_RE = re.compile(
    r"\b(?:km|cm|mm|m|kg|g|lb|oz|ft|in|mi|L|mL|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"dollars?|euros?|pounds?|cents?|"
    r"people|persons?|workers?|students?|books?|apples?|items?|"
    r"meters?|litres?|gallons?|percent|pi)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """
    Safely evaluate a simple arithmetic expression (no eval()).

    Supports:
    - \\frac{a}{b}, \\dfrac, \\tfrac
    - \\text{...}
    - Comma thousands separators
    - Leading $ and % suffix (converts N% -> N/100)
    - Basic arithmetic: +, -, *, /, **
    """
    if not expr or not expr.strip():
        return None

    s = expr.strip()

    # Expand frac first (may be nested one level)
    for _ in range(3):
        s = _FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)

    # Strip \text{...} wrappers
    s = _TEXT_RE.sub(r"\1", s)

    # Remove LaTeX formatting noise
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)   # other \cmd{...}
    s = re.sub(r"\\[a-zA-Z]+", "", s)                   # bare \cmd

    # Handle percentage: 25% -> (25/100)
    s = _PERCENT_RE.sub(lambda m: f"({m.group(1)}/100)", s)

    # Remove dollar signs, unit words
    s = s.replace("$", "").replace(",", "")
    s = _UNIT_STRIP_RE.sub("", s)

    s = s.strip().rstrip(".")

    if not s:
        return None

    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None

    return _eval_ast_node(tree.body)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Numeric extraction ladder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Words that, when immediately following a number, suggest the number is a
# quantity descriptor rather than the final answer (rule 5 filter).
_UNIT_WORD_RE = re.compile(
    r"(?:-|sided|"
    r"km|cm|mm|m\b|kg|g\b|lb|oz|ft|in\b|mi\b|L\b|mL|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"books?|apples?|items?|people|persons?|workers?|students?|"
    r"meters?|litres?|gallons?|dollars?|euros?|cents?|"
    r"sides?|faces?|edges?|vertices?|steps?|times?|parts?|"
    r"coins?|cards?|balls?|boxes?|bags?|rows?|columns?)",
    re.IGNORECASE,
)

# Number pattern (integer, decimal, negative, with optional commas)
_NUM_PAT = r"-?\d[\d,]*(?:\.\d+)?"


def _parse_number(s: str) -> Optional[float]:
    """Try to parse *s* as a float, stripping commas and leading $."""
    s = s.strip().lstrip("$").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text: str) -> Optional[float]:
    """
    Full 5-rung extraction ladder, returns a float or None.

    Rung 1: LAST balanced \\boxed{...} — supports \\frac, \\text, commas, $, %.
    Rung 2: "#### N" (GSM8K style).
    Rung 3: "answer is N" / "final answer: N" / "result is N".
    Rung 4: Number after the LAST "=" in the text (handles step-by-step chains).
    Rung 5: Last number NOT immediately followed by a hyphen or unit word.
    """
    # ── Rung 1: last balanced \boxed{...} ─────────────────────────────────────
    boxed = extract_last_boxed(text)
    if boxed is not None:
        val = safe_eval_arithmetic(boxed)
        if val is not None:
            return val
        # Try parsing it directly as a number if safe_eval failed
        val = _parse_number(re.sub(r"[^\d.,$%-]", " ", boxed))
        if val is not None:
            return val

    # ── Rung 2: #### N  (GSM8K) ──────────────────────────────────────────────
    m = re.search(r"####\s*(\$?" + _NUM_PAT + r")", text)
    if m:
        val = _parse_number(m.group(1))
        if val is not None:
            return val

    # ── Rung 3: "answer is N" / "final answer: N" / "result is N" ────────────
    pattern3 = re.compile(
        r"(?:answer\s+is|final\s+answer\s*[:=]|result\s+is|the\s+answer\s+is)\s*"
        r"(\$?" + _NUM_PAT + r")",
        re.IGNORECASE,
    )
    m = pattern3.search(text)
    if m:
        val = _parse_number(m.group(1))
        if val is not None:
            return val

    # ── Rung 4: number after the LAST "=" ────────────────────────────────────
    # Find the last '=' and grab the number immediately following it.
    eq_matches = list(re.finditer(r"=\s*(\$?" + _NUM_PAT + r")", text))
    if eq_matches:
        val = _parse_number(eq_matches[-1].group(1))
        if val is not None:
            return val

    # ── Rung 5: last number not followed by hyphen or unit word ───────────────
    num_matches = list(re.finditer(r"(" + _NUM_PAT + r")", text))
    for m in reversed(num_matches):
        num_str  = m.group(1)
        tail_pos = m.end()
        tail     = text[tail_pos : tail_pos + 30]
        # Skip numbers immediately followed by "-" or a unit word
        if tail.lstrip() and _UNIT_WORD_RE.match(tail.lstrip()):
            continue
        if tail.startswith("-") and not tail.startswith("--"):
            continue
        val = _parse_number(num_str)
        if val is not None:
            return val

    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verifiers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _word_boundary_match(pattern: str, text: str) -> bool:
    """
    Return True if *pattern* appears in *text* as a whole word (or phrase),
    case-insensitively.

    "100" must NOT match inside "1000":
        (?<!\d)100(?!\d) prevents matching 100 inside 1000.
    For alphanumeric tokens the standard \\b is fine; for tokens that start/end
    with digits we use explicit lookbehind/ahead for digits.
    """
    escaped = re.escape(pattern.strip())
    # Use negative lookbehind/ahead for digits on both ends
    regex = r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])"
    return bool(re.search(regex, text, re.IGNORECASE))


def verify_contains(response: str, answer: list[str]) -> bool:
    """
    Return True if any acceptable answer string appears as a whole word/phrase
    in the response.

    Extraction order:
    1. Check inside the LAST balanced \\boxed{...} span (stripped of LaTeX).
    2. Fall back to the full response text.

    "100" must NOT match inside "1000" — implemented with digit-aware
    lookbehind/lookahead in _word_boundary_match.
    """
    # Normalise the response text
    def _normalise(s: str) -> str:
        return _strip_latex_formatting(s)

    # Candidate texts to search
    candidates: list[str] = []

    boxed = extract_last_boxed(response)
    if boxed is not None:
        candidates.append(_normalise(boxed))

    candidates.append(_normalise(response))

    for candidate in candidates:
        for ans in answer:
            if _word_boundary_match(_normalise(ans), candidate):
                return True
    return False


def verify_numeric(
    response: str,
    answer: float | int,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """
    Return True if the number extracted from *response* matches *answer*
    within tolerance.  Tolerance: max(abs_tol, rel_tol * |expected|).
    """
    extracted = extract_number(response)
    if extracted is None:
        return False
    expected  = float(answer)
    threshold = max(abs_tol, rel_tol * abs(expected))
    return abs(extracted - expected) <= threshold


def verify_mcq_letter(response: str, answer: str) -> bool:
    """
    Return True if the MCQ letter extracted from *response* matches *answer*.

    Extraction order (first match wins):
    1. Prefix "answer is X" or "option is X" — case-insensitive prefix,
       case-SENSITIVE [A-D] letter.  Prevents "answer is a good fit" -> A.
    2. \\boxed{X} where X ∈ {A,B,C,D}.
    3. (X) — parenthesised letter.
    4. "X." at the start of a line.
    5. Fallback: first standalone [A-D] letter (\\b[A-D]\\b).
       Note: this correctly grabs "B" from "definitely B" but does NOT
       extract "D" from inside "definitely" (no word boundary there).
    """
    answer = answer.strip().upper()

    # Pattern 1: "answer is X" — separate prefix match from letter match
    prefix_re = re.compile(
        r"(?:answer|option)\s+is\s+",
        re.IGNORECASE,
    )
    for m in prefix_re.finditer(response):
        rest        = response[m.end():]
        letter_m    = re.match(r"([A-D])\b", rest)   # case-SENSITIVE [A-D]
        if letter_m:
            return letter_m.group(1) == answer

    # Pattern 2: \boxed{X}
    for boxed in _extract_all_boxed(response):
        stripped = boxed.strip()
        if stripped.upper() in ("A", "B", "C", "D"):
            return stripped.upper() == answer

    # Pattern 3: (X) — parenthesised, word-bounded letter
    m = re.search(r"\(([A-D])\)", response)
    if m:
        return m.group(1) == answer

    # Pattern 4: X. at the start of a line
    m = re.search(r"^([A-D])\.\s", response, re.MULTILINE)
    if m:
        return m.group(1) == answer

    # Pattern 5: fallback — first standalone A-D letter
    m = re.search(r"\b([A-D])\b", response)
    if m:
        return m.group(1) == answer

    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Python exec verifier — subprocess only, no exec() / eval()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SUCCESS_SENTINEL = "__MODULAR_SLM_EXEC_OK__"


def _extract_python_code(response: str) -> Optional[str]:
    """
    Extract Python code from *response*.

    Priority:
    1. Content of the first ```python ... ``` fence.
    2. Content of the first ``` ... ``` fence.
    3. Everything starting from the first 'def ' or 'class ' line.
    """
    # ```python fence
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    # generic fence
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    # bare def / class
    m = re.search(r"^(def |class )", response, re.MULTILINE)
    if m:
        return response[m.start():]
    return None


def verify_python_exec(
    response: str,
    test_assertions: str,
    timeout: int = 5,
) -> tuple[bool, str]:
    """
    Extract Python code from *response*, run it plus *test_assertions* in a
    subprocess, and return (passed, error_message).

    Sentinel guard
    --------------
    The sentinel ``__MODULAR_SLM_EXEC_OK__`` is printed ONLY after the
    assertions complete.  This means:
    - If model code calls sys.exit(0) before assertions:
        exit-code=0 but sentinel absent → FAIL ✓
    - If model code calls os._exit(0) before assertions:
        subprocess terminates immediately → sentinel absent → FAIL ✓
    - If assertions raise AssertionError:
        non-zero exit code, no sentinel → FAIL ✓
    - If assertions pass and code runs to completion:
        sentinel printed, exit code 0 → PASS ✓

    No eval() or exec() is used anywhere; the subprocess runs with its own
    Python interpreter under a hard wall-clock timeout.
    """
    code = _extract_python_code(response)
    if code is None:
        return False, "No Python code found in response."

    # Build the script: model code first, then assertions, then sentinel.
    script = textwrap.dedent(f"""\
        {code}

        # ── Test assertions ──────────────────────────────────────────────────
        {test_assertions}

        # Sentinel: only reached if assertions completed without exception
        print("{_SUCCESS_SENTINEL}", flush=True)
    """)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s (possible infinite loop)."
    except Exception as exc:
        return False, f"Subprocess error: {exc}"

    passed = (
        result.returncode == 0
        and _SUCCESS_SENTINEL in result.stdout
    )
    if not passed:
        err = (result.stderr or result.stdout or "").strip()
        return False, err[:500]   # truncate long tracebacks
    return True, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified score_response dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_response(item: dict, response: str) -> float:
    """
    Score *response* against the ground truth in *item*.

    Returns 1.0 (correct) or 0.0 (incorrect).

    item fields used:
        eval_type : "contains" | "numeric" | "mcq_letter" | "python_exec"
        answer    : depends on eval_type
        test_assertions : str (only for python_exec)
    """
    eval_type = item.get("eval_type", "contains")

    if eval_type == "contains":
        return 1.0 if verify_contains(response, item["answer"]) else 0.0

    if eval_type == "numeric":
        return 1.0 if verify_numeric(response, item["answer"]) else 0.0

    if eval_type == "mcq_letter":
        return 1.0 if verify_mcq_letter(response, item["answer"]) else 0.0

    if eval_type == "python_exec":
        passed, _ = verify_python_exec(
            response,
            item.get("test_assertions", ""),
            timeout=item.get("exec_timeout", 5),
        )
        return 1.0 if passed else 0.0

    raise ValueError(f"Unknown eval_type: {eval_type!r}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statistical helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    95% Wilson score confidence interval.

    Returns (lower, upper) clipped to [0, 1].
    """
    if total == 0:
        return (0.0, 1.0)
    p    = successes / total
    z2   = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """
    Exact two-sided McNemar test p-value.

    b = cases where router correct, baseline incorrect.
    c = cases where router incorrect, baseline correct.

    Under H₀: B | (B+C=n) ~ Binomial(n, 0.5).
    Two-sided p-value = 2 * P(B ≤ min(b,c)) capped at 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    m     = min(b, c)
    # P(B <= m) = sum_{k=0}^{m} C(n,k) / 2^n
    p_le  = sum(comb(n, k) for k in range(m + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_le)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SPECIALISTS = ("general", "math", "code")


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute all metrics from *records*.

    Each record must contain:
        item              : the test item dict (with "category", "eval_type", etc.)
        routed_specialist : str — the specialist chosen by the router
        scores            : dict{specialist -> 0.0|1.0} for all three specialists
        baseline_score    : float (0 or 1) — generalist alone

    Returns a dict with all metrics.
    """
    if not records:
        return {
            "n_total":                        0,
            "task_accuracy_router":           0.0,
            "task_accuracy_baseline":         0.0,
            "delta_vs_baseline":              0.0,
            "oracle_accuracy":                0.0,
            "random_router_accuracy":         0.0,
            "routing_regret":                 0.0,
            "category_routing_accuracy":      {},
            "per_specialist_accuracy":        {},
            "head_to_head":                   {},
            "router_ci_95":                   (0.0, 1.0),
            "baseline_ci_95":                 (0.0, 1.0),
            "mcnemar_pvalue":                 1.0,
            "domain_breakdown":               {},
        }

    n = len(records)

    # Per-item scalar scorings
    router_correct   = []
    baseline_correct = []

    for rec in records:
        spec    = rec["routed_specialist"]
        r_score = rec["scores"][spec]
        b_score = rec["baseline_score"]
        router_correct.append(r_score)
        baseline_correct.append(b_score)

    router_sum   = sum(router_correct)
    baseline_sum = sum(baseline_correct)

    # ── Oracle (at least one specialist correct) ───────────────────────────────
    oracle_correct = [
        1.0 if any(rec["scores"][s] >= 1.0 for s in _SPECIALISTS) else 0.0
        for rec in records
    ]
    oracle_sum = sum(oracle_correct)

    # ── Random router (mean per-query fraction of specialists correct) ─────────
    random_scores = [
        sum(rec["scores"][s] for s in _SPECIALISTS) / len(_SPECIALISTS)
        for rec in records
    ]
    random_sum = sum(random_scores)

    # ── Routing regret (router wrong but some specialist would have been right) ─
    regret = sum(
        1 for i, rec in enumerate(records)
        if router_correct[i] < 1.0 and oracle_correct[i] >= 1.0
    )

    # ── Per-specialist accuracy ────────────────────────────────────────────────
    per_specialist: dict[str, float] = {}
    for spec in _SPECIALISTS:
        spec_sum = sum(rec["scores"][spec] for rec in records)
        per_specialist[spec] = spec_sum / n

    # ── Category routing accuracy (did router pick the "right" domain expert?) ─
    # "Right" domain expert = specialist whose name matches item["category"]
    # (general/math/code).  This is kept separate from task accuracy.
    cat_routing: dict[str, list] = {}
    for rec in records:
        cat = rec["item"]["category"]
        cat_routing.setdefault(cat, [])
        # routing correct if routed_specialist matches the item's category
        cat_routing[cat].append(
            1.0 if rec["routed_specialist"] == cat else 0.0
        )

    category_routing_accuracy = {
        cat: sum(v) / len(v) for cat, v in cat_routing.items()
    }

    # ── Head-to-head ──────────────────────────────────────────────────────────
    router_right_baseline_wrong = sum(
        1 for i in range(n)
        if router_correct[i] >= 1.0 and baseline_correct[i] < 1.0
    )
    router_wrong_baseline_right = sum(
        1 for i in range(n)
        if router_correct[i] < 1.0 and baseline_correct[i] >= 1.0
    )
    both_right = sum(
        1 for i in range(n)
        if router_correct[i] >= 1.0 and baseline_correct[i] >= 1.0
    )
    both_wrong = sum(
        1 for i in range(n)
        if router_correct[i] < 1.0 and baseline_correct[i] < 1.0
    )

    # ── McNemar ───────────────────────────────────────────────────────────────
    b = router_right_baseline_wrong
    c = router_wrong_baseline_right
    mcnemar_p = mcnemar_exact_pvalue(b, c)

    # ── Wilson CIs ────────────────────────────────────────────────────────────
    router_ci   = wilson_ci(int(round(router_sum)),   n)
    baseline_ci = wilson_ci(int(round(baseline_sum)), n)

    # ── Domain-level breakdown table ──────────────────────────────────────────
    domain_breakdown: dict[str, dict] = {}
    for cat in sorted(cat_routing.keys()):
        cat_records = [rec for rec in records if rec["item"]["category"] == cat]
        n_cat = len(cat_records)
        r_cat = sum(rec["scores"][rec["routed_specialist"]] for rec in cat_records)
        b_cat = sum(rec["baseline_score"]                   for rec in cat_records)
        domain_breakdown[cat] = {
            "n":                      n_cat,
            "routing_accuracy":       round(category_routing_accuracy.get(cat, 0.0), 4),
            "router_task_accuracy":   round(r_cat / n_cat, 4) if n_cat else 0.0,
            "baseline_task_accuracy": round(b_cat / n_cat, 4) if n_cat else 0.0,
        }

    return {
        "n_total":                   n,
        "task_accuracy_router":      round(router_sum   / n, 4),
        "task_accuracy_baseline":    round(baseline_sum / n, 4),
        "delta_vs_baseline":         round((router_sum - baseline_sum) / n, 4),
        "oracle_accuracy":           round(oracle_sum  / n, 4),
        "random_router_accuracy":    round(random_sum  / n, 4),
        "routing_regret":            round(regret       / n, 4),
        "category_routing_accuracy": {k: round(v, 4) for k, v in category_routing_accuracy.items()},
        "per_specialist_accuracy":   {k: round(v, 4) for k, v in per_specialist.items()},
        "head_to_head": {
            "router_right_baseline_wrong": b,
            "router_wrong_baseline_right": c,
            "both_right":                  both_right,
            "both_wrong":                  both_wrong,
        },
        "router_ci_95":              router_ci,
        "baseline_ci_95":            baseline_ci,
        "mcnemar_pvalue":            round(mcnemar_p, 6),
        "domain_breakdown":          domain_breakdown,
    }

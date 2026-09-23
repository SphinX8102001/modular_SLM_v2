"""src/evaluator.py — evaluation functions and metrics (round 2).

Design decisions recorded here:
- Baseline == scores["general"]: no extra generation needed per item.
- Tie-break: a specialist wins a cluster only if its score is STRICTLY greater
  than the generalist's; ties go to the generalist.
- Val and test sets must be strictly disjoint by id, asserted at startup.
"""

from __future__ import annotations
import ast
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Any
import uuid


# ──────────────────────────────────────────────────────────────────────────────
# Internal extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_last_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} in *text*, or None.

    Handles nested braces by brace counting. Returns None if absent or unbalanced.
    """
    marker = r"\boxed{"
    matches: list[str] = []
    i = 0
    while True:
        pos = text.find(marker, i)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        for j in range(start, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start:j])
                    break
        i = pos + len(marker)
    return matches[-1] if matches else None


def safe_eval_arithmetic(expr: str) -> float | None:
    """Safely evaluate a plain arithmetic expression string; return None on error.

    Allows only numeric constants and binary operators (+ - * / // % **) with unary +/-.
    Rejects names, calls, attributes, and strings.
    Guards against exponents with abs(exponent) > 1000 and expressions longer than 200 chars.
    """
    if not isinstance(expr, str) or len(expr) > 200 or not expr.strip():
        return None

    try:
        parsed = ast.parse(expr.strip(), mode="eval")
    except Exception:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return float(node.value)
            raise ValueError("Non-numeric constant")
        elif isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            elif isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Unsupported unary operator")
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            if isinstance(node.op, ast.Pow):
                right = _eval(node.right)
                if abs(right) > 1000:
                    raise ValueError("Exponent too large")
                return float(left ** right)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return float(left + right)
            elif isinstance(node.op, ast.Sub):
                return float(left - right)
            elif isinstance(node.op, ast.Mult):
                return float(left * right)
            elif isinstance(node.op, ast.Div):
                return float(left / right)
            elif isinstance(node.op, ast.FloorDiv):
                return float(left // right)
            elif isinstance(node.op, ast.Mod):
                return float(left % right)
            raise ValueError("Unsupported binary operator")
        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        res = _eval(parsed)
        if math.isnan(res) or math.isinf(res):
            return None
        return float(res)
    except Exception:
        return None


def extract_number(text: str) -> float | None:
    """Ladder: try \\boxed{}, then arithmetic eval, then last bare number in *text*."""
    # (a) Try last \boxed{} content
    boxed = extract_last_boxed(text)
    if boxed is not None:
        s = boxed.strip().strip("$").strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        s = s.replace(",", "")
        s = re.sub(r"\\d?frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"(\1)/(\2)", s)
        try:
            return float(s)
        except ValueError:
            pass
        val = safe_eval_arithmetic(s)
        if val is not None:
            return val

    # (b) Last bare number in text
    pattern = re.compile(
        r"(?<!\w)[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?!\w)"
    )
    matches = pattern.findall(text)
    if matches:
        last_match = matches[-1].replace(",", "")
        try:
            return float(last_match)
        except ValueError:
            return None

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Verifiers
# ──────────────────────────────────────────────────────────────────────────────

def verify_contains(response: str, answer: list[str]) -> bool:
    """Return True if any string in *answer* appears as a whole word in *response*."""
    if not isinstance(response, str) or not isinstance(answer, (list, tuple)):
        return False
    for item in answer:
        if not item:
            continue
        pattern = rf"(?<!\w){re.escape(item)}(?!\w)"
        if re.search(pattern, response, flags=re.IGNORECASE):
            return True
    return False


def verify_numeric(
    response: str,
    answer: float,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> bool:
    """Extract the last number from *response* and check it equals *answer* within tolerances."""
    num = extract_number(response)
    if num is None:
        return False
    try:
        return math.isclose(num, answer, rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def verify_mcq_letter(response: str, answer: str) -> bool:
    """Return True if *response* selects the correct MCQ letter (A/B/C/D).

    Ladder, first hit wins:
    (a) \\boxed{X}
    (b) "answer is X" / "answer: X" / "correct option is X"
    (c) whole stripped response is just the letter (e.g. "B", "(B)", "B.", "B)")
    """
    target = answer.strip().upper()

    # (a) \boxed{X}
    boxed = extract_last_boxed(response)
    if boxed is not None:
        m = re.search(r"\(?([A-Da-d])\)?", boxed)
        if m:
            return m.group(1).upper() == target

    # (b) Specific key phrases
    phrase_pattern = re.compile(
        r"(?:answer\s*(?:is|:)|correct\s+option\s*(?:is|:))\s*[*_`]*\(?([A-Da-d])\)?\.?[*_`]*",
        re.IGNORECASE,
    )
    phrase_matches = list(phrase_pattern.finditer(response))
    if phrase_matches:
        return phrase_matches[-1].group(1).upper() == target

    # (c) Whole stripped response
    stripped = response.strip()
    m_full = re.fullmatch(r"[*_`]*[([<{]?\s*([A-Da-d])\s*[)\]>.}]?[*_`]*", stripped)
    if m_full:
        return m_full.group(1).upper() == target

    return False


def verify_python_exec(
    response: str,
    test_assertions: str,
    timeout: int = 5,
) -> tuple[bool, str]:
    """Execute Python code block from *response* against *test_assertions*; return (passed, error_message).

    Note: This is NOT a security sandbox; only run trusted model outputs on a research machine.
    """
    pattern = re.compile(r"```(?:python|py)?\s*\r?\n?(.*?)\r?\n?```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(response)
    if not matches:
        return False, "no code block"

    code = matches[-1].strip()
    sentinel = uuid.uuid4().hex
    script = code + "\n\n" + test_assertions + "\nprint(" + repr(sentinel) + ")\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "solution.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

    stdout_lines = proc.stdout.strip().splitlines()
    last_line = stdout_lines[-1].strip() if stdout_lines else ""
    if proc.returncode == 0 and last_line == sentinel:
        return True, ""

    err_msg = proc.stderr[-500:] if proc.stderr else "execution failed: sentinel not reached"
    return False, err_msg


# ──────────────────────────────────────────────────────────────────────────────
# Scoring + metrics
# ──────────────────────────────────────────────────────────────────────────────

def score_response(item: dict, response: str) -> float:
    """Dispatch to the correct verify_* function; return 1.0 (pass) or 0.0 (fail).

    Item schema:
        id: str
        category: str ("general", "math", "code", etc.)
        question: str
        verifier: str, in {"contains", "numeric", "mcq_letter", "python_exec"}
        answer: list[str] (for contains), float/int (for numeric), str (for mcq_letter)
        test_assertions: str (for python_exec)
        rel_tol: float, optional (for numeric)
        abs_tol: float, optional (for numeric)
    """
    verifier = item.get("verifier")
    if verifier == "contains":
        passed = verify_contains(response, item["answer"])
    elif verifier == "numeric":
        rel_tol = item.get("rel_tol", 1e-3)
        abs_tol = item.get("abs_tol", 1e-6)
        passed = verify_numeric(response, float(item["answer"]), rel_tol=rel_tol, abs_tol=abs_tol)
    elif verifier == "mcq_letter":
        passed = verify_mcq_letter(response, str(item["answer"]))
    elif verifier == "python_exec":
        assertions = item.get("test_assertions", "")
        passed, _ = verify_python_exec(response, assertions)
    else:
        raise ValueError(f"Unknown verifier: {verifier!r}")

    return 1.0 if passed else 0.0


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion, clamped to [0, 1]."""
    if total == 0:
        return 0.0, 1.0
    if not (0 <= successes <= total):
        raise ValueError(f"successes ({successes}) must be in [0, total ({total})]")

    p = successes / total
    denom = 1 + (z ** 2) / total
    center = (p + (z ** 2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p)) / total + (z ** 2) / (4 * (total ** 2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant counts *b* and *c*."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob_sum = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * prob_sum)


def compute_metrics(records: list[dict]) -> dict:
    """Aggregate per-item records into accuracy, Wilson CI, and McNemar p-value.

    Note that when routed == general the routed and baseline scores are identical
    by construction, so those items are always concordant.
    Do NOT compute any category routing accuracy metric.

    Each record is a dict with keys:
        item             : the original dataset item dict
        routed_specialist: str, must be in ["general", "math", "code"]
        scores           : dict with keys "general", "math", "code" -> float
        baseline_score   : float, must equal scores["general"]

    Returns a dict with:
        n, routed, baseline, always, oracle, delta, mcnemar, routing_counts, per_category.
    """
    if not records:
        raise ValueError("Records list cannot be empty")

    valid_specialists = {"general", "math", "code"}
    n = len(records)

    # Validate records
    for r in records:
        specialist = r.get("routed_specialist")
        if specialist not in valid_specialists:
            raise ValueError(f"Invalid routed_specialist: {specialist!r}")
        scores = r.get("scores", {})
        baseline_score = r.get("baseline_score")
        if baseline_score != scores.get("general"):
            raise ValueError("baseline_score must equal scores['general']")

    def _calc_stats(k: int, total_n: int) -> dict[str, Any]:
        acc = k / total_n if total_n > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(k, total_n)
        return {"k": k, "n": total_n, "acc": acc, "ci_lo": ci_lo, "ci_hi": ci_hi}

    routed_k = sum(1 for r in records if r["scores"][r["routed_specialist"]] == 1.0)
    baseline_k = sum(1 for r in records if r["baseline_score"] == 1.0)

    always_counts: dict[str, int] = {
        role: sum(1 for r in records if r["scores"].get(role) == 1.0)
        for role in ["general", "math", "code"]
    }
    always: dict[str, dict[str, Any]] = {
        role: _calc_stats(always_counts[role], n)
        for role in ["general", "math", "code"]
    }

    oracle_k = sum(
        1
        for r in records
        if max(r["scores"].get("general", 0.0), r["scores"].get("math", 0.0), r["scores"].get("code", 0.0)) == 1.0
    )
    oracle = {"k": oracle_k, "n": n, "acc": oracle_k / n if n > 0 else 0.0}

    routed_stats = _calc_stats(routed_k, n)
    baseline_stats = _calc_stats(baseline_k, n)
    delta = routed_stats["acc"] - baseline_stats["acc"]

    # McNemar discordant counts
    b = 0
    c = 0
    routing_counts = {"general": 0, "math": 0, "code": 0}
    per_category_counts: dict[str, dict[str, Any]] = {}

    for r in records:
        spec = r["routed_specialist"]
        routing_counts[spec] += 1

        r_score = r["scores"][spec]
        b_score = r["baseline_score"]

        r_correct = (r_score == 1.0)
        b_correct = (b_score == 1.0)

        if r_correct and not b_correct:
            b += 1
        elif not r_correct and b_correct:
            c += 1

        cat = r.get("item", {}).get("category", "unknown")
        if cat not in per_category_counts:
            per_category_counts[cat] = {"n": 0, "routed_k": 0, "baseline_k": 0}
        per_category_counts[cat]["n"] += 1
        if r_correct:
            per_category_counts[cat]["routed_k"] += 1
        if b_correct:
            per_category_counts[cat]["baseline_k"] += 1

    mcnemar = {"b": b, "c": c, "p": mcnemar_exact_pvalue(b, c)}

    per_category = {
        cat: {
            "n": data["n"],
            "routed_acc": data["routed_k"] / data["n"] if data["n"] > 0 else 0.0,
            "baseline_acc": data["baseline_k"] / data["n"] if data["n"] > 0 else 0.0,
        }
        for cat, data in per_category_counts.items()
    }

    return {
        "n": n,
        "routed": routed_stats,
        "baseline": baseline_stats,
        "always": always,
        "oracle": oracle,
        "delta": delta,
        "mcnemar": mcnemar,
        "routing_counts": routing_counts,
        "per_category": per_category,
    }

"""
run_pipeline.py — Modular SLM Research Pipeline CLI

Usage examples
--------------
# Smoke test with mock runner (no weights downloaded):
  python run_pipeline.py --mock --scale 1.5b --benchmark smoke

# Real HuggingFace run, 1.5B scale, 10-query stratified subsample:
  python run_pipeline.py --scale 1.5b --backend huggingface --limit 10

# Reuse a previously calibrated router:
  python run_pipeline.py --scale 1.5b --skip-calibration

# Real benchmark data (GSM8K / MBPP / MMLU):
  python run_pipeline.py --mock --scale 1.5b --benchmark real

Self-check invariants (verified at startup):
  - Val / test ID sets are strictly disjoint.
  - Mock mode can only write to results_mock/ and artifacts_mock/.
  - Tie-break: specialist wins cluster only if score STRICTLY > generalist.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.config import (
    MODELS, GEN_SETTINGS, SYSTEM_PROMPTS, ROLES, NUMERIC_SUFFIX,
    RESULTS_DIR, RESULTS_MOCK_DIR, ARTIFACTS_DIR, ARTIFACTS_MOCK_DIR,
    DATA_DIR, ROUTER_STATE_FILE,
    N_CLUSTERS, EMBEDDING_MODEL, MIN_CLUSTER_WARN, EXEC_TIMEOUT_SECONDS,
    BENCHMARK_SEED, BENCHMARK_VAL_N, BENCHMARK_TEST_N,
    SIZE_CONFOUND_NOTE,
)
from src.models import get_runner, MockRunner, _MOCK_NOTICE
from src.router import EmbeddingRouter
from src.evaluator import score_response, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _pkg_ver(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _set_seeds(seed: int) -> None:
    """Apply seed to random, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_run_metadata(args: argparse.Namespace, is_mock: bool, scale: str) -> dict:
    """Build the run_metadata dict embedded in every results JSON."""
    model_ids = MODELS[scale]
    return {
        "is_mock":          is_mock,
        "scale":            scale,
        "backend":          args.backend,
        "seed":             args.seed,
        "benchmark":        args.benchmark,
        "utc_timestamp":    datetime.now(timezone.utc).isoformat(),
        "git_commit_short": _get_git_hash(),
        "model_ids":        model_ids,
        "system_prompts":   SYSTEM_PROMPTS,   # single source — config.py
        "gen_settings":     GEN_SETTINGS,
        "package_versions": {
            "torch":                _pkg_ver("torch"),
            "transformers":         _pkg_ver("transformers"),
            "sentence_transformers":_pkg_ver("sentence-transformers"),
            "sklearn":              _pkg_ver("scikit-learn"),
            "numpy":                _pkg_ver("numpy"),
        },
        "size_confound": scale == "0.5b",
        "size_confound_note": SIZE_CONFOUND_NOTE if scale == "0.5b" else None,
        "notice": _MOCK_NOTICE if is_mock else None,
    }


def _get_output_dirs(is_mock: bool) -> tuple[Path, Path]:
    """Return (results_dir, artifacts_dir) enforcing mock/real separation."""
    if is_mock:
        return RESULTS_MOCK_DIR, ARTIFACTS_MOCK_DIR
    return RESULTS_DIR, ARTIFACTS_DIR


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_handwritten_data() -> tuple[list[dict], list[dict]]:
    val_path  = DATA_DIR / "val_set.json"
    test_path = DATA_DIR / "test_set.json"
    val_items  = json.loads(val_path.read_text())
    test_items = json.loads(test_path.read_text())
    return val_items, test_items


def _load_benchmark_data(seed: int) -> tuple[list[dict], list[dict]]:
    """
    Pull GSM8K (math), MBPP (code), MMLU (general) via HuggingFace datasets.
    Shuffle with fixed seed; take first BENCHMARK_VAL_N as val, next
    BENCHMARK_TEST_N as test per category.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "HuggingFace 'datasets' library not installed. "
            "Install with: pip install datasets"
        )
        sys.exit(1)

    rng = random.Random(seed)

    def _shuffle_take(items, n_val, n_test):
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled[:n_val], shuffled[n_val : n_val + n_test]

    val_items: list[dict]  = []
    test_items: list[dict] = []

    # ── GSM8K → math / numeric ────────────────────────────────────────────────
    log.info("Loading GSM8K …")
    gsm8k = load_dataset("gsm8k", "main", split="train")
    gsm_all = [
        {
            "id":        f"bm_math_{i:04d}",
            "category":  "math",
            "eval_type": "numeric",
            "query":     row["question"],
            # GSM8K answers end with "#### N"
            "answer":    float(row["answer"].split("####")[-1].strip().replace(",", "")),
        }
        for i, row in enumerate(gsm8k)
    ]
    gsm_val, gsm_test = _shuffle_take(gsm_all, BENCHMARK_VAL_N, BENCHMARK_TEST_N)
    val_items  += gsm_val
    test_items += gsm_test

    # ── MBPP → code / python_exec ─────────────────────────────────────────────
    log.info("Loading MBPP …")
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    mbpp_all = [
        {
            "id":              f"bm_code_{i:04d}",
            "category":        "code",
            "eval_type":       "python_exec",
            "query":           row["text"],
            "answer":          None,
            "test_assertions": "\n".join(row["test_list"]),
        }
        for i, row in enumerate(mbpp)
    ]
    mbpp_val, mbpp_test = _shuffle_take(mbpp_all, BENCHMARK_VAL_N, BENCHMARK_TEST_N)
    val_items  += mbpp_val
    test_items += mbpp_test

    # ── MMLU → general / mcq_letter ───────────────────────────────────────────
    log.info("Loading MMLU …")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    mmlu_all = [
        {
            "id":        f"bm_general_{i:04d}",
            "category":  "general",
            "eval_type": "mcq_letter",
            "query": (
                row["question"] + "\n" +
                "\n".join(
                    f"({letter_map[j]}) {ch}"
                    for j, ch in enumerate(row["choices"])
                )
            ),
            "answer": letter_map[row["answer"]],
        }
        for i, row in enumerate(mmlu)
    ]
    mmlu_val, mmlu_test = _shuffle_take(mmlu_all, BENCHMARK_VAL_N, BENCHMARK_TEST_N)
    val_items  += mmlu_val
    test_items += mmlu_test

    return val_items, test_items


def _assert_val_test_disjoint(val_items: list[dict], test_items: list[dict]) -> None:
    """
    Programmatic assertion that val and test sets share no IDs.
    Runs at pipeline startup — not just by convention.
    """
    val_ids  = {it["id"] for it in val_items}
    test_ids = {it["id"] for it in test_items}
    overlap  = val_ids & test_ids
    if overlap:
        raise AssertionError(
            f"CRITICAL: val and test sets share {len(overlap)} ID(s): "
            f"{sorted(overlap)}.  These sets must be strictly disjoint."
        )
    log.info("Val/test disjointness check: OK (%d val, %d test, 0 overlap).",
             len(val_ids), len(test_ids))


def _stratified_subsample(
    items: list[dict], n: int, seed: int
) -> list[dict]:
    """
    Return a stratified subsample of *n* items, equal per category,
    deterministic given *seed*.
    """
    by_category: dict[str, list[dict]] = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)

    categories = sorted(by_category.keys())
    per_cat    = n // len(categories)
    rng        = random.Random(seed)

    result: list[dict] = []
    for cat in categories:
        pool = list(by_category[cat])
        rng.shuffle(pool)
        result.extend(pool[:per_cat])

    # Deterministic order
    return sorted(result, key=lambda x: x["id"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _prepare_query(item: dict, role: str) -> str:
    """
    Return the query text to send to the model for *role*.

    Appends NUMERIC_SUFFIX for ALL roles when eval_type == "numeric",
    ensuring consistent answer format regardless of which specialist answers.
    """
    q = item["query"]
    if item.get("eval_type") == "numeric":
        q = q + NUMERIC_SUFFIX
    return q


def _run_all_specialists(
    item: dict,
    runners: dict,   # role -> ModelRunner
    is_mock: bool,
) -> dict[str, str]:
    """
    Run ALL three specialists on *item* and return their raw text responses.

    Responses are cached — this function is called once per item; the cached
    results are reused for oracle, random router, per-specialist, and routing
    accuracy computations.
    """
    responses: dict[str, str] = {}
    for role in ROLES:
        query    = _prepare_query(item, role)
        runner   = runners[role]
        log.debug("  [%s] Running %s …", item["id"], role)
        responses[role] = runner.generate(query, role)
    return responses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calibrate_router(
    val_items: list[dict],
    runners: dict,
    seed: int,
) -> EmbeddingRouter:
    """
    Calibrate the router on the validation set.

    For each val item we run all three specialists (cached in memory) and
    score them.  The scores feed the capability profile matrix.
    """
    log.info("=== Calibration: running all specialists on %d val items ===", len(val_items))

    # Cache all responses
    val_responses: dict[str, dict[str, str]] = {}   # item_id -> role -> response
    for item in val_items:
        log.info("  [%s] Embedding + running specialists …", item["id"])
        val_responses[item["id"]] = _run_all_specialists(item, runners, is_mock=False)

    # Score function for the router calibration
    def score_fn(item: dict, specialist: str) -> float:
        resp = val_responses[item["id"]][specialist]
        return score_response(item, resp)

    router = EmbeddingRouter(
        embedding_model_name = EMBEDDING_MODEL,
        n_clusters           = N_CLUSTERS,
        seed                 = seed,
    )
    diagnostics = router.calibrate(
        val_items        = val_items,
        score_fn         = score_fn,
        min_cluster_warn = MIN_CLUSTER_WARN,
    )
    log.info("Calibration diagnostics: %s", json.dumps(diagnostics, indent=2))
    return router


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.time()

    # ── 0. Seed everything ────────────────────────────────────────────────────
    _set_seeds(args.seed)

    is_mock = args.mock
    scale   = args.scale
    results_dir, artifacts_dir = _get_output_dirs(is_mock)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Enforce mock path isolation — double-check even if _get_output_dirs is correct
    if is_mock:
        MockRunner.assert_mock_path(str(results_dir))
        MockRunner.assert_mock_path(str(artifacts_dir))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.benchmark == "real":
        log.info("Loading real benchmark data (GSM8K / MBPP / MMLU) …")
        val_items, test_items = _load_benchmark_data(seed=BENCHMARK_SEED)
    else:
        log.info("Loading hand-written data …")
        val_items, test_items = _load_handwritten_data()

    # ── 1a. Val / test disjointness assertion (ALWAYS runs at startup) ────────
    _assert_val_test_disjoint(val_items, test_items)

    # ── 2. Apply --limit ──────────────────────────────────────────────────────
    if args.limit is not None:
        log.info("--limit %d: stratified subsample of test set.", args.limit)
        test_items = _stratified_subsample(test_items, args.limit, seed=args.seed)
        log.info("Test set reduced to %d items.", len(test_items))

    # ── 3. Build runners ──────────────────────────────────────────────────────
    model_ids = MODELS[scale]
    runners: dict[str, object] = {}
    for role in ROLES:
        runners[role] = get_runner(
            backend  = args.backend,
            model_id = model_ids[role],
            role     = role,
            scale    = scale,
            is_mock  = is_mock,
        )
    # Baseline runner (= general specialist for generalist model)
    baseline_runner = get_runner(
        backend  = args.backend,
        model_id = model_ids["baseline"],
        role     = "baseline",
        scale    = scale,
        is_mock  = is_mock,
    )

    # ── 4. Calibrate or load router ───────────────────────────────────────────
    if args.skip_calibration:
        log.info("--skip-calibration: loading router from %s", ROUTER_STATE_FILE)
        if not ROUTER_STATE_FILE.exists():
            log.error("No saved router state at %s.  Run without --skip-calibration first.",
                      ROUTER_STATE_FILE)
            sys.exit(1)
        router = EmbeddingRouter.load(ROUTER_STATE_FILE)
    else:
        router = calibrate_router(val_items, runners, seed=args.seed)
        router.save(ROUTER_STATE_FILE)
        log.info("Router saved to %s", ROUTER_STATE_FILE)

    # ── 5. Evaluate on test set ───────────────────────────────────────────────
    log.info("=== Evaluation: %d test items ===", len(test_items))
    records: list[dict] = []

    for item in test_items:
        log.info("  [%s] %s …", item["id"], item["query"][:60])

        # Run all three specialists — cached in memory, never re-run
        specialist_responses = _run_all_specialists(item, runners, is_mock)

        # Baseline response (generalist with "general" role prompt)
        baseline_query    = _prepare_query(item, "general")
        baseline_response = baseline_runner.generate(baseline_query, "general")

        # Score all specialists
        scores: dict[str, float] = {
            role: score_response(item, specialist_responses[role])
            for role in ROLES
        }
        baseline_score = score_response(item, baseline_response)

        # Route
        routed_specialist, cluster_idx = router.route_with_cluster(item["query"])

        records.append({
            "item":               item,
            "routed_specialist":  routed_specialist,
            "cluster_idx":        cluster_idx,
            "scores":             scores,
            "baseline_score":     baseline_score,
            "responses":          specialist_responses,
            "baseline_response":  baseline_response,
        })

        log.info(
            "    routed=%s  scores=%s  baseline=%.0f",
            routed_specialist,
            {k: int(v) for k, v in scores.items()},
            baseline_score,
        )

    # ── 6. Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(records)

    # ── 7. Build and save results ─────────────────────────────────────────────
    run_metadata = _build_run_metadata(args, is_mock, scale)
    run_metadata["elapsed_seconds"] = round(time.time() - t_start, 1)

    # Serialisable records (strip large response text for summary JSON)
    summary_records = [
        {
            "id":               r["item"]["id"],
            "category":         r["item"]["category"],
            "routed_specialist":r["routed_specialist"],
            "cluster_idx":      r["cluster_idx"],
            "scores":           r["scores"],
            "baseline_score":   r["baseline_score"],
        }
        for r in records
    ]

    results_payload = {
        "run_metadata": run_metadata,
        "metrics":      metrics,
        "records":      summary_records,
    }

    # Detailed payload with raw responses (for inspection)
    detailed_payload = {
        "run_metadata": run_metadata,
        "metrics":      metrics,
        "records": [
            {
                **sr,
                "query":             r["item"]["query"],
                "responses":         r["responses"],
                "baseline_response": r["baseline_response"],
            }
            for sr, r in zip(summary_records, records)
        ],
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"mock_{scale}" if is_mock else scale
    results_file  = results_dir  / f"results_{tag}_{ts}.json"
    detailed_file = artifacts_dir / f"detailed_{tag}_{ts}.json"

    results_file.write_text(json.dumps(results_payload, indent=2))
    detailed_file.write_text(json.dumps(detailed_payload, indent=2))

    log.info("Results saved to %s", results_file)
    log.info("Detailed results saved to %s", detailed_file)

    # ── 8. Print summary ──────────────────────────────────────────────────────
    _print_summary(metrics, scale, is_mock)


def _print_summary(metrics: dict, scale: str, is_mock: bool) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MODULAR SLM RESULTS  |  scale={scale}  |  mock={is_mock}")
    print(sep)
    print(f"  N total:                {metrics['n_total']}")
    print(f"  Router accuracy:        {metrics['task_accuracy_router']:.3f}  "
          f"95% CI {metrics['router_ci_95'][0]:.3f}–{metrics['router_ci_95'][1]:.3f}")
    print(f"  Baseline accuracy:      {metrics['task_accuracy_baseline']:.3f}  "
          f"95% CI {metrics['baseline_ci_95'][0]:.3f}–{metrics['baseline_ci_95'][1]:.3f}")
    print(f"  Δ vs baseline:          {metrics['delta_vs_baseline']:+.3f}")
    print(f"  Oracle accuracy:        {metrics['oracle_accuracy']:.3f}")
    print(f"  Random router:          {metrics['random_router_accuracy']:.3f}")
    print(f"  Routing regret:         {metrics['routing_regret']:.3f}")
    print(f"  McNemar p-value:        {metrics['mcnemar_pvalue']:.4f}")
    print()
    print("  Head-to-head:")
    h = metrics["head_to_head"]
    print(f"    Router ✓ / Baseline ✗: {h['router_right_baseline_wrong']}")
    print(f"    Router ✗ / Baseline ✓: {h['router_wrong_baseline_right']}")
    print(f"    Both ✓:                {h['both_right']}")
    print(f"    Both ✗:                {h['both_wrong']}")
    print()
    print("  Per-specialist accuracy:")
    for sp, acc in metrics["per_specialist_accuracy"].items():
        print(f"    {sp:10s}: {acc:.3f}")
    print()
    print("  Domain breakdown:")
    for cat, row in metrics["domain_breakdown"].items():
        print(f"    {cat:10s}: N={row['n']:3d}  "
              f"route_acc={row['routing_accuracy']:.2f}  "
              f"router={row['router_task_accuracy']:.2f}  "
              f"baseline={row['baseline_task_accuracy']:.2f}")
    if is_mock:
        print(f"\n  ⚠  MOCK RUN — all scores near zero is expected.")
    print(sep + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modular SLM — training-free routing research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scale", choices=["1.5b", "0.5b"], default="1.5b",
        help="Specialist model scale to test.",
    )
    p.add_argument(
        "--backend", choices=["huggingface", "ollama"], default="huggingface",
        help="Inference backend.",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use MockRunner (no weights).  Writes to results_mock/ only.",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stratified subsample test set to N items (equal per category).",
    )
    p.add_argument(
        "--skip-calibration", action="store_true",
        help="Load a previously saved router state instead of recalibrating.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility.",
    )
    p.add_argument(
        "--benchmark", choices=["smoke", "real"], default="smoke",
        help=(
            "smoke: use hand-written data/val_set.json + data/test_set.json.  "
            "real: pull GSM8K/MBPP/MMLU via HuggingFace datasets."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Fail fast for 0.5b scale — surface the confound note early
    if args.scale == "0.5b":
        log.warning(SIZE_CONFOUND_NOTE)

    run_pipeline(args)

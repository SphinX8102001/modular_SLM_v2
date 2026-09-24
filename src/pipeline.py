"""src/pipeline.py — end-to-end routing pipeline execution (round 5).

Design decisions:
- Role-major generation: for each role, load runner → generate all needed items
  → unload runner. Only one model is in memory at a time.
- Resumable via JSONL generation cache (<run_dir>/generation_cache.jsonl).
  Re-run the same command to resume; delete the file to force a fresh run.
- score_fn never triggers new generations after the pre-generation phase.
- Preflight check (non-mock): validates torch/transformers availability before
  any model is loaded or the embedding model is touched.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

import src
from src import cache as cache_module
from src import config, data, evaluator, models, router


def hash_embed(texts: list[str], dim: int | None = None) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding for mock runs.

    Lowercase \\w+ tokens (add a fixed "<empty>" token if none), stable hash via
    hashlib.md5 -> index = int % dim, sign from another byte, sum, L2-normalise, float64.
    """
    if dim is None:
        dim = config.EMBEDDING_DIM

    if not texts:
        return np.empty((0, dim), dtype=np.float64)

    vectors: list[np.ndarray] = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            tokens = ["<empty>"]
        v = np.zeros(dim, dtype=np.float64)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], byteorder="little") % dim
            sign = 1.0 if (digest[8] % 2 == 0) else -1.0
            v[idx] += sign
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        else:
            v[0] = 1.0
        vectors.append(v)

    return np.array(vectors, dtype=np.float64)


def _preflight(args: Any) -> int | None:
    """Non-mock preflight checks; return exit code on failure, None on success."""
    if getattr(args, "mock", False):
        return None

    backend = getattr(args, "backend", "huggingface")

    if backend == "ollama":
        sys.stderr.write(
            "Ollama backend is not implemented yet (arrives in round 7). "
            "Re-run with --backend huggingface.\n"
        )
        return 2

    if backend == "huggingface":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            sys.stderr.write(
                f"Missing dependency: {e}. "
                "Install dependencies with: pip install -r requirements.txt\n"
            )
            return 2

    return None


def run(
    args: Any,
    runner_factory: Callable[..., models.ModelRunner] | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> int:
    """Run the entire routing pipeline end-to-end and return the exit code."""
    try:
        # Configure defaults
        if runner_factory is None:
            runner_factory = models.get_runner

        if embed_fn is None and getattr(args, "mock", False):
            embed_fn = hash_embed

        device = getattr(args, "device", "cpu")

        # Step a: Seed random and numpy
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Step b: Output directories
        if args.mock:
            results_dir = config.RESULTS_MOCK_DIR
            artifacts_dir = config.ARTIFACTS_MOCK_DIR
            state_path = artifacts_dir / "router_state.json"
        else:
            results_dir = config.RESULTS_DIR
            artifacts_dir = config.ARTIFACTS_DIR
            state_path = config.ROUTER_STATE_FILE

        run_dir = results_dir / f"{args.scale}_{args.benchmark}_seed{args.seed}"

        if args.mock:
            models.MockRunner.assert_mock_path(str(run_dir))
            models.MockRunner.assert_mock_path(str(state_path))

        # Step c: Load datasets
        val_path, test_path = data.dataset_paths(args.benchmark)
        val_items = data.load_items(val_path)
        test_items = data.load_items(test_path)
        data.assert_disjoint(val_items, test_items)

        # val_limit handling (debugging only)
        val_limit = getattr(args, "val_limit", None)
        if val_limit is not None:
            if val_limit < config.N_CLUSTERS:
                sys.stderr.write(
                    f"--val-limit {val_limit} is less than N_CLUSTERS={config.N_CLUSTERS}; "
                    "the router cannot be calibrated with so few items.\n"
                )
                return 2
            val_items = val_items[:val_limit]

        if getattr(args, "limit", None) is not None:
            test_items = test_items[: args.limit]

        # Step d: Build runners
        runners: dict[str, models.ModelRunner] = {}
        for role in config.ROLES:
            model_id = config.MODEL_REGISTRY[args.scale][role]
            runners[role] = runner_factory(
                args.backend,
                model_id,
                role,
                args.scale,
                args.mock,
                device,
            )

        # Preflight (runs after dataset loading and runner creation, before embed/model)
        preflight_code = _preflight(args)
        if preflight_code is not None:
            return preflight_code

        # Initialise the persistent generation cache
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_path = run_dir / "generation_cache.jsonl"
        if args.mock:
            models.MockRunner.assert_mock_path(str(cache_path))
        gen_cache = cache_module.GenerationCache(cache_path)

        # In-memory memo: (item_id, role) -> response
        response_memo: dict[tuple[str, str], str] = {}

        # Per-role timing stats
        timing: dict[str, dict[str, Any]] = {
            role: {"n_generated": 0, "n_cached": 0, "seconds": 0.0, "new_tokens": 0, "_has_tokens": False}
            for role in config.ROLES
        }

        val_id_set = {str(it["id"]) for it in val_items}
        generations: dict[str, dict[str, dict[str, str]]] = {"val": {}, "test": {}}

        def generate_response(runner: models.ModelRunner, item: dict, role: str) -> str:
            """Cached generation: check mem cache → disk cache → generate."""
            item_id = str(item["id"])
            mem_key = (item_id, role)
            if mem_key in response_memo:
                return response_memo[mem_key]

            system_prompt = config.SYSTEM_PROMPTS[role]
            gen_settings = models.build_generation_kwargs(role)
            query = data.build_query(item)
            cache_key = cache_module.make_key(runner, role, system_prompt, gen_settings, query)

            cached = gen_cache.get(cache_key)
            if cached is not None:
                timing[role]["n_cached"] += 1
                response_memo[mem_key] = cached
                split = "val" if item_id in val_id_set else "test"
                if item_id not in generations[split]:
                    generations[split][item_id] = {}
                generations[split][item_id][role] = cached
                return cached

            t0 = time.perf_counter()
            response = runner.generate(query, role)
            elapsed = time.perf_counter() - t0

            timing[role]["n_generated"] += 1
            timing[role]["seconds"] += elapsed
            new_toks = getattr(runner, "last_new_tokens", None)
            if new_toks is not None:
                timing[role]["new_tokens"] += new_toks
                timing[role]["_has_tokens"] = True

            gen_cache.put(cache_key, response)
            response_memo[mem_key] = response

            split = "val" if item_id in val_id_set else "test"
            if item_id not in generations[split]:
                generations[split][item_id] = {}
            generations[split][item_id][role] = response
            return response

        # Score cache (item_id, role) -> float
        score_cache: dict[tuple[str, str], float] = {}

        def score_fn(item: dict, role: str) -> float:
            item_id = str(item["id"])
            key = (item_id, role)
            if key in score_cache:
                return score_cache[key]
            response = generate_response(runners[role], item, role)
            sc = evaluator.score_response(item, response)
            score_cache[key] = sc
            return sc

        # Step e: Role-major pre-generation phase
        skip_calibration = getattr(args, "skip_calibration", False)
        items_to_generate = val_items + test_items if not skip_calibration else test_items

        for role in config.ROLES:
            runner = runners[role]
            try:
                runner.load()
                for item in items_to_generate:
                    generate_response(runner, item, role)
            except RuntimeError as e:
                sys.stderr.write(
                    f"RuntimeError loading/running {role} runner: {e}. "
                    "Try --device cpu or a smaller --scale.\n"
                )
                return 2
            finally:
                runner.unload()

        # Step f: Router calibration or loading
        if not skip_calibration:
            r = router.EmbeddingRouter(
                embedding_model_name=config.EMBEDDING_MODEL,
                n_clusters=config.N_CLUSTERS,
                seed=args.seed,
            )
            if embed_fn is not None:
                r._embed = embed_fn
            calibration_summary = r.calibrate(
                val_items,
                score_fn=score_fn,
                min_cluster_warn=config.MIN_CLUSTER_WARN,
            )
            if args.mock:
                models.MockRunner.assert_mock_path(str(state_path))
            r.save(state_path)
        else:
            if not state_path.is_file():
                sys.stderr.write(f"Router state file not found: {state_path}\n")
                return 2
            r = router.EmbeddingRouter.load(state_path)
            if embed_fn is not None:
                r._embed = embed_fn
            calibration_summary = {"skipped": True}

        # Step g: Test phase
        records: list[dict[str, Any]] = []
        for item in test_items:
            routed_role = r.route(item["question"])
            scores = {role: score_fn(item, role) for role in config.ROLES}
            record = {
                "item": item,
                "routed_specialist": routed_role,
                "scores": scores,
                "baseline_score": scores["general"],
            }
            records.append(record)

        metrics = evaluator.compute_metrics(records)

        # Step h: Write outputs into run_dir
        if args.mock:
            models.MockRunner.assert_mock_path(str(run_dir))
            for fn in ("metrics.json", "records.json", "calibration.json", "generations.json",
                       "run_config.json", "timing.json"):
                models.MockRunner.assert_mock_path(str(run_dir / fn))

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(run_dir / "records.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        with open(run_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration_summary, f, indent=2)

        with open(run_dir / "generations.json", "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2)

        # Build timing.json
        timing_out: dict[str, dict[str, Any]] = {}
        for role, t in timing.items():
            n_gen = t["n_generated"]
            n_cached = t["n_cached"]
            seconds = t["seconds"]
            new_toks = t["new_tokens"] if t["_has_tokens"] else None

            sec_per_gen = (seconds / n_gen) if n_gen > 0 else None
            tokens_per_sec = (new_toks / seconds) if (new_toks and seconds > 0) else None

            timing_out[role] = {
                "n_generated": n_gen,
                "n_cached": n_cached,
                "seconds": seconds,
                "sec_per_generation": sec_per_gen,
                "new_tokens": new_toks,
                "tokens_per_sec": tokens_per_sec,
            }

        with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing_out, f, indent=2)

        # Build run_config.json
        git_hash = None
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(config.ROOT_DIR),
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                git_hash = res.stdout.strip()
        except Exception:
            git_hash = None

        embedding_name = (
            "hash_embed"
            if (embed_fn is not None or getattr(args, "mock", False))
            else config.EMBEDDING_MODEL
        )

        # Collect runner descriptions after generation phase
        runner_descriptions = {role: runners[role].describe() for role in config.ROLES}

        args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        run_config: dict[str, Any] = {
            "args": args_dict,
            "models": config.MODEL_REGISTRY[args.scale],
            "system_prompts": config.SYSTEM_PROMPTS,
            "gen_settings": config.GEN_SETTINGS,
            "n_val": len(val_items),
            "n_test": len(test_items),
            "mock": bool(args.mock),
            "embedding": embedding_name,
            "version": src.__version__,
            "git_commit": git_hash,
            "python_version": sys.version,
            "device": device,
            "val_limit": val_limit,
            "runner_describe": runner_descriptions,
        }
        if args.scale == "0.5b":
            run_config["size_confound_note"] = config.SIZE_CONFOUND_NOTE

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        # Step i: Print short summary and return 0
        routed_acc = metrics["routed"]["acc"]
        baseline_acc = metrics["baseline"]["acc"]
        delta = metrics["delta"]
        mcnemar_p = metrics["mcnemar"]["p"]
        oracle_acc = metrics["oracle"]["acc"]
        routing_counts = metrics["routing_counts"]

        total_cached = sum(t["n_cached"] for t in timing.values())
        total_generated = sum(t["n_generated"] for t in timing.values())

        print(
            f"Run summary ({args.scale}, benchmark={args.benchmark}, mock={args.mock}):\n"
            f"  Routed accuracy   : {routed_acc:.4f}\n"
            f"  Baseline accuracy : {baseline_acc:.4f}\n"
            f"  Delta (routed - bl): {delta:+.4f}\n"
            f"  McNemar p-value   : {mcnemar_p:.4f}\n"
            f"  Oracle accuracy   : {oracle_acc:.4f}\n"
            f"  Routing counts    : {routing_counts}\n"
            f"  Generations (new/cached): {total_generated}/{total_cached}\n"
            f"  Run directory     : {run_dir}"
        )
        return 0

    except FileNotFoundError as e:
        if getattr(args, "benchmark", "") == "real":
            sys.stderr.write(
                f"Data file not found: {e}. Real benchmark data arrives in a later round when benchmark == 'real'.\n"
            )
        else:
            sys.stderr.write(f"Data file not found: {e}\n")
        return 2
    except ValueError as e:
        sys.stderr.write(f"Validation error: {e}\n")
        return 2

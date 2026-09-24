# Mdular SLM

> **Status**: Round 5: mock pipeline and real HuggingFace runner.
> Inspired by [arXiv:2505.19797](https://arxiv.org/abs/2505.19797).

## Research Question

Does embedding-clustering-capability-profiling routing scale down effectively to sub-2B models on CPU?

## Architecture (planned, training-free)

1. Embed query with `sentence-transformers/all-MiniLM-L6-v2`
2. Cluster with Spherical K-Means (K=3)
3. Look up best specialist per cluster via Capability Profile Matrix
4. Dispatch to winning Qwen2.5 specialist model

## Models under test

| Scale | General | Math | Code |
|-------|---------|------|------|
| 1.5B  | Qwen2.5-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct |
| 0.5B  | Qwen2.5-0.5B-Instruct | Qwen2.5-Math-1.5B-Instruct* | Qwen2.5-Coder-0.5B-Instruct |

\* SIZE CONFOUND: no Qwen2.5-Math variant below 1.5B exists.

### Model Download Sizes

When running real HuggingFace models (`--backend huggingface`), model weights are downloaded from the Hugging Face Hub on the first run:
- **0.5B models**: roughly 1 GB for each model (`Qwen2.5-0.5B-Instruct`, `Qwen2.5-Coder-0.5B-Instruct`).
- **1.5B models**: roughly 3 GB for each model (`Qwen2.5-1.5B-Instruct`, `Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-Coder-1.5B-Instruct`).

## Design Decisions

- **Baseline = general specialist.** `baseline_score` is defined as `scores["general"]`; no separate baseline runner or extra generation per item is needed.
- **Tie-break rule.** A specialist wins a cluster only if its calibration score is *strictly* greater than the generalist's; ties go to the generalist.
- **score_fn callback.** Router calibration is decoupled from model runners via a `score_fn(item, response) → float` callback, so it can be unit-tested without loading any model weights.
- **Val/test disjointness.** The pipeline asserts that val-set and test-set ids are strictly disjoint at startup (not enforced in the router).
- **Separated output dirs.** Mock runs write only to `results_mock/` and `artifacts_mock/`; real runs write to `results/` and `artifacts/`.
- **Known limitation.** K=3 clusters need not align with the 3 task domains; this is a research hypothesis, not a guarantee.

## Usage

### Mock pipeline (no model downloads, CPU-only)

Run the mock smoke pipeline end-to-end (fast, zero downloads):

```bash
python run_pipeline.py --scale 0.5b --mock --seed 7
```

### Real HuggingFace pipeline

Run with real Qwen2.5 models via HuggingFace transformers:

```bash
# CPU run (default)
python run_pipeline.py --scale 0.5b --backend huggingface --benchmark smoke --seed 7

# CUDA GPU run
python run_pipeline.py --scale 0.5b --backend huggingface --benchmark smoke --device cuda --seed 7

# Fast debug run with validation limit
python run_pipeline.py --scale 0.5b --backend huggingface --benchmark smoke --val-limit 5 --seed 7
```

### CLI Flags

- `--device cpu|cuda`: Target execution hardware for model generation. Default is `cpu`. When `cuda` is specified, HuggingFace runners load model weights in `float16` precision onto GPU (`cpu` uses `float32`).
- `--val-limit N` (debugging only): Truncates validation items to `N` (must be $\ge$ 3) to test calibration and data flow quickly without running the full validation split.

### Memory & Execution: Role-Major Generation

Generation proceeds role-major (`general` -> `math` -> `code`):
- **One model in memory at a time**: For each role, the specialist runner is loaded once (`runner.load()`), generates all needed validation and test items, and is immediately unloaded (`runner.unload()`).
- This guarantees only one model is loaded in memory at any point, preventing out-of-memory errors on commodity CPUs and consumer GPUs.

### Resumable Generation Cache

- Every raw generation is saved to `<run dir>/generation_cache.jsonl`, keyed by `(model_id, prompt_hash)`.
- **Resuming**: If a run is interrupted or fails, rerun the exact same command to resume; cached responses are loaded without querying the model again.
- **Fresh run**: Delete `<run dir>/generation_cache.jsonl` to force a fresh run.

### Run Directory Naming

- Mock runs write to `results_mock/` and `artifacts_mock/`.
- Real runs write to `results/` and `artifacts/`.
- Directory name format:
  - CPU and default runs: `<scale>_<benchmark>_seed<seed>` (e.g. `results_mock/0.5b_smoke_seed7` or `results/0.5b_smoke_seed7`)
  - CUDA runs (`--device cuda`): `<scale>_<benchmark>_seed<seed>_cuda` (e.g. `results/0.5b_smoke_seed7_cuda`)

### Output Files

Outputs are written to `<run dir>` (under `results_mock/` or `results/`) and `artifacts_mock/` or `artifacts/`:

| Path / File | Description |
|-------------|-------------|
| `metrics.json` | Aggregated evaluation metrics: routed accuracy, baseline accuracy, delta, McNemar test p-value, Wilson CI, oracle accuracy, and routing counts. |
| `records.json` | Per-test-item evaluation records: question item, routed specialist, per-role scores, and baseline score. |
| `calibration.json` | Router calibration summary from the validation set (or `{"skipped": true}` when `--skip-calibration` is set). |
| `generations.json` | Dump of raw text generations per item and role across val and test splits (`{"val": {...}, "test": {...}}`). |
| `generation_cache.jsonl` | Append-only persistent JSONL generation cache for resuming interrupted runs. |
| `timing.json` | Generation benchmarks per role, including `sec_per_generation` (average latency per generation) and `tokens_per_sec` (throughput). |
| `run_config.json` | Metadata and configuration of the run (CLI arguments, model registry entries, prompts, seed, version, git commit, python version, and size confound note). |
| `router_state.json` | Persisted router state (cluster centroids, capability matrix, and specialist assignments). Written to `artifacts_mock/` or `artifacts/`. |

### timing.json Metrics

`<run dir>/timing.json` records generation timing and throughput per role (with no totals):
- `n_generated`: Number of items newly generated by the model.
- `n_cached`: Number of items retrieved from the cache.
- `seconds`: Total wall-clock seconds spent generating for this role.
- `sec_per_generation`: Average wall-clock seconds spent per newly generated item (`seconds / n_generated`), or `null` if none were generated.
- `new_tokens`: Total newly generated tokens produced for this role, or `null` if token counting is unavailable.
- `tokens_per_sec`: Generation throughput in new tokens per second (`new_tokens / seconds`), or `null` if token counting is unavailable or no seconds elapsed.

> [!WARNING]
> **Smoke Benchmark Notice**: `--benchmark smoke` contains only 12 validation items and 9 test items. Running `--benchmark smoke` with real models validates plumbing and execution flow only; accuracy metrics carry no statistical significance and do not reflect real routing quality.

> **Note**: `--mock` uses a deterministic hashed bag-of-words embedder (`hash_embed`) with no downloads and no model weights. Smoke run results validate pipeline plumbing and data flow only, and say nothing about true routing quality.

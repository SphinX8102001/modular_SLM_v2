# Mdular SLM

> **Status**: Round 0 skeleton — no logic implemented yet.
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

## Design Decisions

- **Baseline = general specialist.** `baseline_score` is defined as `scores["general"]`; no separate baseline runner or extra generation per item is needed.
- **Tie-break rule.** A specialist wins a cluster only if its calibration score is *strictly* greater than the generalist's; ties go to the generalist.
- **score_fn callback.** Router calibration is decoupled from model runners via a `score_fn(item, response) → float` callback, so it can be unit-tested without loading any model weights.
- **Val/test disjointness.** The pipeline asserts that val-set and test-set ids are strictly disjoint at startup (not enforced in the router).
- **Separated output dirs.** Mock runs write only to `results_mock/` and `artifacts_mock/`; real runs write to `results/` and `artifacts/`.
- **Known limitation.** K=3 clusters need not align with the 3 task domains; this is a research hypothesis, not a guarantee.

## Usage

Run the mock smoke pipeline end-to-end (no model downloads, CPU-only):

```bash
python run_pipeline.py --scale 0.5b --mock --seed 7
```

### Output Files

Outputs are written to `results_mock/<scale>_<benchmark>_seed<seed>/` and `artifacts_mock/`:

| Path / File | Description |
|-------------|-------------|
| `metrics.json` | Aggregated evaluation metrics: routed accuracy, baseline accuracy, delta, McNemar test p-value, Wilson CI, oracle accuracy, and routing counts. |
| `records.json` | Per-test-item evaluation records: question item, routed specialist, per-role scores, and baseline score. |
| `calibration.json` | Router calibration summary from the validation set (or `{"skipped": true}` when `--skip-calibration` is set). |
| `generations.json` | Dump of raw text generations per item and role across val and test splits (`{"val": {...}, "test": {...}}`). |
| `run_config.json` | Metadata and configuration of the run (CLI arguments, model registry entries, prompts, seed, version, git commit, python version, and size confound note). |
| `artifacts_mock/router_state.json` | Persisted router state (cluster centroids, capability matrix, and specialist assignments). |

> **Note**: `--mock` uses a deterministic hashed bag-of-words embedder (`hash_embed`) with no downloads and no model weights. Smoke run results validate pipeline plumbing and data flow only, and say nothing about true routing quality.

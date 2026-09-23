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

## Usage (coming in later rounds)

```bash
python run_pipeline.py --scale 1.5b --mock --benchmark smoke
```

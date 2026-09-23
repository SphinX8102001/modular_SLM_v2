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

## Usage (coming in later rounds)

```bash
python run_pipeline.py --scale 1.5b --mock --benchmark smoke
```

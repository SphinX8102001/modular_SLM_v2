"""run_pipeline.py — CLI skeleton for the Modular SLM pipeline (round 0)."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser (no src imports)."""
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Modular SLM research pipeline — round 0 skeleton.",
    )
    p.add_argument(
        "--scale",
        choices=["1.5b", "0.5b"],
        default="1.5b",
        help="Parameter scale of the Qwen2.5 specialist family.",
    )
    p.add_argument(
        "--backend",
        choices=["huggingface", "ollama"],
        default="huggingface",
        help="Model backend to use.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use MockRunner — no model weights downloaded.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of test items evaluated (None = all).",
    )
    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration and load a pre-saved router instead.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    p.add_argument(
        "--benchmark",
        choices=["smoke", "real"],
        default="smoke",
        help="Benchmark mode: smoke (fast, mock-safe) or real.",
    )
    return p


def main() -> None:
    """Parse arguments, execute pipeline, and exit with status code."""
    parser = build_parser()
    args = parser.parse_args()
    from src.pipeline import run

    code = run(args)
    sys.exit(code)


if __name__ == "__main__":
    main()

"""src/data.py — dataset loading, validation, and query formatting (round 4)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from src import config


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON items from path and validate against schema.
    
    Raises ValueError with item id and problem description on validation failure.
    FileNotFoundError propagates if the file does not exist.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"Expected a list of items in {path}, got {type(items).__name__}")

    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Item must be a dict, got {type(item).__name__}")

        if "id" not in item:
            raise ValueError("Item missing required 'id' key")
        item_id = str(item["id"])
        if item_id in seen_ids:
            raise ValueError(f"Duplicate item id {item_id!r} in {path}")
        seen_ids.add(item_id)

        for req_key in ("category", "question", "verifier"):
            if req_key not in item:
                raise ValueError(f"Item {item_id!r} missing required key: {req_key!r}")

        if item["category"] not in config.ROLES:
            raise ValueError(
                f"Item {item_id!r} has invalid category {item['category']!r}, "
                f"expected one of {config.ROLES}"
            )

        verifier = item["verifier"]
        if verifier not in ("contains", "numeric", "mcq_letter", "python_exec"):
            raise ValueError(f"Item {item_id!r} has unknown verifier: {verifier!r}")

        if verifier == "python_exec":
            if "test_assertions" not in item:
                raise ValueError(f"Item {item_id!r} missing 'test_assertions' for verifier 'python_exec'")
            if not isinstance(item["test_assertions"], str) or not item["test_assertions"].strip():
                raise ValueError(f"Item {item_id!r} 'test_assertions' must be a non-empty string")
        else:
            if "answer" not in item:
                raise ValueError(f"Item {item_id!r} missing 'answer' for verifier {verifier!r}")
            if verifier == "contains":
                if not isinstance(item["answer"], list) or not all(isinstance(x, str) for x in item["answer"]):
                    raise ValueError(f"Item {item_id!r} answer must be list[str] for 'contains'")
            elif verifier == "numeric":
                if isinstance(item["answer"], bool) or not isinstance(item["answer"], (int, float)):
                    raise ValueError(f"Item {item_id!r} answer must be a number for 'numeric'")
            elif verifier == "mcq_letter":
                if not isinstance(item["answer"], str) or item["answer"] not in ("A", "B", "C", "D"):
                    raise ValueError(f"Item {item_id!r} answer must be 'A', 'B', 'C', or 'D' for 'mcq_letter'")

    return items


def assert_disjoint(val_items: list[dict], test_items: list[dict]) -> None:
    """Assert that val_items and test_items do not share any item IDs."""
    val_ids = {str(item["id"]) for item in val_items}
    test_ids = {str(item["id"]) for item in test_items}
    overlap = sorted(val_ids & test_ids)
    if overlap:
        raise ValueError(f"Validation and test sets share ids: {overlap}")


def build_query(item: dict) -> str:
    """Format question for runners (appending NUMERIC_SUFFIX for numeric items)."""
    question = item["question"]
    if item.get("verifier") == "numeric":
        return f"{question}{config.NUMERIC_SUFFIX}"
    return question


def dataset_paths(benchmark: str) -> tuple[Path, Path]:
    """Return (val_path, test_path) for given benchmark ('smoke' or 'real')."""
    if benchmark == "smoke":
        return config.DATA_DIR / "smoke_val.json", config.DATA_DIR / "smoke_test.json"
    if benchmark == "real":
        return config.DATA_DIR / "val_set.json", config.DATA_DIR / "test_set.json"
    raise ValueError(f"Unknown benchmark: {benchmark!r}. Expected 'smoke' or 'real'.")

"""src/cache.py — persistent JSONL generation cache (round 5).

Design decisions:
- Append-only JSONL file; each line is {"key": str, "response": str}.
- Truncated last lines (from a crash mid-write) are silently skipped on load.
- os.fsync ensures data is on disk before returning from put().
- make_key covers runner type, model id, role, system prompt, gen settings,
  device, dtype, and query; changing any of these produces a different key.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def make_key(
    runner: Any,
    role: str,
    system_prompt: str,
    gen_settings: dict[str, Any],
    query: str,
) -> str:
    """Return a sha256 hex digest covering all inputs that affect the response.

    Changing runner type, model_id, role, system_prompt, gen_settings, device,
    dtype, or query produces a different key.
    """
    payload = [
        type(runner).__qualname__,
        getattr(runner, "model_id", None),
        role,
        system_prompt,
        gen_settings,
        getattr(runner, "device", None),
        getattr(runner, "dtype", None),
        query,
    ]
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


class GenerationCache:
    """Persistent append-only generation cache backed by a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Create parent directories and load any existing entries from disk."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load existing cache entries; skip lines that fail to parse."""
        if not self._path.is_file():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._data[obj["key"]] = obj["response"]
                except (json.JSONDecodeError, KeyError):
                    # Truncated or malformed line — skip silently
                    continue

    def get(self, key: str) -> str | None:
        """Return the cached response for key, or None if absent."""
        return self._data.get(key)

    def put(self, key: str, response: str) -> None:
        """Append the key/response pair to the JSONL file and flush to disk."""
        self._data[key] = response
        line = json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

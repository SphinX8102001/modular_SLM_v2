"""src/models.py — model runner stubs (skeleton, round 0.5).

Design decisions recorded here:
- Baseline is the general specialist: baseline_score == scores["general"].
  There is no separate baseline runner; avoiding a redundant generation per item.
- Mock runs write only to results_mock/ and artifacts_mock/.
- The public `generate` method will look up SYSTEM_PROMPTS[role] and delegate to
  the abstract `_raw_generate`; callers never construct prompts themselves.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class ModelRunner(ABC):
    """Abstract base: runs a specialist language model and returns a string response."""

    def generate(self, query: str, role: str) -> str:
        """Look up SYSTEM_PROMPTS[role], prepend it to *query*, call _raw_generate."""
        raise NotImplementedError("round_0")

    @abstractmethod
    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Send the fully-formed prompt to the backend and return raw text output."""
        raise NotImplementedError("round_0")

    @abstractmethod
    def is_mock(self) -> bool:
        """Return True iff this runner uses no real model weights."""
        raise NotImplementedError("round_0")


class HuggingFaceRunner(ModelRunner):
    """Loads and runs a HuggingFace transformers model locally."""

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Tokenize, run forward pass, decode, and return the model's text output."""
        raise NotImplementedError("round_0")

    def is_mock(self) -> bool:
        """Return False — HuggingFaceRunner always uses real weights."""
        raise NotImplementedError("round_0")


class OllamaRunner(ModelRunner):
    """Calls a locally running Ollama server via its HTTP API."""

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """POST query to Ollama /api/generate and return the response text."""
        raise NotImplementedError("round_0")

    def is_mock(self) -> bool:
        """Return False — OllamaRunner requires a live Ollama server."""
        raise NotImplementedError("round_0")


class MockRunner(ModelRunner):
    """Deterministic fake runner used in tests and smoke runs (no model weights)."""

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Return a canned, deterministic string without touching any model."""
        raise NotImplementedError("round_0")

    def is_mock(self) -> bool:
        """Return True — MockRunner never loads weights."""
        raise NotImplementedError("round_0")


def get_runner(
    backend: str,
    model_id: str,
    role: str,
    scale: str,
    is_mock: bool,
) -> ModelRunner:
    """Factory: return the correct ModelRunner subclass for the given backend/flags."""
    raise NotImplementedError("round_0")

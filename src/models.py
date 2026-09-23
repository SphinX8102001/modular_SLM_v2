"""src/models.py — model runner interfaces and implementations (round 1).

Design decisions recorded here:
- Baseline is the general specialist: baseline_score == scores["general"].
  There is no separate baseline runner; avoiding a redundant generation per item.
- Mock runs write only to results_mock/ and artifacts_mock/.
- The public `generate` method looks up SYSTEM_PROMPTS[role] and delegates to
  `_raw_generate`; callers never construct prompts themselves.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from src import config


class ModelRunner(ABC):
    """Abstract base: runs a specialist language model and returns a string response."""

    def generate(self, query: str, role: str) -> str:
        """Look up SYSTEM_PROMPTS[role] and delegate to _raw_generate."""
        system_prompt = config.SYSTEM_PROMPTS[role]
        return self._raw_generate(query=query, system_prompt=system_prompt, role=role)

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

    def __init__(self, model_id: str, role: str, scale: str) -> None:
        """Store model configuration attributes; no weights loaded here."""
        self.model_id = model_id
        self.role = role
        self.scale = scale

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Tokenize, run forward pass, decode, and return the model's text output."""
        raise NotImplementedError("round_4")

    def is_mock(self) -> bool:
        """Return False — HuggingFaceRunner always uses real weights."""
        return False


class OllamaRunner(ModelRunner):
    """Calls a locally running Ollama server via its HTTP API."""

    def __init__(
        self,
        model_id: str,
        role: str,
        scale: str,
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Store connection configuration; no socket opened here."""
        self.model_id = model_id
        self.role = role
        self.scale = scale
        self.base_url = base_url

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """POST query to Ollama /api/generate and return the response text."""
        raise NotImplementedError("round_4")

    def is_mock(self) -> bool:
        """Return False — OllamaRunner requires a live Ollama server."""
        return False


class MockRunner(ModelRunner):
    """Deterministic fake runner used in tests and smoke runs (no model weights)."""

    def __init__(self, model_id: str, role: str, scale: str) -> None:
        """Store model metadata."""
        self.model_id = model_id
        self.role = role
        self.scale = scale

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Return a canned, deterministic placeholder with no digits."""
        return f"MOCK_PLACEHOLDER [role={role}] not answered"

    def is_mock(self) -> bool:
        """Return True — MockRunner never loads weights."""
        return True

    @staticmethod
    def assert_mock_path(path_str: str) -> None:
        """Raise RuntimeError if the path points into results/ or artifacts/ and allows results_mock/ and artifacts_mock/."""
        parts = set(Path(path_str).parts) | set(str(path_str).replace("\\", "/").split("/"))
        if "results" in parts or "artifacts" in parts:
            raise RuntimeError(
                f"Mock runs cannot write to production directories ('results' or 'artifacts'): {path_str}"
            )


def get_runner(
    backend: str,
    model_id: str,
    role: str,
    scale: str,
    is_mock: bool,
) -> ModelRunner:
    """Factory: return the correct ModelRunner subclass for the given backend/flags."""
    if is_mock:
        return MockRunner(model_id=model_id, role=role, scale=scale)
    if backend == "huggingface":
        return HuggingFaceRunner(model_id=model_id, role=role, scale=scale)
    if backend == "ollama":
        return OllamaRunner(model_id=model_id, role=role, scale=scale)
    raise ValueError(f"Unknown backend: {backend!r}. Expected 'huggingface' or 'ollama'.")

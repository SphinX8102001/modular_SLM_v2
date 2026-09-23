"""src/models.py — model runner stubs (skeleton, round 0)."""

from __future__ import annotations
from abc import ABC, abstractmethod


class ModelRunner(ABC):
    """Abstract base: runs a specialist language model and returns a string response."""

    @abstractmethod
    def generate(self, query: str, role: str) -> str:
        """Generate a response for *query* using the specialist identified by *role*."""
        raise NotImplementedError("round_0")


class HuggingFaceRunner(ModelRunner):
    """Loads and runs a HuggingFace transformers model locally."""

    def __init__(self, model_id: str, role: str, scale: str) -> None:
        """Store identifiers; do NOT load weights here."""
        raise NotImplementedError("round_0")

    def generate(self, query: str, role: str) -> str:
        """Tokenize, run forward pass, decode, and return the model's text output."""
        raise NotImplementedError("round_0")


class OllamaRunner(ModelRunner):
    """Calls a locally running Ollama server via its HTTP API."""

    def __init__(self, model_id: str, role: str) -> None:
        """Store connection details; do NOT open a socket here."""
        raise NotImplementedError("round_0")

    def generate(self, query: str, role: str) -> str:
        """POST query to Ollama /api/generate and return the response text."""
        raise NotImplementedError("round_0")


class MockRunner(ModelRunner):
    """Deterministic fake runner used in tests and smoke runs (no model weights)."""

    def __init__(self, model_id: str, role: str) -> None:
        """Record model_id and role for later deterministic output generation."""
        raise NotImplementedError("round_0")

    def generate(self, query: str, role: str) -> str:
        """Return a canned, deterministic string without touching any model."""
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

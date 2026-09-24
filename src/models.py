"""src/models.py — model runner interfaces and implementations (round 5).

Design decisions recorded here:
- Baseline is the general specialist: baseline_score == scores["general"].
  There is no separate baseline runner; avoiding a redundant generation per item.
- Mock runs write only to results_mock/ and artifacts_mock/.
- The public `generate` method looks up SYSTEM_PROMPTS[role] and delegates to
  `_raw_generate`; callers never construct prompts themselves.
- HuggingFaceRunner.load() is idempotent; imports torch/transformers lazily.
- Role-major pipeline: one model is in memory at a time; load/unload per role.
"""

from __future__ import annotations
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src import config


# ──────────────────────────────────────────────────────────────────────────────
# Module-level pure helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_generation_kwargs(role: str) -> dict[str, Any]:
    """Return generation kwargs for the given role from config.GEN_SETTINGS.

    Explicitly excludes temperature (ignored when do_sample=False and only
    produces warnings on some backends).
    """
    settings = config.GEN_SETTINGS[role]
    return {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": False,
        "repetition_penalty": settings["repetition_penalty"],
    }


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

    def load(self) -> None:
        """Load model weights into memory (no-op by default)."""

    def unload(self) -> None:
        """Unload model weights from memory (no-op by default)."""

    def describe(self) -> dict[str, Any]:
        """Return a dict of metadata about the runner (empty by default)."""
        return {}


class HuggingFaceRunner(ModelRunner):
    """Loads and runs a HuggingFace transformers model locally."""

    _VALID_DEVICES = {"cpu", "cuda"}
    _VALID_DTYPES = {None, "float32", "float16", "bfloat16"}

    def __init__(
        self,
        model_id: str,
        role: str,
        scale: str,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> None:
        """Store model configuration attributes; no weights loaded here."""
        if device not in self._VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(self._VALID_DEVICES)!r}, got {device!r}"
            )
        if dtype not in self._VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(str(d) for d in self._VALID_DTYPES)!r}, got {dtype!r}"
            )
        self.model_id = model_id
        self.role = role
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self.last_new_tokens: int | None = None

    def resolve_dtype_name(self, device: str, dtype: str | None) -> str:
        """Return the concrete dtype name: explicit dtype wins; cpu->float32, cuda->float16."""
        if dtype is not None:
            return dtype
        return "float32" if device == "cpu" else "float16"

    def load(self) -> None:
        """Idempotently load the model and tokenizer; imports torch/transformers lazily."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available on this machine. "
                f"Re-run with --device cpu instead."
            )

        dtype_name = self.resolve_dtype_name(self.device, self.dtype)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[dtype_name]

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            # Older transformers versions use torch_dtype instead of dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )

        model = model.to(self.device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def unload(self) -> None:
        """Drop model and tokenizer from memory; safe to call when not loaded."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        """Tokenize, run forward pass, decode only new tokens, and return the response."""
        self.load()

        import torch

        tokenizer = self._tokenizer
        model = self._model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **build_generation_kwargs(role),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_token_ids = output_ids[0][input_len:]
        self.last_new_tokens = len(new_token_ids)
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return response

    def is_mock(self) -> bool:
        """Return False — HuggingFaceRunner always uses real weights."""
        return False

    def describe(self) -> dict[str, Any]:
        """Return torch/transformers version info and device details."""
        try:
            import torch
            import transformers

            cuda_name = None
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    cuda_name = torch.cuda.get_device_name(0)
                except Exception:
                    cuda_name = None

            return {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": self.device,
                "dtype": self.resolve_dtype_name(self.device, self.dtype),
                "num_threads": torch.get_num_threads(),
                "cuda_name": cuda_name,
            }
        except ImportError:
            return {"device": self.device}


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
        raise NotImplementedError("round_7")

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
        p = Path(path_str)
        root = Path(config.ROOT_DIR)
        is_under_root = False
        rel_path = None
        try:
            if p.is_relative_to(root):
                is_under_root = True
                rel_path = p.relative_to(root)
            elif p.resolve().is_relative_to(root.resolve()):
                is_under_root = True
                rel_path = p.resolve().relative_to(root.resolve())
        except Exception:
            pass

        if is_under_root and rel_path is not None:
            parts = set(rel_path.parts) | set(str(rel_path).replace("\\", "/").split("/"))
        else:
            parts = set(p.parts) | set(str(path_str).replace("\\", "/").split("/"))

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
    device: str = "cpu",
) -> ModelRunner:
    """Factory: return the correct ModelRunner subclass for the given backend/flags."""
    if is_mock:
        return MockRunner(model_id=model_id, role=role, scale=scale)
    if backend == "huggingface":
        return HuggingFaceRunner(model_id=model_id, role=role, scale=scale, device=device)
    if backend == "ollama":
        return OllamaRunner(model_id=model_id, role=role, scale=scale)
    raise ValueError(f"Unknown backend: {backend!r}. Expected 'huggingface' or 'ollama'.")

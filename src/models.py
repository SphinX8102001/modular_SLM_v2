"""
Model runners for Modular SLM.

Public API
----------
get_runner(backend, scale, is_mock) -> ModelRunner
    Factory that returns the correct concrete runner.

ModelRunner.generate(query, role) -> str
    Single shared code path used at calibration time AND test time.
    Looks up the system prompt from config.SYSTEM_PROMPTS[role] — never
    duplicates the prompt strings.
"""
from __future__ import annotations

import abc
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports — heavy libraries only imported when actually used
# ──────────────────────────────────────────────────────────────────────────────

def _import_torch():
    import torch
    return torch


def _import_transformers():
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    return AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


# ──────────────────────────────────────────────────────────────────────────────
# Stopping criteria: stop when a balanced \boxed{...} appears in generation
# ──────────────────────────────────────────────────────────────────────────────

def _has_balanced_boxed(text: str) -> bool:
    """Return True if *text* contains at least one complete balanced \\boxed{…}."""
    idx = 0
    marker = r"\boxed{"
    marker_len = len(marker)
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            return False
        depth = 0
        for ch in text[pos + marker_len:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    return True   # closing brace of the outer \boxed{
                depth -= 1
        idx = pos + 1
    return False  # unreachable


class BoxedStoppingCriteria:
    """
    Custom stopping criteria for HuggingFace generate().

    Decodes only the *newly generated* tokens at each step (not the full
    sequence) to keep overhead low.  Stops as soon as a complete,
    brace-balanced \\boxed{...} is detected in the accumulated new text.
    """

    def __init__(self, tokenizer, prompt_length: int) -> None:
        self._tokenizer     = tokenizer
        self._prompt_length = prompt_length
        self._last_len      = prompt_length
        self._accumulated   = ""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        current_len = input_ids.shape[-1]
        if current_len <= self._last_len:
            return False
        new_ids   = input_ids[0][self._last_len:current_len]
        new_text  = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        self._accumulated += new_text
        self._last_len = current_len
        return _has_balanced_boxed(self._accumulated)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────────────────

class ModelRunner(abc.ABC):
    """
    Abstract model runner.

    Subclasses must implement ``_raw_generate(prompt_text, role)``.
    The public ``generate(query, role)`` method builds the prompt using the
    shared system-prompt lookup from ``config.SYSTEM_PROMPTS`` — this is the
    SINGLE shared code path used by calibration and test-time evaluation.
    """

    def __init__(self, model_id: str, role: str, scale: str) -> None:
        self.model_id = model_id
        self.role     = role   # the specialist role this runner fills
        self.scale    = scale

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, query: str, role: str) -> str:
        """
        Generate a response for *query* using the system prompt for *role*.

        Parameters
        ----------
        query : str
            The user query (possibly already prefixed with NUMERIC_SUFFIX).
        role : str
            One of "general" / "math" / "code".  Looked up in
            ``config.SYSTEM_PROMPTS`` — single source of truth.

        Returns
        -------
        str
            The model's response text (decoded, stripped).
        """
        from src.config import SYSTEM_PROMPTS  # single source of truth
        system_prompt = SYSTEM_PROMPTS[role]
        return self._raw_generate(query=query, system_prompt=system_prompt, role=role)

    @abc.abstractmethod
    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        ...

    @abc.abstractmethod
    def is_mock(self) -> bool:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace runner
# ──────────────────────────────────────────────────────────────────────────────

class HuggingFaceRunner(ModelRunner):
    """Run a HuggingFace causal-LM model locally (CPU or CUDA)."""

    def __init__(self, model_id: str, role: str, scale: str) -> None:
        super().__init__(model_id, role, scale)
        self._tokenizer = None
        self._model     = None

    # Lazy load so tests can import this module without downloading weights
    def _load(self) -> None:
        if self._model is not None:
            return
        AutoTokenizer, AutoModelForCausalLM, _, _ = _import_transformers()
        torch = _import_torch()
        log.info("Loading %s (HuggingFace) …", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # CPU-friendly
            device_map="cpu",
            trust_remote_code=True,
        )
        self._model.eval()

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        self._load()
        _, _, StoppingCriteria, StoppingCriteriaList = _import_transformers()
        torch = _import_torch()
        from src.config import GEN_SETTINGS

        settings = GEN_SETTINGS[role].copy()

        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": query},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_length = input_ids.shape[-1]

        stopping_criteria = None
        if role == "math":
            # Stop as soon as a complete \boxed{...} appears
            criteria = BoxedStoppingCriteria(self._tokenizer, prompt_length)

            # Wrap in a StoppingCriteriaList-compatible object
            class _Wrapper:
                def __call__(self_inner, iids, scores, **kw):
                    return criteria(iids, scores, **kw)

            stopping_criteria = StoppingCriteriaList([_Wrapper()])

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens    = settings["max_new_tokens"],
                repetition_penalty= settings["repetition_penalty"],
                do_sample         = settings["do_sample"],
            )
            if settings["temperature"] == 0.0:
                gen_kwargs["temperature"] = None  # greedy when do_sample=False
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria

            output_ids = self._model.generate(**gen_kwargs)

        # Decode only the newly generated tokens
        new_ids   = output_ids[0][prompt_length:]
        response  = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()

    def is_mock(self) -> bool:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Ollama runner
# ──────────────────────────────────────────────────────────────────────────────

class OllamaRunner(ModelRunner):
    """Run a model via the Ollama local REST API."""

    def __init__(
        self,
        model_id: str,
        role: str,
        scale: str,
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(model_id, role, scale)
        self._base_url = base_url.rstrip("/")

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        import requests
        from src.config import GEN_SETTINGS

        settings = GEN_SETTINGS[role]
        payload = {
            "model":  self.model_id,
            "prompt": query,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature":       settings["temperature"],
                "repeat_penalty":    settings["repetition_penalty"],
                "num_predict":       settings["max_new_tokens"],
            },
        }
        resp = requests.post(
            f"{self._base_url}/api/generate", json=payload, timeout=300
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def is_mock(self) -> bool:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Mock runner — for testing pipeline infrastructure WITHOUT model weights
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_NOTICE = (
    "MOCK RUN — responses are generic placeholders only. "
    "No real model answers are embedded here."
)


class MockRunner(ModelRunner):
    """
    Returns a generic, labelled placeholder string for every query.

    Critical invariants:
    - The placeholder contains NO real answers to any test-set question.
    - The runner enforces that results are written to ``results_mock/`` and
      ``artifacts_mock/`` only; it raises if a real path is supplied.
    """

    def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
        # Generic placeholder — deliberately wrong / uninformative so that
        # mock scores stay near zero, proving that scoring works.
        return (
            f"MOCK_PLACEHOLDER [role={role}] "
            "This response is intentionally empty of real content. "
            "Query was received but not answered. "
            f"NOTICE: {_MOCK_NOTICE}"
        )

    def is_mock(self) -> bool:
        return True

    @staticmethod
    def assert_mock_path(path_str: str) -> None:
        """Raise RuntimeError if *path_str* points to a real (non-mock) results dir."""
        import os
        p = str(path_str).replace("\\", "/")
        if "/results/" in p or p.endswith("/results"):
            raise RuntimeError(
                f"MockRunner attempted to write to a real results path: {path_str}. "
                "Mock runs must write to results_mock/ only."
            )
        if "/artifacts/" in p or p.endswith("/artifacts"):
            raise RuntimeError(
                f"MockRunner attempted to write to a real artifacts path: {path_str}. "
                "Mock runs must write to artifacts_mock/ only."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_runner(
    backend: str,
    model_id: str,
    role: str,
    scale: str,
    is_mock: bool = False,
    ollama_base_url: str = "http://localhost:11434",
) -> ModelRunner:
    """
    Return the appropriate ModelRunner instance.

    Parameters
    ----------
    backend : str
        "huggingface" or "ollama".
    model_id : str
        Full model identifier (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
    role : str
        Specialist role: "general", "math", or "code".
    scale : str
        "1.5b" or "0.5b".
    is_mock : bool
        If True, returns MockRunner regardless of backend.
    """
    if is_mock:
        return MockRunner(model_id=model_id, role=role, scale=scale)
    if backend == "huggingface":
        return HuggingFaceRunner(model_id=model_id, role=role, scale=scale)
    if backend == "ollama":
        return OllamaRunner(
            model_id=model_id, role=role, scale=scale,
            base_url=ollama_base_url,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'huggingface' or 'ollama'.")

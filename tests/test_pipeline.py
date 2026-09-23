"""tests/test_pipeline.py — placeholder test suite (round 1)."""

import inspect
import pytest
import src
import src.config as config
import src.models as models
import src.router as router
import src.evaluator as evaluator  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Existing round-0 tests (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class TestSkeletonImports:
    """Verify that all modules import cleanly and key constants are correct."""

    def test_model_registry_has_both_scales(self) -> None:
        """MODEL_REGISTRY must contain exactly the two planned parameter scales."""
        assert "1.5b" in config.MODEL_REGISTRY
        assert "0.5b" in config.MODEL_REGISTRY

    def test_roles_order(self) -> None:
        """ROLES must be ['general', 'math', 'code'] in that order."""
        assert config.ROLES == ["general", "math", "code"]

    def test_version_string_exists(self) -> None:
        """src package must expose a __version__ string."""
        assert isinstance(src.__version__, str)
        assert src.__version__ == "0.0.1"


# ──────────────────────────────────────────────────────────────────────────────
# Round-0.5: signature assertions (inspect only, no stubs called)
# ──────────────────────────────────────────────────────────────────────────────

class TestSignatures:
    """Assert that round-0.5 signatures match the spec (parameter names + defaults)."""

    # ── models.py ─────────────────────────────────────────────────────────────

    def test_raw_generate_signature(self) -> None:
        params = list(inspect.signature(models.HuggingFaceRunner._raw_generate).parameters)
        assert params == ["self", "query", "system_prompt", "role"]

    # ── router.py ─────────────────────────────────────────────────────────────

    def test_spherical_kmeans_init(self) -> None:
        sig = inspect.signature(router.SphericalKMeans.__init__)
        params = list(sig.parameters)
        assert params == ["self", "n_clusters", "random_state"]
        assert sig.parameters["n_clusters"].default == 3
        assert sig.parameters["random_state"].default == 42

    def test_spherical_kmeans_has_predict_one(self) -> None:
        params = list(inspect.signature(router.SphericalKMeans.predict_one).parameters)
        assert params == ["self", "x"]

    def test_embedding_router_init(self) -> None:
        sig = inspect.signature(router.EmbeddingRouter.__init__)
        params = list(sig.parameters)
        assert params == ["self", "embedding_model_name", "n_clusters", "seed"]
        assert sig.parameters["n_clusters"].default == 3
        assert sig.parameters["seed"].default == 42

    def test_calibrate_signature(self) -> None:
        sig = inspect.signature(router.EmbeddingRouter.calibrate)
        params = list(sig.parameters)
        assert params == ["self", "val_items", "score_fn", "min_cluster_warn"]
        assert sig.parameters["min_cluster_warn"].default == 10

    def test_route_with_cluster_exists(self) -> None:
        params = list(inspect.signature(router.EmbeddingRouter.route_with_cluster).parameters)
        assert params == ["self", "query"]

    def test_load_is_classmethod(self) -> None:
        assert isinstance(
            inspect.getattr_static(router.EmbeddingRouter, "load"),
            classmethod,
        )

    # ── evaluator.py ──────────────────────────────────────────────────────────

    def test_verify_python_exec_signature(self) -> None:
        sig = inspect.signature(evaluator.verify_python_exec)
        params = list(sig.parameters)
        assert params == ["response", "test_assertions", "timeout"]
        assert sig.parameters["timeout"].default == 5

    def test_verify_numeric_signature(self) -> None:
        sig = inspect.signature(evaluator.verify_numeric)
        params = list(sig.parameters)
        assert params == ["response", "answer", "rel_tol", "abs_tol"]
        assert sig.parameters["rel_tol"].default == 1e-3
        assert sig.parameters["abs_tol"].default == 1e-6

    def test_wilson_ci_signature(self) -> None:
        sig = inspect.signature(evaluator.wilson_ci)
        params = list(sig.parameters)
        assert params == ["successes", "total", "z"]
        assert sig.parameters["z"].default == 1.96

    def test_helper_stubs_exist(self) -> None:
        for name in ("extract_last_boxed", "safe_eval_arithmetic", "extract_number"):
            assert callable(getattr(evaluator, name)), f"Missing helper: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Round-1: ModelRunner behavior, factory, and mock path assertions
# ──────────────────────────────────────────────────────────────────────────────

class TestModelRunnerRound1:
    """Verify Round 1 ModelRunner behavior, mock assertions, and config integration."""

    def test_mock_runner_generate_contains_placeholder_and_no_digits(self) -> None:
        """MockRunner.generate must contain 'MOCK_PLACEHOLDER' and NO digits for all roles."""
        for role in config.ROLES:
            runner = models.MockRunner(model_id="mock-id", role=role, scale="1.5b")
            response = runner.generate(query="Calculate 42 + 100", role=role)
            assert "MOCK_PLACEHOLDER" in response
            assert not any(ch.isdigit() for ch in response), f"Found digits in {response!r}"

    def test_generate_reads_prompts_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate() must read SYSTEM_PROMPTS from config and pass it to _raw_generate."""
        sentinel_prompt = "SENTINEL_PROMPT_XYZ"
        monkeypatch.setitem(config.SYSTEM_PROMPTS, "general", sentinel_prompt)

        spy_calls: dict[str, str] = {}
        runner = models.MockRunner(model_id="mock-id", role="general", scale="1.5b")

        def spy_raw(query: str, system_prompt: str, role: str) -> str:
            spy_calls["query"] = query
            spy_calls["system_prompt"] = system_prompt
            spy_calls["role"] = role
            return "ok"

        monkeypatch.setattr(runner, "_raw_generate", spy_raw)
        result = runner.generate(query="test question", role="general")

        assert result == "ok"
        assert spy_calls["system_prompt"] == sentinel_prompt
        assert spy_calls["role"] == "general"
        assert spy_calls["query"] == "test question"

    def test_get_runner_factory(self) -> None:
        """get_runner returns MockRunner when is_mock=True, correct runner otherwise, raises on unknown."""
        # is_mock=True always gives MockRunner
        r1 = models.get_runner("huggingface", "id1", "general", "1.5b", is_mock=True)
        assert isinstance(r1, models.MockRunner)
        assert r1.is_mock() is True

        r2 = models.get_runner("ollama", "id2", "math", "0.5b", is_mock=True)
        assert isinstance(r2, models.MockRunner)
        assert r2.is_mock() is True

        # is_mock=False returns respective runners without calling generate
        r3 = models.get_runner("huggingface", "id3", "general", "1.5b", is_mock=False)
        assert isinstance(r3, models.HuggingFaceRunner)
        assert r3.is_mock() is False

        r4 = models.get_runner("ollama", "id4", "code", "1.5b", is_mock=False)
        assert isinstance(r4, models.OllamaRunner)
        assert r4.is_mock() is False

        # unknown backend raises ValueError
        with pytest.raises(ValueError):
            models.get_runner("unsupported", "id5", "general", "1.5b", is_mock=False)

    def test_assert_mock_path(self) -> None:
        """assert_mock_path raises for results/ and artifacts/, allows results_mock/ and artifacts_mock/."""
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/x/results/run.json")
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/x/artifacts/run.json")

        # mock variants must not raise
        models.MockRunner.assert_mock_path("/x/results_mock/run.json")
        models.MockRunner.assert_mock_path("/x/artifacts_mock/run.json")

    def test_real_runners_generate_raises_not_implemented(self) -> None:
        """Calling generate() on HuggingFaceRunner or OllamaRunner raises NotImplementedError."""
        hf = models.HuggingFaceRunner("hf-id", "general", "1.5b")
        with pytest.raises(NotImplementedError):
            hf.generate("query", "general")

        ol = models.OllamaRunner("ol-id", "math", "0.5b")
        with pytest.raises(NotImplementedError):
            ol.generate("query", "math")

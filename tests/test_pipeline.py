"""tests/test_pipeline.py — placeholder test suite (skeleton, round 0.5)."""

import inspect
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

"""tests/test_pipeline.py — placeholder test suite (skeleton, round 0)."""

import src
import src.config as config
import src.models  # noqa: F401
import src.router  # noqa: F401
import src.evaluator  # noqa: F401


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

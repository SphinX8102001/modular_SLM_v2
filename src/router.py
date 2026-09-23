"""src/router.py — embedding router stubs (skeleton, round 0)."""

from __future__ import annotations
from typing import Any


class SphericalKMeans:
    """K-Means on the unit sphere (cosine ≡ Euclidean after L2-normalisation)."""

    def fit(self, embeddings: Any) -> "SphericalKMeans":
        """Fit cluster centroids on *embeddings*; normalise centroids to unit length."""
        raise NotImplementedError("round_0")

    def predict(self, embeddings: Any) -> Any:
        """Assign each row of *embeddings* to its nearest centroid index."""
        raise NotImplementedError("round_0")


class EmbeddingRouter:
    """
    Full routing pipeline: embed → cluster → capability-profile lookup → specialist.
    """

    def calibrate(self, val_items: list[dict], runners: dict[str, Any]) -> None:
        """Build the Capability Profile Matrix S(cluster, specialist) from val data."""
        raise NotImplementedError("round_0")

    def route(self, query: str) -> str:
        """Embed *query*, find its cluster, return the best specialist role name."""
        raise NotImplementedError("round_0")

    def save(self, path: str) -> None:
        """Persist the fitted router (centroids + capability matrix) to *path*."""
        raise NotImplementedError("round_0")

    def load(self, path: str) -> None:
        """Restore a previously saved router from *path*."""
        raise NotImplementedError("round_0")

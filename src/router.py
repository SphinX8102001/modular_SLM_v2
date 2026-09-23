"""src/router.py — embedding router stubs (skeleton, round 0.5).

Design decisions recorded here:
- Router calibration is decoupled from models via a score_fn callback so it can
  be unit-tested without loading any model weights.
- Tie-break: a specialist wins a cluster only if its score is STRICTLY greater
  than the generalist's; ties go to the generalist.
- Known limitation: K=3 clusters need not align with the 3 domains.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np


class SphericalKMeans:
    """K-Means on the unit sphere (cosine == Euclidean after L2-normalisation)."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        """Store hyperparameters; do NOT fit anything here."""
        raise NotImplementedError("round_0")

    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        """Fit cluster centroids on *X*; L2-normalise centroids after each update."""
        raise NotImplementedError("round_0")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of *X* to its nearest centroid index."""
        raise NotImplementedError("round_0")

    def predict_one(self, x: np.ndarray) -> int:
        """Return the centroid index nearest to the single vector *x*."""
        raise NotImplementedError("round_0")


class EmbeddingRouter:
    """Full routing pipeline: embed → cluster → capability-profile lookup → specialist.

    Design decisions:
    - calibrate() takes a score_fn callback (not runners) so tests never load weights.
    - Tie-break: specialist wins only if STRICTLY better than generalist.
    - Val/test disjointness is asserted by the pipeline, not here.
    """

    def __init__(
        self,
        embedding_model_name: str,
        n_clusters: int = 3,
        seed: int = 42,
    ) -> None:
        """Store config; do NOT load the embedding model here."""
        raise NotImplementedError("round_0")

    def calibrate(
        self,
        val_items: list[dict],
        score_fn: Callable[[dict, str], float],
        min_cluster_warn: int = 10,
    ) -> dict:
        """Build the Capability Profile Matrix S(cluster, specialist) from val data.

        Uses *score_fn* callback — no runner/model objects needed here.
        Warns if any cluster has fewer than *min_cluster_warn* items.
        Returns a summary dict (schema finalised in round 1).
        """
        raise NotImplementedError("round_0")

    def route(self, query: str) -> str:
        """Embed *query*, find its cluster, return the best specialist role name."""
        raise NotImplementedError("round_0")

    def route_with_cluster(self, query: str) -> tuple[str, int]:
        """Like route(), but also return the assigned cluster index."""
        raise NotImplementedError("round_0")

    def save(self, path: str | Path) -> None:
        """Persist the fitted router (centroids + capability matrix) to *path*."""
        raise NotImplementedError("round_0")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingRouter":
        """Restore a previously saved EmbeddingRouter from *path*."""
        raise NotImplementedError("round_0")

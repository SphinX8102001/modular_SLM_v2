"""
Embedding-based router for Modular SLM.

Architecture (training-free, 4 stages):
  1. Embed query with sentence-transformers/all-MiniLM-L6-v2 (384-dim), L2-normalised.
  2. Cluster with Spherical K-Means (K=3) fitted on the val set; centroids are
     normalised to the unit sphere so Euclidean distance == cosine distance.
  3. Build Capability Profile Matrix S[cluster][specialist] from val-set scores.
  4. Route query to the specialist that scored best on that cluster — with the
     critical TIE-BREAK RULE: a specialist must STRICTLY outscore the generalist
     to win a cluster.  Ties and generalist-wins go to the generalist.

Save / Load
-----------
Router state is saved as JSON (centroids, capability matrix, assignments).
Loading is implemented as **pure numpy distance computation** against stored
centroids — no sklearn fitted-state dependency.  Prediction on the same input
will produce identical cluster assignments before and after a save/load cycle.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Spherical K-Means
# ──────────────────────────────────────────────────────────────────────────────

class SphericalKMeans:
    """
    Wrapper around sklearn KMeans that normalises centroids to the unit sphere
    after fitting, so that Euclidean nearest-centroid == cosine nearest-centroid.

    Prediction is implemented as **pure numpy** distance computation so that a
    loaded router (centroids stored as JSON) produces identical assignments
    without any sklearn object in memory.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters    = n_clusters
        self.random_state  = random_state
        self.centroids_: Optional[np.ndarray] = None  # shape (K, D), unit-normed

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        """
        Fit K-Means on L2-normalised *X*, then normalise the centroids.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Already L2-normalised embedding matrix.
        """
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters   = self.n_clusters,
            random_state = self.random_state,
            n_init       = "auto",
        )
        km.fit(X)
        raw_centroids   = km.cluster_centers_          # (K, D)
        norms           = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        # Avoid division by zero for degenerate centroids
        norms           = np.where(norms == 0, 1.0, norms)
        self.centroids_ = raw_centroids / norms        # unit-normalised
        return self

    # ------------------------------------------------------------------
    # Prediction — pure numpy, no sklearn state
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each row of *X* to its nearest unit-sphere centroid.

        Implemented as pure numpy L2 distance computation (which equals cosine
        distance on the unit sphere) so that loaded routers produce identical
        results without any sklearn fitted object.

        Parameters
        ----------
        X : np.ndarray, shape (N, D) or (D,)
            L2-normalised query embeddings.

        Returns
        -------
        np.ndarray, shape (N,) — cluster indices.
        """
        if self.centroids_ is None:
            raise RuntimeError("SphericalKMeans has not been fitted yet.")
        X2 = np.atleast_2d(X)  # (N, D)
        # L2 distance²  = ||x - c||² = 2 - 2 * x·c  (on unit sphere)
        # Minimise L2 ↔ maximise dot product
        dots = X2 @ self.centroids_.T          # (N, K)
        return np.argmax(dots, axis=1)         # (N,)

    def predict_one(self, x: np.ndarray) -> int:
        return int(self.predict(x[np.newaxis, :])[0])


# ──────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """Lazy-loaded sentence-transformer encoder with L2 normalisation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model     = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s …", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, D) or (D,) for single string."""
        self._load()
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embs  = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs  = embs / norms
        return embs[0] if single else embs


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Router
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingRouter:
    """
    Full 4-stage training-free routing pipeline.

    Usage
    -----
    router = EmbeddingRouter(embedding_model_name, n_clusters, seed)
    router.calibrate(val_items, score_fn)   # fits clustering + capability matrix
    specialist = router.route(query)         # predict specialist for new query
    router.save(path)
    router.load(path)                        # pure numpy, no sklearn state needed
    """

    SPECIALISTS = ("general", "math", "code")

    def __init__(
        self,
        embedding_model_name: str,
        n_clusters: int = 3,
        seed: int = 42,
    ) -> None:
        self._embedder    = EmbeddingModel(embedding_model_name)
        self._kmeans      = SphericalKMeans(n_clusters=n_clusters, random_state=seed)
        self.n_clusters   = n_clusters
        self.seed         = seed

        # Set after calibration
        self.centroids_: Optional[np.ndarray]          = None  # (K, D)
        self.capability_matrix_: Optional[np.ndarray]  = None  # (K, 3) — S[cluster][specialist]
        self.cluster_assignments_: Optional[dict]      = None  # cluster_idx -> specialist name
        self.val_cluster_labels_: Optional[np.ndarray] = None  # for diagnostics

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        val_items: list[dict],
        score_fn,
        min_cluster_warn: int = 10,
    ) -> dict:
        """
        Fit the router on the validation set.

        Parameters
        ----------
        val_items : list[dict]
            Each item must have at minimum "query" (str).
        score_fn : callable(item, specialist_name) -> float  [0.0 or 1.0]
            Returns the binary score for specialist *specialist_name* on *item*.
        min_cluster_warn : int
            Log a warning for any cluster that has fewer than this many items.

        Returns
        -------
        dict with calibration diagnostics.
        """
        queries   = [it["query"] for it in val_items]
        embeddings = self._embedder.encode(queries)   # (N, D), L2-normalised

        # Stage 2: spherical K-Means
        log.info("Fitting SphericalKMeans (K=%d) on %d val embeddings …",
                 self.n_clusters, len(queries))
        self._kmeans.fit(embeddings)
        self.centroids_ = self._kmeans.centroids_

        # Assign each val item to a cluster
        labels = self._kmeans.predict(embeddings)   # (N,)
        self.val_cluster_labels_ = labels

        # Cluster-size warning
        for k in range(self.n_clusters):
            count = int((labels == k).sum())
            if count < min_cluster_warn:
                warnings.warn(
                    f"[Router] Cluster {k} has only {count} validation items "
                    f"(threshold={min_cluster_warn}). Capability estimate may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
                log.warning(
                    "SMALL-SAMPLE WARNING: cluster %d has only %d val items (< %d).",
                    k, count, min_cluster_warn,
                )

        # Stage 3: build capability profile matrix S[cluster, specialist]
        n_specs = len(self.SPECIALISTS)
        S = np.zeros((self.n_clusters, n_specs), dtype=float)
        counts = np.zeros((self.n_clusters,), dtype=int)

        for item, cluster_idx in zip(val_items, labels):
            counts[cluster_idx] += 1
            for s_idx, spec in enumerate(self.SPECIALISTS):
                S[cluster_idx, s_idx] += score_fn(item, spec)

        # Normalise by cluster size (avoid div-by-zero)
        for k in range(self.n_clusters):
            if counts[k] > 0:
                S[k] /= counts[k]

        self.capability_matrix_ = S
        log.info("Capability matrix S[cluster, specialist]:\n%s", S)

        # Stage 4: cluster → specialist assignment
        # TIE-BREAK RULE: specialist wins ONLY if score STRICTLY > generalist.
        # On a tie or if generalist scores highest, generalist wins.
        general_idx = self.SPECIALISTS.index("general")
        assignments: dict[int, str] = {}

        for k in range(self.n_clusters):
            general_score = S[k, general_idx]
            best_spec     = "general"
            best_score    = general_score

            for s_idx, spec in enumerate(self.SPECIALISTS):
                if spec == "general":
                    continue
                # STRICTLY greater — not >=
                if S[k, s_idx] > general_score and S[k, s_idx] > best_score:
                    best_spec  = spec
                    best_score = S[k, s_idx]

            assignments[k] = best_spec
            log.info(
                "Cluster %d → '%s'  (scores: %s)",
                k,
                best_spec,
                {sp: round(float(S[k, si]), 4) for si, sp in enumerate(self.SPECIALISTS)},
            )

        self.cluster_assignments_ = assignments

        return {
            "capability_matrix": S.tolist(),
            "cluster_assignments": {str(k): v for k, v in assignments.items()},
            "cluster_counts": counts.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Routing
    # ──────────────────────────────────────────────────────────────────

    def route(self, query: str) -> str:
        """
        Return the specialist name for *query*.

        Prediction is pure numpy distance against stored centroids — does NOT
        require the sklearn KMeans fitted state.
        """
        self._assert_calibrated()
        emb     = self._embedder.encode(query)          # (D,)
        cluster = self._kmeans.predict_one(emb)
        return self.cluster_assignments_[cluster]

    def route_with_cluster(self, query: str) -> tuple[str, int]:
        """Return (specialist_name, cluster_index) for diagnostics."""
        self._assert_calibrated()
        emb     = self._embedder.encode(query)
        cluster = self._kmeans.predict_one(emb)
        return self.cluster_assignments_[cluster], cluster

    # ──────────────────────────────────────────────────────────────────
    # Save / Load — pure numpy, no sklearn state dependency
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist router state as JSON.  Centroids stored as plain lists."""
        self._assert_calibrated()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "embedding_model":     self._embedder.model_name,
            "n_clusters":          self.n_clusters,
            "seed":                self.seed,
            "specialists":         list(self.SPECIALISTS),
            # Store as nested lists for JSON serialisability
            "centroids":           self.centroids_.tolist(),
            "capability_matrix":   self.capability_matrix_.tolist(),
            "cluster_assignments": {str(k): v for k, v in self.cluster_assignments_.items()},
        }
        path.write_text(json.dumps(state, indent=2))
        log.info("Router state saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingRouter":
        """
        Load a previously saved router.

        The returned router uses **pure numpy** centroid distance for prediction —
        the sklearn KMeans fitted state is NOT required or stored.  Prediction
        on the same input produces identical assignments as before saving.
        """
        path  = Path(path)
        state = json.loads(path.read_text())

        router = cls(
            embedding_model_name = state["embedding_model"],
            n_clusters           = state["n_clusters"],
            seed                 = state.get("seed", 42),
        )
        router.centroids_          = np.array(state["centroids"])
        router.capability_matrix_  = np.array(state["capability_matrix"])
        router.cluster_assignments_= {int(k): v for k, v in state["cluster_assignments"].items()}

        # Inject centroids into SphericalKMeans so predict() works via pure numpy
        router._kmeans.centroids_  = router.centroids_

        log.info("Router state loaded from %s", path)
        return router

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _assert_calibrated(self) -> None:
        if self.centroids_ is None or self.cluster_assignments_ is None:
            raise RuntimeError(
                "Router has not been calibrated yet.  "
                "Call calibrate() first or load a saved state."
            )

    @property
    def is_calibrated(self) -> bool:
        return self.centroids_ is not None and self.cluster_assignments_ is not None

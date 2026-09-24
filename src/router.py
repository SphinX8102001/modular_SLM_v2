"""src/router.py — embedding router and spherical K-means clustering (round 3).

Design decisions recorded here:
- Router calibration is decoupled from models via a score_fn callback so it can
  be unit-tested without loading any model weights.
- Tie-break: a specialist wins a cluster only if its score is STRICTLY greater
  than the generalist's; ties go to the generalist.
- Known limitation: K=3 clusters need not align with the 3 domains.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
from src import config


class SphericalKMeans:
    """K-Means on the unit sphere (cosine == Euclidean after L2-normalisation)."""

    N_INIT: int = 10
    MAX_ITER: int = 100
    TOL: float = 1e-6

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        """Store hyperparameters; do NOT fit anything here."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        """Fit cluster centroids on *X*; L2-normalise centroids after each update."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2-D numpy array.")
        n_samples, d = X.shape
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters}).")

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0) or np.any(np.isnan(norms)):
            raise ValueError("X contains zero-norm or NaN rows.")
        X_norm = X / norms

        best_similarity = -np.inf
        best_centroids: np.ndarray | None = None

        for init_index in range(self.N_INIT):
            rng = np.random.RandomState(self.random_state + init_index)

            # k-means++-style initialization using cosine distance (1 - dot)
            first_idx = rng.randint(0, n_samples)
            centroids = [X_norm[first_idx]]

            for _ in range(1, self.n_clusters):
                dots = np.dot(X_norm, np.array(centroids).T)
                min_dist = np.maximum(0.0, 1.0 - np.max(dots, axis=1))
                probs = min_dist ** 2
                total_prob = np.sum(probs)
                if total_prob > 0:
                    probs = probs / total_prob
                else:
                    probs = np.ones(n_samples) / n_samples
                next_idx = rng.choice(n_samples, p=probs)
                centroids.append(X_norm[next_idx])

            centroids_arr = np.array(centroids, dtype=np.float64)
            centroids_arr /= np.linalg.norm(centroids_arr, axis=1, keepdims=True)

            for _ in range(self.MAX_ITER):
                similarities = np.dot(X_norm, centroids_arr.T)
                labels = np.argmax(similarities, axis=1)

                new_centroids = np.zeros_like(centroids_arr)
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if np.sum(mask) == 0:
                        sim_to_k = np.dot(X_norm, centroids_arr[k])
                        worst_idx = np.argmin(sim_to_k)
                        new_centroids[k] = X_norm[worst_idx]
                    else:
                        mean_vec = np.mean(X_norm[mask], axis=0)
                        norm = np.linalg.norm(mean_vec)
                        if norm == 0:
                            sim_to_k = np.dot(X_norm, centroids_arr[k])
                            worst_idx = np.argmin(sim_to_k)
                            new_centroids[k] = X_norm[worst_idx]
                        else:
                            new_centroids[k] = mean_vec / norm

                new_centroids /= np.linalg.norm(new_centroids, axis=1, keepdims=True)
                shift = np.max(np.linalg.norm(new_centroids - centroids_arr, axis=1))
                centroids_arr = new_centroids
                if shift < self.TOL:
                    break

            sims = np.dot(X_norm, centroids_arr.T)
            total_sim = float(np.sum(np.max(sims, axis=1)))
            if total_sim > best_similarity:
                best_similarity = total_sim
                best_centroids = centroids_arr.copy()

        self.centroids_ = best_centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of *X* to its nearest centroid index."""
        if self.centroids_ is None:
            raise RuntimeError("SphericalKMeans is not fitted yet.")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2-D numpy array.")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("X contains zero-norm rows.")
        X_norm = X / norms
        dots = np.dot(X_norm, self.centroids_.T)
        return np.argmax(dots, axis=1)

    def predict_one(self, x: np.ndarray) -> int:
        """Return the centroid index nearest to the single vector *x*."""
        if self.centroids_ is None:
            raise RuntimeError("SphericalKMeans is not fitted yet.")
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError("x must be a 1-D numpy array.")
        return int(self.predict(x.reshape(1, -1))[0])


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
        self.embedding_model_name = embedding_model_name
        self.n_clusters = n_clusters
        self.seed = seed
        self._model: Any = None
        self._kmeans: SphericalKMeans | None = None
        self.capability_matrix_: dict[int, dict[str, float]] | None = None
        self.assignment_: dict[int, str] | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Lazily embed texts using SentenceTransformer on CPU; unit-normalized."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float64)

    def calibrate(
        self,
        val_items: list[dict],
        score_fn: Callable[[dict, str], float],
        min_cluster_warn: int = 10,
    ) -> dict:
        """Build the Capability Profile Matrix S(cluster, specialist) from val data.

        Raises ValueError if val_items is empty, shorter than n_clusters, or contains
        duplicate item IDs (compared as str).
        """
        if not val_items or len(val_items) < self.n_clusters:
            raise ValueError(f"val_items must contain at least n_clusters ({self.n_clusters}) items.")

        seen_ids: set[str] = set()
        for item in val_items:
            item_id = str(item["id"])
            if item_id in seen_ids:
                raise ValueError(f"Duplicate item id found in val_items: {item_id!r}")
            seen_ids.add(item_id)

        # 1. Embed questions and fit SphericalKMeans
        questions = [item["question"] for item in val_items]
        embeddings = self._embed(questions)
        self._kmeans = SphericalKMeans(n_clusters=self.n_clusters, random_state=self.seed).fit(embeddings)
        clusters = self._kmeans.predict(embeddings)

        # 2. Score every item on every role exactly once
        item_scores: dict[str, dict[str, float]] = {}
        for item in val_items:
            item_id = str(item["id"])
            item_scores[item_id] = {}
            for role in config.ROLES:
                item_scores[item_id][role] = float(score_fn(item, role))

        # 3. Capability profile matrix: mean score per role in each cluster
        capability_matrix: dict[int, dict[str, float]] = {c: {} for c in range(self.n_clusters)}
        cluster_sizes: list[int] = [0] * self.n_clusters
        cluster_category_counts: dict[int, dict[str, int]] = {c: {} for c in range(self.n_clusters)}

        for c in range(self.n_clusters):
            items_in_c = [val_items[i] for i in range(len(val_items)) if clusters[i] == c]
            cluster_sizes[c] = len(items_in_c)
            for it in items_in_c:
                cat = it.get("category", "unknown")
                cluster_category_counts[c][cat] = cluster_category_counts[c].get(cat, 0) + 1

            for role in config.ROLES:
                if items_in_c:
                    mean_score = sum(item_scores[str(it["id"])][role] for it in items_in_c) / len(items_in_c)
                else:
                    mean_score = 0.0
                capability_matrix[c][role] = mean_score

        self.capability_matrix_ = capability_matrix

        # 4. Assignment rule: "general" by default; specialist wins only if STRICTLY greater
        assignment: dict[int, str] = {}
        for c in range(self.n_clusters):
            if cluster_sizes[c] == 0:
                assignment[c] = "general"
                continue

            gen_score = capability_matrix[c].get("general", 0.0)
            math_score = capability_matrix[c].get("math", 0.0)
            code_score = capability_matrix[c].get("code", 0.0)

            math_beats = math_score > gen_score
            code_beats = code_score > gen_score

            if math_beats and code_beats:
                if code_score > math_score:
                    assignment[c] = "code"
                else:
                    # math_score > code_score or exact tie: take math (ROLES order)
                    assignment[c] = "math"
            elif math_beats:
                assignment[c] = "math"
            elif code_beats:
                assignment[c] = "code"
            else:
                assignment[c] = "general"

        self.assignment_ = assignment

        # 5. Cluster size warnings
        warnings_list: list[str] = []
        for c in range(self.n_clusters):
            if cluster_sizes[c] < min_cluster_warn:
                msg = f"Cluster {c} has only {cluster_sizes[c]} items (< min_cluster_warn={min_cluster_warn})."
                warnings.warn(msg, UserWarning)
                warnings_list.append(msg)

        # 6. Return summary dict
        return {
            "n_items": len(val_items),
            "cluster_sizes": cluster_sizes,
            "capability_matrix": self.capability_matrix_,
            "assignment": self.assignment_,
            "cluster_category_counts": cluster_category_counts,
            "item_scores": item_scores,
            "warnings": warnings_list,
        }

    def route_with_cluster(self, query: str) -> tuple[str, int]:
        """Embed *query*, find its cluster, return (role, cluster_index)."""
        if self._kmeans is None or self.assignment_ is None:
            raise RuntimeError("EmbeddingRouter is not calibrated yet.")
        emb = self._embed([query])
        cluster = self._kmeans.predict_one(emb[0])
        role = self.assignment_[cluster]
        return role, cluster

    def route(self, query: str) -> str:
        """Embed *query*, find its cluster, return the best specialist role name."""
        role, _ = self.route_with_cluster(query)
        return role

    def save(self, path: str | Path) -> None:
        """Persist the fitted router (centroids + capability matrix) to *path*."""
        if (
            self._kmeans is None
            or self._kmeans.centroids_ is None
            or self.capability_matrix_ is None
            or self.assignment_ is None
        ):
            raise RuntimeError("EmbeddingRouter is not calibrated yet.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "embedding_model_name": self.embedding_model_name,
            "n_clusters": self.n_clusters,
            "seed": self.seed,
            "centroids": self._kmeans.centroids_.tolist(),
            "capability_matrix": {str(k): v for k, v in self.capability_matrix_.items()},
            "assignment": {str(k): v for k, v in self.assignment_.items()},
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingRouter":
        """Restore a previously saved EmbeddingRouter from *path* without loading model."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        schema_version = data.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"Unknown or unsupported schema_version: {schema_version}")

        router = cls(
            embedding_model_name=data["embedding_model_name"],
            n_clusters=data["n_clusters"],
            seed=data["seed"],
        )
        km = SphericalKMeans(n_clusters=data["n_clusters"], random_state=data["seed"])
        km.centroids_ = np.array(data["centroids"], dtype=np.float64)
        router._kmeans = km
        router.capability_matrix_ = {int(k): v for k, v in data["capability_matrix"].items()}
        router.assignment_ = {int(k): v for k, v in data["assignment"].items()}
        return router

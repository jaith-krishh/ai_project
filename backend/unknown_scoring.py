"""
backend/unknown_scoring.py
==========================
Unknown-sound scoring shared by calibration (calibrate_unknown.py) and
inference (p3_event_merging.label_windows).

Every method returns a score where HIGHER means "more likely unknown"; a window
is flagged Unknown when its score is greater than the calibrated threshold.

Methods
-------
distance             Euclidean distance to the nearest class centroid.
normalized_distance  Distance to each centroid divided by that class's typical
                     spread (RMS distance of its training clips to the centroid),
                     minimised over classes. Tight classes get a tight boundary,
                     spread-out classes a loose one.
max_softmax          1 - highest softmax probability (low model confidence).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

METHODS = ("distance", "normalized_distance", "max_softmax")


def _pairwise_distances(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(N, D) x (C, D) -> (N, C) Euclidean distances."""
    diff = embeddings[:, None, :] - centroids[None, :, :]
    return np.linalg.norm(diff, axis=2)


def unknown_scores(
    method: str,
    embeddings: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return one unknown-score per row.

    Args:
        method: One of METHODS.
        embeddings: (N, D) window/clip embeddings (distance methods).
        centroids: (C, D) known-class centroids (distance methods).
        radii: (C,) per-class spread (normalized_distance).
        probs: (N, C) softmax probabilities (max_softmax).
    """
    if method == "distance":
        return _pairwise_distances(np.asarray(embeddings, float), np.asarray(centroids, float)).min(axis=1)
    if method == "normalized_distance":
        if radii is None:
            raise ValueError("normalized_distance needs class radii; rerun backend.compute_centroids.")
        d = _pairwise_distances(np.asarray(embeddings, float), np.asarray(centroids, float))
        return (d / np.maximum(np.asarray(radii, float), 1e-8)[None, :]).min(axis=1)
    if method == "max_softmax":
        return 1.0 - np.asarray(probs, float).max(axis=1)
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")


def auroc(known_scores: np.ndarray, unknown_scores_: np.ndarray) -> float:
    """Probability that a random unknown clip scores higher than a random known clip
    (area under the ROC curve; 0.5 = no separation, 1.0 = perfect). Unaffected by
    how many known vs unknown clips there are, unlike F1."""
    k = np.asarray(known_scores, float)
    u = np.asarray(unknown_scores_, float)
    if len(k) == 0 or len(u) == 0:
        return float("nan")
    ranks = np.concatenate([k, u]).argsort().argsort().astype(float) + 1.0
    # Average ranks for ties
    allv = np.concatenate([k, u])
    for v in np.unique(allv):
        idx = allv == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    rank_sum_u = ranks[len(k):].sum()
    return float((rank_sum_u - len(u) * (len(u) + 1) / 2) / (len(k) * len(u)))

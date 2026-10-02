"""
P3 Module — Event Merging + Unknown Similarity List
Acoustic Sound Analyzer Project
Assigned Developer: Sarah (P3)

This module implements post-processing for P2 raw sliding-window predictions:
1. Merges windows with identical predicted labels into event intervals (per label, so
   simultaneous sounds stay separate; overlapping window time is only counted once).
2. Calculates merged event start/end boundaries and mean confidence.
3. Computes distance-based similarity rankings against known class centroids for "Unknown" events.
4. Outputs standardized event dictionaries compatible with P4 reporting.

Expected P2 Input Format:
List of dictionaries, each containing:
  - "start" (or "start_time"): float
  - "end" (or "end_time"): float
  - "label" (or "predicted_label"): str
  - "confidence" (or "probability"): float
  - "embedding" (or "vector" / "features"): list[float] or np.ndarray

Standard P3 Event Output Format:
{
    "label": str,
    "start": float,
    "end": float,
    "confidence": float,
    "similar_to": list[dict] | None
}
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union


def _extract_window_field(window: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Helper to flexibly retrieve field values supporting alias key names from P2."""
    for key in keys:
        if key in window:
            return window[key]
    return default


def compute_unknown_similarity(
    embedding: Union[List[float], np.ndarray],
    centroids: Dict[str, Union[List[float], np.ndarray]],
    top_k: int = 3,
    metric: str = "euclidean"
) -> List[Dict[str, Union[str, float]]]:
    """
    Calculates distance between an Unknown embedding and all known-class centroids,
    sorts classes from closest to farthest, and converts distances to normalized 
    similarity values summing to ~1.0 (100%). Excludes 'Unknown' class.

    Args:
        embedding: Feature embedding vector of the Unknown event.
        centroids: Dict mapping known class labels to centroid vectors.
        top_k: Number of top similar classes to return (default 3).
        metric: Distance metric to use ("euclidean" or "cosine").

    Returns:
        List of dicts: [{"label": str, "similarity": float}, ...] sorted by similarity descending.
    """
    if not centroids:
        return []

    emb = np.array(embedding, dtype=np.float64)

    # Filter out 'Unknown' or non-class entries
    valid_centroids = {
        label: np.array(vec, dtype=np.float64)
        for label, vec in centroids.items()
        if label.lower() != "unknown"
    }

    if not valid_centroids:
        return []

    labels = list(valid_centroids.keys())
    distances = []

    for label in labels:
        c_vec = valid_centroids[label]
        if metric == "cosine":
            norm_e = np.linalg.norm(emb)
            norm_c = np.linalg.norm(c_vec)
            if norm_e > 0 and norm_c > 0:
                sim = np.dot(emb, c_vec) / (norm_e * norm_c)
                dist = 1.0 - sim
            else:
                dist = 1.0
        else:  # Euclidean distance
            dist = np.linalg.norm(emb - c_vec)
        distances.append(float(dist))

    distances = np.array(distances, dtype=np.float64)

    # Convert distances to similarity weights.
    # Inverse distance weighting with small epsilon to prevent division by zero:
    epsilon = 1e-8
    inv_dist = 1.0 / (distances + epsilon)
    raw_similarities = inv_dist / np.sum(inv_dist)

    # Combine labels with similarity scores
    ranked_pairs = list(zip(labels, distances, raw_similarities))
    # Sort closest to farthest (smallest distance first, highest similarity first)
    ranked_pairs.sort(key=lambda x: x[1])

    # Select top-k candidate classes
    top_candidates = ranked_pairs[:top_k]

    # Re-normalize top-k similarity scores so they sum to 1.0 (100%)
    top_sims = np.array([item[2] for item in top_candidates], dtype=np.float64)
    if np.sum(top_sims) > 0:
        norm_top_sims = top_sims / np.sum(top_sims)
    else:
        norm_top_sims = np.ones(len(top_candidates)) / len(top_candidates)

    # Format result list, rounding similarity to 4 decimal places (or 2 decimal places)
    # Adjust last item so sum equals exactly 1.00 if needed after rounding
    rounded_sims = [round(float(s), 2) for s in norm_top_sims]
    diff = round(1.0 - sum(rounded_sims), 2)
    if rounded_sims and diff != 0:
        rounded_sims[0] = round(rounded_sims[0] + diff, 2)

    similarity_list = [
        {"label": top_candidates[i][0], "similarity": rounded_sims[i]}
        for i in range(len(top_candidates))
    ]

    return similarity_list


def load_centroids(centroids_path: str) -> Dict[str, np.ndarray]:
    """
    Loads P1's centroids.pt ({"centroids": Tensor (C, D), "class_names": list[str]})
    and returns it as the {label: vector} dict used by this module.
    """
    import torch

    payload = torch.load(centroids_path, map_location="cpu")
    vectors = payload["centroids"].cpu().numpy()
    return {name: vectors[i] for i, name in enumerate(payload["class_names"])}


def load_threshold(threshold_path: str) -> float:
    """Loads the calibrated unknown-distance cutoff from P1's threshold.pt."""
    import torch

    return float(torch.load(threshold_path, map_location="cpu")["threshold"])


def _owned_spans(windows: List[Dict[str, Any]]) -> Dict[float, float]:
    """
    Maps each distinct window start time to the end of the time span that window
    "owns". Sliding windows overlap (e.g. 4s windows every 2s), so each window only
    owns the time up to the next window's start; the last window owns up to its
    own end. This makes every second of audio belong to exactly one window, so
    merged events never double-count overlapping time.
    """
    ends: Dict[float, float] = {}
    for w in windows:
        start = float(_extract_window_field(w, ["start", "start_time"], 0.0))
        end = float(_extract_window_field(w, ["end", "end_time"], 0.0))
        ends[start] = max(ends.get(start, end), end)

    starts = sorted(ends)
    return {
        s: min(ends[s], starts[i + 1]) if i + 1 < len(starts) else ends[s]
        for i, s in enumerate(starts)
    }


def label_windows(
    windows: List[Dict[str, Any]],
    centroids: Dict[str, Union[List[float], np.ndarray]],
    threshold: Optional[float],
    prob_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Converts P2's raw window output ({"start", "end", "probs", "embedding"}) into the
    labelled predictions consumed by merge_events / postprocess_predictions.

    - If the Euclidean distance from a window's embedding to every known centroid
      exceeds P1's threshold, the window yields a single "Unknown" prediction
      (confidence = its highest class probability).
    - Otherwise it yields the highest-probability class, plus every other class
      whose softmax probability is >= prob_threshold, so overlapping sounds
      (e.g. traffic + birds sharing the probability mass) are all kept.
    - If threshold is None (P1's threshold.pt not available yet), unknown
      detection is skipped.

    Each prediction's start/end is trimmed to the span the window owns (see
    _owned_spans), so predictions from consecutive windows never overlap.
    """
    known = [np.asarray(v, dtype=np.float64) for k, v in centroids.items() if k.lower() != "unknown"]
    centroid_matrix = np.stack(known) if known else None
    spans = _owned_spans(windows)

    labelled = []
    for win in windows:
        probs = win["probs"]
        start = float(win["start"])
        end = spans[start]

        embedding = np.asarray(win["embedding"], dtype=np.float64)
        is_unknown = False
        if centroid_matrix is not None and threshold is not None:
            min_dist = float(np.min(np.linalg.norm(centroid_matrix - embedding, axis=1)))
            is_unknown = min_dist > threshold

        if is_unknown:
            active = [("Unknown", float(max(probs.values())))]
        else:
            top_label = max(probs, key=probs.get)
            active = [
                (label, float(p)) for label, p in probs.items()
                if label == top_label or p >= prob_threshold
            ]

        for label, confidence in active:
            labelled.append({
                "start": start,
                "end": end,
                "label": label,
                "confidence": confidence,
                "embedding": win["embedding"]
            })

    return labelled


def merge_events(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges window predictions with identical labels into event intervals.

    Predictions are first trimmed to the span each window owns (so overlapping
    windows don't overlap in time), then, separately for each label, windows whose
    spans touch or overlap are joined into one event. Different labels are tracked
    independently, so simultaneous sounds produce simultaneous events.

    Args:
        predictions: List of window prediction dictionaries (one per active label per window).

    Returns:
        List of intermediate event dictionaries containing merged time boundaries,
        mean confidence, label, and aggregated embedding vector, sorted by start time.
    """
    if not predictions:
        return []

    spans = _owned_spans(predictions)
    tolerance = 1e-6

    def _create_event(label: str, windows: List[Dict[str, Any]], start: float, end: float) -> Dict[str, Any]:
        # Calculate mean confidence across merged windows
        confidences = [
            float(_extract_window_field(w, ["confidence", "probability", "score"], 0.0))
            for w in windows
        ]
        mean_confidence = round(sum(confidences) / len(confidences), 4)

        # Aggregate window embeddings (mean vector) if available
        embeddings = [
            _extract_window_field(w, ["embedding", "vector", "features"])
            for w in windows
        ]
        valid_embeddings = [e for e in embeddings if e is not None]

        if valid_embeddings:
            mean_embedding = np.mean(valid_embeddings, axis=0).tolist()
        else:
            mean_embedding = None

        return {
            "label": label,
            "start": start,
            "end": end,
            "confidence": mean_confidence,
            "_embedding": mean_embedding
        }

    # Group predictions by label, keeping chronological order within each label
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(predictions, key=lambda w: float(_extract_window_field(w, ["start", "start_time"], 0.0))):
        label = str(_extract_window_field(p, ["label", "predicted_label"], "Unknown"))
        by_label.setdefault(label, []).append(p)

    merged_events = []
    for label, windows in by_label.items():
        current: List[Dict[str, Any]] = []
        cur_start = cur_end = 0.0
        for win in windows:
            w_start = float(_extract_window_field(win, ["start", "start_time"], 0.0))
            w_end = spans[w_start]
            if current and w_start <= cur_end + tolerance:
                current.append(win)
                cur_end = max(cur_end, w_end)
            else:
                if current:
                    merged_events.append(_create_event(label, current, cur_start, cur_end))
                current, cur_start, cur_end = [win], w_start, w_end
        if current:
            merged_events.append(_create_event(label, current, cur_start, cur_end))

    # Sort chronologically by start time
    merged_events.sort(key=lambda e: (e["start"], e["end"]))
    return merged_events


def postprocess_predictions(
    predictions: List[Dict[str, Any]],
    centroids: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Full P3 post-processing pipeline:
    1. Accepts raw sliding-window predictions from P2.
    2. Merges consecutive windows with identical labels into single event intervals.
    3. For 'Unknown' events, calculates distance-based similarity list against known centroids.
    4. Returns clean events matching the required P3->P4 event output specification.

    Args:
        predictions: Raw window predictions from P2.
        centroids: Known-class centroids dictionary from P1.
        top_k: Number of top similar classes to include for Unknown events (default 3).

    Returns:
        List of finalized event dictionaries adhering strictly to:
        {
            "label": str,
            "start": float,
            "end": float,
            "confidence": float,
            "similar_to": list[dict] | None
        }
    """
    merged = merge_events(predictions)
    final_events = []

    for event in merged:
        label = event["label"]
        start = event["start"]
        end = event["end"]
        confidence = event["confidence"]
        embedding = event.get("_embedding")

        if label.lower() == "unknown":
            if centroids and embedding is not None:
                similar_to = compute_unknown_similarity(
                    embedding=embedding,
                    centroids=centroids,
                    top_k=top_k
                )
            else:
                similar_to = []
        else:
            similar_to = None

        final_events.append({
            "label": label,
            "start": start,
            "end": end,
            "confidence": confidence,
            "similar_to": similar_to
        })

    return final_events

"""
P3 Module — Event Merging + Unknown Similarity List
Acoustic Sound Analyzer Project
Assigned Developer: Sarah (P3)

This module implements post-processing for P2 raw sliding-window predictions:
1. Merges consecutive/overlapping windows with identical predicted labels into event intervals.
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


def label_windows(
    windows: List[Dict[str, Any]],
    centroids: Dict[str, Union[List[float], np.ndarray]],
    threshold: float
) -> List[Dict[str, Any]]:
    """
    Converts P2's raw window output ({"start", "end", "probs", "embedding"}) into the
    labelled predictions consumed by merge_events / postprocess_predictions.

    Each window gets its highest-probability class as label, unless the Euclidean
    distance from its embedding to every known centroid exceeds P1's threshold,
    in which case it is labelled "Unknown".
    """
    known = [np.asarray(v, dtype=np.float64) for k, v in centroids.items() if k.lower() != "unknown"]
    centroid_matrix = np.stack(known) if known else None

    labelled = []
    for win in windows:
        probs = win["probs"]
        top_label = max(probs, key=probs.get)
        confidence = float(probs[top_label])

        embedding = np.asarray(win["embedding"], dtype=np.float64)
        if centroid_matrix is not None:
            min_dist = float(np.min(np.linalg.norm(centroid_matrix - embedding, axis=1)))
            if min_dist > threshold:
                top_label = "Unknown"

        labelled.append({
            "start": float(win["start"]),
            "end": float(win["end"]),
            "label": top_label,
            "confidence": confidence,
            "embedding": win["embedding"]
        })

    return labelled


def merge_events(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive/overlapping raw window predictions with identical predicted labels.

    Args:
        predictions: List of raw sliding-window prediction dictionaries from P2.

    Returns:
        List of intermediate event dictionaries containing merged time boundaries,
        mean confidence, label, and aggregated embedding vector.
    """
    if not predictions:
        return []

    merged_events = []
    current_windows = [predictions[0]]

    def _create_event(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        first_win = windows[0]
        last_win = windows[-1]

        start_time = float(_extract_window_field(first_win, ["start", "start_time"], 0.0))
        end_time = float(_extract_window_field(last_win, ["end", "end_time"], 0.0))
        label = str(_extract_window_field(first_win, ["label", "predicted_label"], "Unknown"))

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
            "start": start_time,
            "end": end_time,
            "confidence": mean_confidence,
            "_embedding": mean_embedding
        }

    for win in predictions[1:]:
        curr_label = _extract_window_field(current_windows[-1], ["label", "predicted_label"])
        next_label = _extract_window_field(win, ["label", "predicted_label"])

        # Merge if consecutive window has identical label
        if curr_label == next_label:
            current_windows.append(win)
        else:
            merged_events.append(_create_event(current_windows))
            current_windows = [win]

    if current_windows:
        merged_events.append(_create_event(current_windows))

    # Sort chronologically by start time
    merged_events.sort(key=lambda e: e["start"])
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

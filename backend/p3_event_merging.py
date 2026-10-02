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
    metric: str = "euclidean",
    temperature: Optional[float] = 0.5
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
        temperature: Softness of the distance -> similarity conversion. Similarity is
            proportional to exp(-(d - d_closest) / temperature), so it depends on how much
            further each class is than the closest one. Smaller = sharper. None falls back
            to plain inverse-distance weighting (tends to give near-equal percentages,
            because embedding distances are large compared to the gaps between classes).

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
    if temperature is not None and temperature > 0:
        weights = np.exp(-(distances - distances.min()) / temperature)
    else:
        # Inverse distance weighting with small epsilon to prevent division by zero:
        epsilon = 1e-8
        weights = 1.0 / (distances + epsilon)
    raw_similarities = weights / np.sum(weights)

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
    """Loads the calibrated unknown cutoff from P1's threshold.pt."""
    return load_unknown_config(threshold_path)["threshold"]


def load_unknown_config(threshold_path: str) -> Dict[str, Any]:
    """
    Loads P1's threshold.pt as {"method": str, "threshold": float}. Files written
    before calibration compared methods have no "method" key and use "distance".
    """
    import torch

    payload = torch.load(threshold_path, map_location="cpu")
    return {
        "method": payload.get("method", "distance"),
        "threshold": float(payload["threshold"]),
    }


def load_class_radii(centroids_path: str) -> Optional[Dict[str, float]]:
    """Loads per-class spread from centroids.pt ({label: radius}), or None if absent."""
    import torch

    payload = torch.load(centroids_path, map_location="cpu")
    if "class_radius" not in payload:
        return None
    radii = payload["class_radius"].cpu().numpy()
    return {name: float(radii[i]) for i, name in enumerate(payload["class_names"])}


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
    prob_threshold: float = 0.3,
    method: str = "distance",
    radii: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.3,
    silence_db: Optional[float] = -50.0,
    background_db: Optional[float] = 20.0
) -> List[Dict[str, Any]]:
    """
    Converts P2's raw window output ({"start", "end", "probs", "embedding", "rms_db"})
    into the labelled predictions consumed by merge_events / postprocess_predictions.

    For each window, in order:
    1. Silence / background: it yields nothing if its loudness ("rms_db") is below
       silence_db (absolute), or more than background_db dB quieter than the
       loudest window in the recording (quiet background between the main sounds,
       which the model can't name and would otherwise be flagged Unknown).
    2. Unknown: if its unknown-score (see backend/unknown_scoring.py; method
       "distance", "normalized_distance" or "max_softmax") exceeds P1's threshold,
       it yields a single "Unknown" prediction (confidence = highest class probability).
    3. Uncertain: if its highest class probability is below min_confidence, it
       yields nothing (background the model can't name with any confidence).
    4. Otherwise it yields the highest-probability class, plus every other class
       whose softmax probability is >= prob_threshold, so overlapping sounds
       (e.g. traffic + birds sharing the probability mass) are all kept.

    If threshold is None (P1's threshold.pt not available yet), step 2 is skipped.

    Each prediction's start/end is trimmed to the span the window owns (see
    _owned_spans), so predictions from consecutive windows never overlap.
    """
    from backend.unknown_scoring import unknown_scores

    known_labels = [k for k in centroids if k.lower() != "unknown"]
    centroid_matrix = (
        np.stack([np.asarray(centroids[k], dtype=np.float64) for k in known_labels])
        if known_labels else None
    )
    radius_vec = np.array([radii[k] for k in known_labels]) if radii and known_labels else None
    detect_unknown = threshold is not None and (method == "max_softmax" or centroid_matrix is not None)
    if method == "normalized_distance" and radius_vec is None:
        raise ValueError("normalized_distance needs class radii; rerun backend.compute_centroids.")
    spans = _owned_spans(windows)

    loudness = [w["rms_db"] for w in windows if w.get("rms_db") is not None]
    floor = silence_db
    if background_db is not None and loudness:
        relative_floor = max(loudness) - background_db
        floor = relative_floor if floor is None else max(floor, relative_floor)

    labelled = []
    for win in windows:
        if floor is not None and win.get("rms_db") is not None and win["rms_db"] < floor:
            continue

        probs = win["probs"]
        start = float(win["start"])
        end = spans[start]
        top_label = max(probs, key=probs.get)
        top_prob = float(probs[top_label])

        is_unknown = False
        if detect_unknown:
            score = unknown_scores(
                method,
                embeddings=np.asarray(win["embedding"], dtype=np.float64)[None, :],
                centroids=centroid_matrix,
                radii=radius_vec,
                probs=np.array([list(probs.values())]),
            )[0]
            is_unknown = score > threshold

        if is_unknown:
            active = [("Unknown", top_prob)]
        elif top_prob < min_confidence:
            continue
        else:
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
    top_k: int = 3,
    similarity_temperature: Optional[float] = 0.5
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
        similarity_temperature: See compute_unknown_similarity (default 0.5).

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
                    top_k=top_k,
                    temperature=similarity_temperature
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

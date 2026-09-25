"""
backend/aggregate_report.py
===========================
Location classification and final report generator for the Acoustic Sound Analyzer.

P3 → P4 Integration Contract
----------------------------
P4 consumes the output of P3 (event merging and unknown sound detection).
The input to P4 is a list of sound event dictionaries::

    events: list[dict]

Each event dict contains exactly::

    {
        "label":      str,                    # e.g. 'traffic', 'bird_chirp', or 'Unknown'
        "start":      float,                  # event start in seconds
        "end":        float,                  # event end in seconds
        "confidence": float,                  # sigmoid probability, 0.0 - 1.0
        "similar_to": list[dict] | None,      # only populated when label == 'Unknown',
                                              # None for known classes
    }

When ``label == 'Unknown'``, ``similar_to`` is a list of ranked closest matches::

    [
        {"label": "drilling", "similarity": 0.70},
        {"label": "engine_idling", "similarity": 0.20},
        {"label": "jackhammer", "similarity": 0.10},
    ]

P4 does NOT need to know about window-level sliding inference, temporal smoothing,
event merging, embedding distance metrics, or centroid definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Category Mapping
# ---------------------------------------------------------------------------
# Broad category groupings for location classification.
# Entries are based on typical ESC-50 and UrbanSound8K classes (with esc_ and
# us8k_ prefixes as generated in backend/dataset.py) alongside generic label names.
#
# NOTE: This mapping should be tuned once real label names come back from
# the trained model's checkpoint class_names.
CATEGORY_MAP: Dict[str, List[str]] = {
    "industrial": [
        # Generic / unprefixed sound labels
        "traffic",
        "construction",
        "machinery",
        "engine",
        "engine_idling",
        "drilling",
        "jackhammer",
        "air_conditioner",
        "car_horn",
        "siren",
        "chainsaw",
        "hand_saw",
        "train",
        "helicopter",
        "airplane",
        # UrbanSound8K classes (us8k_ prefix)
        "us8k_drilling",
        "us8k_jackhammer",
        "us8k_engine_idling",
        "us8k_air_conditioner",
        "us8k_car_horn",
        "us8k_siren",
        # ESC-50 classes (esc_ prefix)
        "esc_engine",
        "esc_chainsaw",
        "esc_car_horn",
        "esc_siren",
        "esc_train",
        "esc_helicopter",
        "esc_airplane",
        "esc_hand_saw",
        "esc_washing_machine",
        "esc_vacuum_cleaner",
    ],
    "natural": [
        # Generic / unprefixed sound labels
        "bird",
        "bird_chirp",
        "chirping_birds",
        "wind",
        "insect",
        "insects",
        "crickets",
        "water",
        "water_drops",
        "rain",
        "sea_waves",
        "pouring_water",
        "thunderstorm",
        "natural",
        # UrbanSound8K classes (us8k_ prefix)
        "us8k_dog_bark",
        # ESC-50 classes (esc_ prefix)
        "esc_chirping_birds",
        "esc_crow",
        "esc_crickets",
        "esc_insects",
        "esc_frog",
        "esc_dog",
        "esc_rooster",
        "esc_pig",
        "esc_cow",
        "esc_cat",
        "esc_hen",
        "esc_sheep",
        "esc_rain",
        "esc_sea_waves",
        "esc_water_drops",
        "esc_wind",
        "esc_pouring_water",
        "esc_thunderstorm",
        "esc_crackling_fire",
    ],
}

# Inverted mapping: raw sound label -> broad category ('industrial' | 'natural')
LABEL_TO_CATEGORY: Dict[str, str] = {
    label: category
    for category, labels in CATEGORY_MAP.items()
    for label in labels
}


# ---------------------------------------------------------------------------
# Location Classification
# ---------------------------------------------------------------------------

def classify_location(category_percentages: Dict[str, float]) -> str:
    """Classify the recording location based on sound category percentages.

    Implements Requirement 4 from requirements_new.md:
      - If (traffic + construction + machinery + engine) % > 60 -> 'Industrial'
      - If (bird + wind + insect + natural) % > 60             -> 'Natural habitat'
      - Otherwise                                              -> 'Mixed / Residential'

    Parameters
    ----------
    category_percentages:
        Dictionary mapping category names (e.g. 'industrial', 'natural')
        or specific sound labels (e.g. 'traffic', 'bird') to their percentage
        share across the recording (0 to 100). If percentages are provided
        as proportions (0.0 to 1.0), they will be automatically scaled to 0-100.

    Returns
    -------
    str
        'Industrial', 'Natural habitat', or 'Mixed / Residential'.
    """
    if not category_percentages:
        return "Mixed / Residential"

    # Auto-detect if percentages were provided on a [0.0, 1.0] scale vs [0.0, 100.0]
    max_val = max(category_percentages.values()) if category_percentages else 0.0
    scale = 100.0 if 0.0 < max_val <= 1.0 else 1.0

    # 1. Industrial percentage:
    # Check broad category key 'industrial' or aggregate individual industrial labels
    industrial_labels = set(CATEGORY_MAP.get("industrial", [])) | {
        "traffic", "construction", "machinery", "engine", "engine_idling",
        "drilling", "jackhammer", "air_conditioner", "car_horn", "siren"
    }
    component_industrial = sum(
        val * scale for k, val in category_percentages.items()
        if k.lower() in industrial_labels or any(ind in k.lower() for ind in ("traffic", "construction", "machinery", "engine"))
    )
    broad_industrial = category_percentages.get("industrial", 0.0) * scale
    industrial_pct = max(broad_industrial, component_industrial)

    # 2. Natural percentage:
    # Check broad category keys ('natural', 'natural_habitat') or aggregate natural labels
    natural_labels = set(CATEGORY_MAP.get("natural", [])) | {
        "bird", "bird_chirp", "chirping_birds", "wind", "insect", "insects",
        "crickets", "water", "water_drops", "rain", "sea_waves", "natural"
    }
    component_natural = sum(
        val * scale for k, val in category_percentages.items()
        if k.lower() in natural_labels or any(nat in k.lower() for nat in ("bird", "wind", "insect", "water", "natural"))
    )
    broad_natural = max(
        category_percentages.get("natural", 0.0),
        category_percentages.get("natural_habitat", 0.0),
        category_percentages.get("natural habitat", 0.0),
    ) * scale
    natural_pct = max(broad_natural, component_natural)

    # 3. Decision rule (strictly > 60% as specified in requirements_new.md)
    if industrial_pct > 60.0:
        return "Industrial"
    elif natural_pct > 60.0:
        return "Natural habitat"
    else:
        return "Mixed / Residential"


# ---------------------------------------------------------------------------
# Event Aggregation
# ---------------------------------------------------------------------------

def get_category_for_label(label: str) -> str:
    """Resolve a raw sound label to its broad category ('industrial', 'natural', or 'other')."""
    raw_lower = label.lower().strip()
    if raw_lower in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[raw_lower]

    # Check without dataset prefix if present (e.g. us8k_drilling -> drilling)
    for prefix in ("us8k_", "esc_"):
        if raw_lower.startswith(prefix):
            unprefixed = raw_lower[len(prefix):]
            if unprefixed in LABEL_TO_CATEGORY:
                return LABEL_TO_CATEGORY[unprefixed]

    # Partial substring match against category members
    for cat, labels in CATEGORY_MAP.items():
        for l in labels:
            clean_l = l.replace("us8k_", "").replace("esc_", "").lower()
            if clean_l and (clean_l in raw_lower or raw_lower in clean_l):
                return cat

    return "other"


def aggregate_events(
    events: List[Dict[str, Any]],
    total_duration: float,
) -> Dict[str, Any]:
    """Aggregate sound events across the entire recording and classify location.

    Parameters
    ----------
    events:
        List of event dictionaries conforming to the P3 -> P4 Integration Contract.
        Each dict has keys: 'label', 'start', 'end', 'confidence', 'similar_to'.
    total_duration:
        Total duration of the audio recording in seconds.

    Returns
    -------
    dict
        {
            'label_percentages': dict[str, float],
            'category_percentages': dict[str, float],
            'location': str,
            'dominant_category': str,
            'dominant_pct': float,
        }
    """
    # Guard against invalid or missing total_duration
    if total_duration <= 0.0:
        total_duration = max((float(e.get("end", 0.0)) for e in events), default=0.0)

    # 1. Sum duration per label across all events
    label_durations: Dict[str, float] = {}
    for ev in events:
        label = str(ev.get("label", "Unknown"))
        start = float(ev.get("start", 0.0))
        end = float(ev.get("end", 0.0))
        dur = max(0.0, end - start)
        label_durations[label] = label_durations.get(label, 0.0) + dur

    # 2. Convert each label's total duration into percentage of total_duration
    label_percentages: Dict[str, float] = {}
    if total_duration > 0.0:
        for label, dur in label_durations.items():
            label_percentages[label] = round((dur / total_duration) * 100.0, 2)

    # 3. Roll individual labels up into broad categories using CATEGORY_MAP
    category_percentages: Dict[str, float] = {
        "industrial": 0.0,
        "natural": 0.0,
    }
    for label, pct in label_percentages.items():
        cat = get_category_for_label(label)
        if cat in category_percentages:
            category_percentages[cat] = round(category_percentages[cat] + pct, 2)
        else:
            category_percentages[cat] = round(category_percentages.get(cat, 0.0) + pct, 2)

    # 4. Call classify_location() on the category percentages
    location = classify_location(category_percentages)

    # 5. Determine dominant category and dominant percentage
    if category_percentages and any(v > 0.0 for v in category_percentages.values()):
        dominant_cat, dominant_val = max(
            category_percentages.items(),
            key=lambda item: item[1],
        )
        dominant_category = dominant_cat
        dominant_pct = round(dominant_val, 2)
    else:
        dominant_category = "none"
        dominant_pct = 0.0

    return {
        "label_percentages": label_percentages,
        "category_percentages": category_percentages,
        "location": location,
        "dominant_category": dominant_category,
        "dominant_pct": dominant_pct,
    }


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to MM:SS format."""
    total_sec = max(0, int(round(seconds)))
    minutes = total_sec // 60
    sec = total_sec % 60
    return f"{minutes:02d}:{sec:02d}"


def format_label_name(raw_label: str) -> str:
    """Format raw sound label for display (e.g. 'bird_chirp' -> 'Bird chirp')."""
    if str(raw_label).strip().lower() == "unknown":
        return "Unknown"
    cleaned = str(raw_label).strip()
    for prefix in ("us8k_", "esc_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else raw_label


def format_report(
    events: List[Dict[str, Any]],
    aggregation_result: Dict[str, Any],
) -> str:
    """Format detected events and location classification into the final text report.

    Produces output conforming exactly to the Final Output Format in requirements_new.md:

        1. Detected sounds:
           - Traffic       | 00:00–00:45 | confidence: 0.91
           - Bird chirp    | 00:10–00:30 | confidence: 0.84
           - Unknown       | 02:13–02:18 | closest matches: 70% drilling, 20% engine idling, 10% jackhammer

        2. Location classification: Industrial (traffic + construction = 68% of total audio)

    Parameters
    ----------
    events:
        List of event dicts with 'label', 'start', 'end', 'confidence', 'similar_to'.
    aggregation_result:
        Output dict from aggregate_events() containing 'location', 'dominant_category',
        'dominant_pct', and 'label_percentages'.

    Returns
    -------
    str
        Single formatted report string.
    """
    lines: List[str] = ["1. Detected sounds:"]

    if not events:
        lines.append("   (No sounds detected)")
    else:
        # Determine column width for labels (at least 13 chars so label + space matches example alignment)
        formatted_labels = [format_label_name(str(ev.get("label", "Unknown"))) for ev in events]
        max_label_len = max([len(l) for l in formatted_labels], default=13)
        col_width = max(13, max_label_len)

        for ev, display_label in zip(events, formatted_labels):
            start_str = format_timestamp(float(ev.get("start", 0.0)))
            end_str = format_timestamp(float(ev.get("end", 0.0)))
            time_range = f"{start_str}\u2013{end_str}"

            raw_label = str(ev.get("label", "")).strip().lower()
            is_unknown = raw_label == "unknown"

            if is_unknown:
                similar_to = ev.get("similar_to") or []
                if similar_to:
                    matches = []
                    for match in similar_to:
                        m_label = str(match.get("label", "")).replace("_", " ").strip()
                        for prefix in ("us8k_", "esc_"):
                            if m_label.startswith(prefix):
                                m_label = m_label[len(prefix):]
                        m_sim = float(match.get("similarity", 0.0))
                        pct = int(round(m_sim * 100)) if m_sim <= 1.0 else int(round(m_sim))
                        matches.append(f"{pct}% {m_label}")
                    detail = f"closest matches: {', '.join(matches)}"
                else:
                    detail = "closest matches: none"
            else:
                conf = float(ev.get("confidence", 0.0))
                detail = f"confidence: {conf:.2f}"

            padded_label = display_label.ljust(col_width)
            lines.append(f"   - {padded_label} | {time_range} | {detail}")

    # Section 2: Location classification
    location = aggregation_result.get("location", "Mixed / Residential")
    dominant_cat = aggregation_result.get("dominant_category", "")
    dominant_pct = float(aggregation_result.get("dominant_pct", 0.0))

    # Format dominant percentage string (e.g. 68% or 68.2%)
    if dominant_pct == int(dominant_pct):
        pct_str = f"{int(dominant_pct)}"
    else:
        pct_str = f"{dominant_pct:g}"

    # Determine contributing sound labels for the dominant category
    label_percentages = aggregation_result.get("label_percentages", {})
    contributing_labels: List[str] = []
    if dominant_cat and dominant_cat != "none":
        for lbl, pct in label_percentages.items():
            if pct > 0 and get_category_for_label(lbl) == dominant_cat:
                clean_lbl = lbl
                for prefix in ("us8k_", "esc_"):
                    if clean_lbl.startswith(prefix):
                        clean_lbl = clean_lbl[len(prefix):]
                clean_lbl = clean_lbl.replace("_", " ").lower()
                contributing_labels.append(clean_lbl)

    if contributing_labels:
        breakdown = f"{' + '.join(contributing_labels)} = {pct_str}% of total audio"
    elif dominant_cat and dominant_cat != "none" and dominant_pct > 0:
        breakdown = f"{dominant_cat} = {pct_str}% of total audio"
    else:
        breakdown = "no dominant sound category"

    lines.append("")
    lines.append(f"2. Location classification: {location} ({breakdown})")

    return "\n".join(lines)




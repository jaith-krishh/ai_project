"""
Standalone Demo & Test Script for P3 — Event Merging + Similarity List
Acoustic Sound Analyzer Project

Run this script to verify P3 post-processing functionality:
  py tests/run_p3_demo.py
"""

import sys
import os
import json

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.p3_event_merging import postprocess_predictions


def run_demo():
    print("============================================================")
    print("   ACOUSTIC SOUND ANALYZER — P3 EVENT MERGING DEMO          ")
    print("============================================================\n")

    # 1. Define Known-Class Centroids (P1)
    centroids = {
        "drilling": [0.85, 0.10, 0.05],
        "engine_idling": [0.20, 0.70, 0.10],
        "jackhammer": [0.60, 0.15, 0.25],
        "traffic": [0.10, 0.30, 0.60],
        "siren": [0.05, 0.05, 0.90],
        "Unknown": [9.99, 9.99, 9.99]  # Excluded from similarity list
    }

    print("--- STEP 1: P1 Known Centroids ---")
    print(json.dumps({k: v for k, v in centroids.items() if k != "Unknown"}, indent=2))
    print()

    # 2. Raw P2 Sliding-Window Predictions
    raw_p2_predictions = [
        # Event 1: traffic (3 consecutive overlapping 2s windows, hop=1s)
        {"start": 0.0, "end": 2.0, "label": "traffic", "confidence": 0.90, "embedding": [0.10, 0.30, 0.60]},
        {"start": 1.0, "end": 3.0, "label": "traffic", "confidence": 0.94, "embedding": [0.12, 0.28, 0.60]},
        {"start": 2.0, "end": 4.0, "label": "traffic", "confidence": 0.88, "embedding": [0.08, 0.32, 0.60]},

        # Event 2: drilling (2 consecutive windows)
        {"start": 4.0, "end": 6.0, "label": "drilling", "confidence": 0.92, "embedding": [0.85, 0.10, 0.05]},
        {"start": 5.0, "end": 7.0, "label": "drilling", "confidence": 0.96, "embedding": [0.86, 0.09, 0.05]},

        # Event 3: Unknown sound (4 consecutive windows, embedding closest to drilling/jackhammer)
        {"start": 7.0, "end": 9.0, "label": "Unknown", "confidence": 0.65, "embedding": [0.75, 0.15, 0.10]},
        {"start": 8.0, "end": 10.0, "label": "Unknown", "confidence": 0.70, "embedding": [0.74, 0.16, 0.10]},
        {"start": 9.0, "end": 11.0, "label": "Unknown", "confidence": 0.68, "embedding": [0.76, 0.14, 0.10]},
        {"start": 10.0, "end": 12.0, "label": "Unknown", "confidence": 0.69, "embedding": [0.75, 0.15, 0.10]},

        # Event 4: siren (2 consecutive windows)
        {"start": 12.0, "end": 14.0, "label": "siren", "confidence": 0.98, "embedding": [0.05, 0.05, 0.90]},
        {"start": 13.0, "end": 15.0, "label": "siren", "confidence": 0.96, "embedding": [0.05, 0.05, 0.90]}
    ]

    print("--- STEP 2: Raw P2 Sliding-Window Predictions Input ---")
    print(f"Total raw input windows: {len(raw_p2_predictions)}")
    for win in raw_p2_predictions:
        print(f"  [{win['start']:4.1f}s - {win['end']:4.1f}s] Label: {win['label']:12s} Conf: {win['confidence']:.2f}")
    print()

    # 3. P3 Post-processing (Event Merging + Unknown Similarity)
    p3_events = postprocess_predictions(
        predictions=raw_p2_predictions,
        centroids=centroids,
        top_k=3
    )

    print("--- STEP 3: P3 Output Merged Events ---")
    print(json.dumps(p3_events, indent=2))
    print()

    print("P3 Post-Processing Complete. Output ready for P4 consumption.")


if __name__ == "__main__":
    run_demo()

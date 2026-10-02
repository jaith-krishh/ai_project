"""
Unit Tests for P3 — Event Merging + Similarity List
Acoustic Sound Analyzer Project
"""

import unittest
import numpy as np
from backend.p3_event_merging import (
    merge_events,
    compute_unknown_similarity,
    postprocess_predictions,
    label_windows
)


class TestP3EventMerging(unittest.TestCase):

    def setUp(self):
        # Known class centroids sample data
        self.centroids = {
            "drilling": np.array([1.0, 0.0, 0.0]),
            "engine_idling": np.array([0.0, 1.0, 0.0]),
            "jackhammer": np.array([0.0, 0.0, 1.0]),
            "traffic": np.array([0.5, 0.5, 0.0]),
            "Unknown": np.array([9.0, 9.0, 9.0])  # Should be excluded from similarity
        }

    def test_merge_consecutive_same_labels(self):
        """Test multiple consecutive windows with the same label become one event."""
        windows = [
            {"start": 0.0, "end": 2.0, "label": "traffic", "confidence": 0.90, "embedding": [0.5, 0.5, 0.0]},
            {"start": 1.0, "end": 3.0, "label": "traffic", "confidence": 0.94, "embedding": [0.5, 0.5, 0.0]},
            {"start": 2.0, "end": 4.0, "label": "traffic", "confidence": 0.86, "embedding": [0.5, 0.5, 0.0]}
        ]
        events = postprocess_predictions(windows, centroids=self.centroids)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["label"], "traffic")
        self.assertEqual(event["start"], 0.0)
        self.assertEqual(event["end"], 4.0)
        self.assertAlmostEqual(event["confidence"], 0.90, places=4)
        self.assertIsNone(event["similar_to"])

    def test_separate_different_labels(self):
        """Test different consecutive labels remain separate events."""
        windows = [
            {"start": 0.0, "end": 2.0, "label": "traffic", "confidence": 0.90, "embedding": [0.5, 0.5, 0.0]},
            {"start": 2.0, "end": 4.0, "label": "drilling", "confidence": 0.85, "embedding": [1.0, 0.0, 0.0]},
            {"start": 4.0, "end": 6.0, "label": "traffic", "confidence": 0.92, "embedding": [0.5, 0.5, 0.0]}
        ]
        events = postprocess_predictions(windows, centroids=self.centroids)

        self.assertEqual(len(events), 3)
        self.assertEqual([e["label"] for e in events], ["traffic", "drilling", "traffic"])
        self.assertEqual(events[0]["start"], 0.0)
        self.assertEqual(events[0]["end"], 2.0)
        self.assertEqual(events[1]["start"], 2.0)
        self.assertEqual(events[1]["end"], 4.0)
        self.assertEqual(events[2]["start"], 4.0)
        self.assertEqual(events[2]["end"], 6.0)

    def test_unknown_event_similarity_list(self):
        """Test an Unknown event produces a ranked top 3-4 similarity list."""
        # Embedding closest to drilling [1.0, 0.0, 0.0]
        unknown_embedding = [0.9, 0.1, 0.0]
        windows = [
            {"start": 12.0, "end": 14.0, "label": "Unknown", "confidence": 0.65, "embedding": unknown_embedding},
            {"start": 13.0, "end": 15.0, "label": "Unknown", "confidence": 0.71, "embedding": unknown_embedding}
        ]
        events = postprocess_predictions(windows, centroids=self.centroids, top_k=3)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["label"], "Unknown")
        self.assertEqual(event["start"], 12.0)
        self.assertEqual(event["end"], 15.0)
        self.assertAlmostEqual(event["confidence"], 0.68, places=2)

        similar_to = event["similar_to"]
        self.assertIsNotNone(similar_to)
        self.assertLessEqual(len(similar_to), 3)

        # Ensure drilling is top ranked
        top_label = similar_to[0]["label"]
        self.assertEqual(top_label, "drilling")

        # Check normalization: sum of similarity values equals 1.0 (or 100%)
        total_sim = sum(item["similarity"] for item in similar_to)
        self.assertAlmostEqual(total_sim, 1.0, places=2)

        # Check 'Unknown' is excluded from similar_to list
        similar_labels = [item["label"] for item in similar_to]
        self.assertNotIn("Unknown", similar_labels)

    def test_chronological_ordering(self):
        """Test that resulting events preserve chronological order."""
        windows = [
            {"start": 0.0, "end": 2.0, "label": "traffic", "confidence": 0.90, "embedding": [0.5, 0.5, 0.0]},
            {"start": 2.0, "end": 5.0, "label": "drilling", "confidence": 0.80, "embedding": [1.0, 0.0, 0.0]},
            {"start": 5.0, "end": 8.0, "label": "Unknown", "confidence": 0.60, "embedding": [0.0, 0.0, 1.0]}
        ]
        events = postprocess_predictions(windows, centroids=self.centroids)

        start_times = [e["start"] for e in events]
        self.assertEqual(start_times, sorted(start_times))

    def test_flexible_p2_input_field_names(self):
        """Test compatibility with P2 field name variations (start_time, probability, etc.)."""
        windows = [
            {"start_time": 0.0, "end_time": 2.0, "predicted_label": "siren", "probability": 0.95, "vector": [0.1, 0.2]},
            {"start_time": 1.5, "end_time": 3.5, "predicted_label": "siren", "probability": 0.85, "vector": [0.1, 0.2]}
        ]
        events = postprocess_predictions(windows)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "siren")
        self.assertEqual(events[0]["start"], 0.0)
        self.assertEqual(events[0]["end"], 3.5)
        self.assertEqual(events[0]["confidence"], 0.90)

    def test_label_windows_from_p2_output(self):
        """Test real P2 window dicts (probs + embedding) are labelled, with far embeddings flagged Unknown."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.2}, "embedding": [1.0, 0.0, 0.0]},
            {"start": 2.0, "end": 6.0, "probs": {"drilling": 0.2, "traffic": 0.8}, "embedding": [0.5, 0.5, 0.0]},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.6, "traffic": 0.1}, "embedding": [9.0, 9.0, 9.0]}
        ]
        labelled = label_windows(windows, centroids=self.centroids, threshold=1.0)

        self.assertEqual([w["label"] for w in labelled], ["drilling", "traffic", "Unknown"])
        self.assertAlmostEqual(labelled[2]["confidence"], 0.6)

        events = postprocess_predictions(labelled, centroids=self.centroids)
        self.assertEqual(events[2]["label"], "Unknown")
        self.assertIsNotNone(events[2]["similar_to"])

    def test_overlapping_windows_do_not_double_count(self):
        """Test 4s windows every 2s produce non-overlapping events when the label changes."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0]},
            {"start": 2.0, "end": 6.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0]},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.1, "traffic": 0.9}, "embedding": [0.5, 0.5, 0.0]},
            {"start": 6.0, "end": 10.0, "probs": {"drilling": 0.1, "traffic": 0.9}, "embedding": [0.5, 0.5, 0.0]}
        ]
        events = postprocess_predictions(label_windows(windows, self.centroids, threshold=1.0), self.centroids)

        self.assertEqual([(e["label"], e["start"], e["end"]) for e in events],
                         [("drilling", 0.0, 4.0), ("traffic", 4.0, 10.0)])
        self.assertAlmostEqual(sum(e["end"] - e["start"] for e in events), 10.0)

    def test_simultaneous_sounds_produce_parallel_events(self):
        """Test multi-label windows keep overlapping sounds, and gaps split events of the same label."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.8}, "embedding": [0.5, 0.5, 0.0]},
            {"start": 2.0, "end": 6.0, "probs": {"drilling": 0.1, "traffic": 0.7}, "embedding": [0.5, 0.5, 0.0]},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.1, "traffic": 0.2, "siren": 0.7}, "embedding": [0.5, 0.5, 0.0]},
            {"start": 6.0, "end": 10.0, "probs": {"drilling": 0.6, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0]}
        ]
        events = postprocess_predictions(label_windows(windows, self.centroids, threshold=1.0), self.centroids)

        self.assertEqual([(e["label"], e["start"], e["end"]) for e in events],
                         [("drilling", 0.0, 2.0), ("traffic", 0.0, 4.0), ("siren", 4.0, 6.0), ("drilling", 6.0, 10.0)])
        self.assertAlmostEqual(events[1]["confidence"], 0.75)

    def test_top_class_kept_and_threshold_optional(self):
        """Test a window keeps its top class even below prob_threshold, and threshold=None skips Unknown."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.35, "traffic": 0.2, "siren": 0.2}, "embedding": [9.0, 9.0, 9.0]}
        ]
        labelled = label_windows(windows, self.centroids, threshold=None, prob_threshold=0.5)
        self.assertEqual([w["label"] for w in labelled], ["drilling"])

    def test_silent_and_uncertain_windows_skipped(self):
        """Test quiet windows and windows below min_confidence produce no events."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0], "rms_db": -20.0},
            {"start": 2.0, "end": 6.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0], "rms_db": -80.0},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.22, "traffic": 0.2}, "embedding": [1.0, 0.0, 0.0], "rms_db": -20.0}
        ]
        labelled = label_windows(windows, self.centroids, threshold=1.0)
        self.assertEqual([(w["label"], w["start"]) for w in labelled], [("drilling", 0.0)])

    def test_max_softmax_method(self):
        """Test max_softmax flags a low-confidence window as Unknown instead of dropping it."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [9.0, 9.0, 9.0]},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.25, "traffic": 0.2}, "embedding": [1.0, 0.0, 0.0]}
        ]
        labelled = label_windows(windows, self.centroids, threshold=0.5, method="max_softmax")
        self.assertEqual([w["label"] for w in labelled], ["drilling", "Unknown"])

    def test_normalized_distance_method(self):
        """Test normalized_distance uses each class's spread: same distance is known for a wide class, unknown for a tight one."""
        centroids = {"wide": [0.0, 0.0], "tight": [10.0, 0.0]}
        radii = {"wide": 4.0, "tight": 0.5}
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"wide": 0.9, "tight": 0.1}, "embedding": [3.0, 0.0]},
            {"start": 4.0, "end": 8.0, "probs": {"wide": 0.1, "tight": 0.9}, "embedding": [10.0, 3.0]}
        ]
        labelled = label_windows(windows, centroids, threshold=2.0, method="normalized_distance", radii=radii)
        self.assertEqual([w["label"] for w in labelled], ["wide", "Unknown"])


    def test_similarity_reflects_distance_gaps(self):
        """Test a clearly closest class dominates, while near-ties stay close to even."""
        centroids = {"a": [5.0, 0.0], "b": [0.0, 5.8], "c": [-6.6, 0.0], "d": [0.0, -9.0]}
        clear = compute_unknown_similarity([0.0, 0.0], centroids, top_k=3)
        self.assertEqual([x["label"] for x in clear], ["a", "b", "c"])
        self.assertGreater(clear[0]["similarity"], 0.7)
        self.assertAlmostEqual(sum(x["similarity"] for x in clear), 1.0, places=2)

        tie = compute_unknown_similarity([0.0, 0.0], {"a": [7.28, 0.0], "b": [0.0, 7.34], "c": [-7.64, 0.0]}, top_k=3)
        self.assertLess(tie[0]["similarity"], 0.5)

        legacy = compute_unknown_similarity([0.0, 0.0], centroids, top_k=3, temperature=None)
        self.assertLess(legacy[0]["similarity"], 0.5)


    def test_quiet_background_relative_to_recording_is_skipped(self):
        """Test a window far quieter than the recording's loudest part is background, even above silence_db."""
        windows = [
            {"start": 0.0, "end": 4.0, "probs": {"drilling": 0.9, "traffic": 0.1}, "embedding": [1.0, 0.0, 0.0], "rms_db": -15.0},
            {"start": 4.0, "end": 8.0, "probs": {"drilling": 0.2, "traffic": 0.2}, "embedding": [1.0, 0.0, 0.0], "rms_db": -45.0},
            {"start": 8.0, "end": 12.0, "probs": {"drilling": 0.1, "traffic": 0.8}, "embedding": [0.5, 0.5, 0.0], "rms_db": -30.0}
        ]
        labelled = label_windows(windows, self.centroids, threshold=0.5, method="max_softmax")
        self.assertEqual([w["label"] for w in labelled], ["drilling", "traffic"])

        kept = label_windows(windows, self.centroids, threshold=0.5, method="max_softmax", background_db=None)
        self.assertEqual([w["label"] for w in kept], ["drilling", "Unknown", "traffic"])


if __name__ == "__main__":
    unittest.main()

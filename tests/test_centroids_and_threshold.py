"""
tests/test_centroids_and_threshold.py
======================================
Unit and integration tests for:
  - backend/compute_centroids.py  (P1 – centroid computation)
  - backend/calibrate_unknown.py  (P1 – unknown-threshold calibration)

Run with:
    python -m pytest tests/test_centroids_and_threshold.py -v

All tests that require the trained checkpoint are automatically skipped
when the checkpoint is absent so the test suite can run in CI without data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Make sure the project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.model import AudioSEDNet
from backend.utils import SAMPLE_RATE, TARGET_LENGTH, audio_window_to_mel
from backend.compute_centroids import compute_centroids
from backend.calibrate_unknown import (
    calibrate_unknown,
    min_centroid_distances,
    _embed_audio_array,
    _embed_synthetic_unknown,
    _find_best_threshold,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = os.path.join("outputs", "checkpoints", "best_model.pt")
DEFAULT_CENTROIDS = os.path.join("outputs", "centroids.pt")
CHECKPOINT_PRESENT = os.path.exists(DEFAULT_CHECKPOINT)
CENTROIDS_PRESENT = os.path.exists(DEFAULT_CENTROIDS)

NUM_CLASSES = 4  # dummy model size for unit tests
EMB_DIM = 128


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_dummy_model(num_classes: int = NUM_CLASSES) -> AudioSEDNet:
    """Untrained AudioSEDNet on CPU, in eval mode with no grad."""
    model = AudioSEDNet(num_classes=num_classes)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_random_centroids(num_classes: int = NUM_CLASSES, dim: int = EMB_DIM) -> torch.Tensor:
    """Random centroid matrix."""
    return torch.randn(num_classes, dim)


def _make_sine(duration: float = 4.0) -> np.ndarray:
    """440 Hz sine wave as float32."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ===========================================================================
# Tests for min_centroid_distances
# ===========================================================================

class TestMinCentroidDistances:
    """Unit tests for the distance-computation helper."""

    def test_output_shape(self):
        N, C, D = 10, 4, 128
        embs = torch.randn(N, D)
        centroids = torch.randn(C, D)
        dists = min_centroid_distances(embs, centroids)
        assert dists.shape == (N,), f"Expected ({N},), got {dists.shape}"

    def test_perfect_match_is_zero(self):
        """An embedding identical to a centroid must have distance 0."""
        centroids = torch.randn(4, EMB_DIM)
        # Use the first centroid as the query
        emb = centroids[0:1]
        dists = min_centroid_distances(emb, centroids)
        assert float(dists[0]) == pytest.approx(0.0, abs=1e-5)

    def test_distances_are_non_negative(self):
        embs = torch.randn(20, EMB_DIM)
        centroids = torch.randn(5, EMB_DIM)
        dists = min_centroid_distances(embs, centroids)
        assert (dists >= 0).all(), "Distances must be non-negative"

    def test_far_point_has_large_distance(self):
        """A vector at (100, 100, …) must be far from centroids near the origin."""
        centroids = torch.zeros(4, EMB_DIM)  # all at origin
        emb = torch.ones(1, EMB_DIM) * 100.0
        dists = min_centroid_distances(emb, centroids)
        # Euclidean distance = 100 * sqrt(128) ≈ 1131
        assert float(dists[0]) > 100.0

    def test_works_with_single_centroid(self):
        embs = torch.randn(5, EMB_DIM)
        centroids = torch.randn(1, EMB_DIM)
        dists = min_centroid_distances(embs, centroids)
        assert dists.shape == (5,)


# ===========================================================================
# Tests for _find_best_threshold
# ===========================================================================

class TestFindBestThreshold:
    """Unit tests for the threshold-sweep logic."""

    def _make_well_separated_dists(self):
        """Known distances in [0, 5], unknown distances in [10, 15]."""
        rng = np.random.default_rng(0)
        known = torch.tensor(rng.uniform(0, 5, 100).astype(np.float32))
        unknown = torch.tensor(rng.uniform(10, 15, 100).astype(np.float32))
        return known, unknown

    def test_returns_two_floats(self):
        known, unknown = self._make_well_separated_dists()
        thresh, f1 = _find_best_threshold(known, unknown)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)

    def test_f1_near_one_for_separated_distributions(self):
        """When known and unknown are well-separated, best F1 should be ~1."""
        known, unknown = self._make_well_separated_dists()
        thresh, f1 = _find_best_threshold(known, unknown)
        assert f1 > 0.95, f"Expected F1 ≈ 1 for well-separated dists, got {f1:.4f}"

    def test_threshold_between_distributions(self):
        """Threshold must lie between the two distributions (max known < thresh < min unknown)."""
        known, unknown = self._make_well_separated_dists()
        thresh, _ = _find_best_threshold(known, unknown)
        # known in [0, 5], unknown in [10, 15]; threshold should split them
        # Allow a small margin for discrete sweep granularity
        assert float(known.max()) - 0.1 <= thresh <= float(unknown.min()) + 0.1, (
            f"Threshold {thresh:.4f} should separate known (max={known.max():.2f}) "
            f"from unknown (min={unknown.min():.2f})"
        )

    def test_f1_non_negative(self):
        """F1 is always ≥ 0."""
        known = torch.rand(50) * 3
        unknown = torch.rand(50) * 3  # overlapping – deliberately hard
        _, f1 = _find_best_threshold(known, unknown)
        assert f1 >= 0.0

    def test_output_stable_across_calls(self):
        """Same inputs → same outputs (deterministic sweep)."""
        known, unknown = self._make_well_separated_dists()
        t1, f1_1 = _find_best_threshold(known, unknown)
        t2, f1_2 = _find_best_threshold(known, unknown)
        assert t1 == pytest.approx(t2)
        assert f1_1 == pytest.approx(f1_2)


# ===========================================================================
# Tests for _embed_audio_array
# ===========================================================================

class TestEmbedAudioArray:
    """Unit tests for the raw-audio → embedding helper."""

    def test_output_shape(self):
        model = _make_dummy_model()
        audio = _make_sine(4.0)
        emb = _embed_audio_array(audio, model, torch.device("cpu"))
        assert emb.shape == (EMB_DIM,), f"Expected ({EMB_DIM},), got {emb.shape}"

    def test_output_is_cpu_tensor(self):
        model = _make_dummy_model()
        audio = _make_sine(4.0)
        emb = _embed_audio_array(audio, model, torch.device("cpu"))
        assert emb.device.type == "cpu"

    def test_no_nan_values(self):
        model = _make_dummy_model()
        audio = _make_sine(4.0)
        emb = _embed_audio_array(audio, model, torch.device("cpu"))
        assert not torch.isnan(emb).any(), "Embedding must not contain NaN"

    def test_short_audio_handled(self):
        """Audio shorter than TARGET_LENGTH must be padded successfully."""
        model = _make_dummy_model()
        audio = _make_sine(1.0)  # shorter than 4 s window
        emb = _embed_audio_array(audio, model, torch.device("cpu"))
        assert emb.shape == (EMB_DIM,)

    def test_values_are_non_negative(self):
        """fc_embedding + ReLU → embeddings must be ≥ 0."""
        model = _make_dummy_model()
        audio = _make_sine(4.0)
        emb = _embed_audio_array(audio, model, torch.device("cpu"))
        assert (emb >= 0).all(), "Embeddings after ReLU must be non-negative"


# ===========================================================================
# Tests for _embed_synthetic_unknown
# ===========================================================================

class TestEmbedSyntheticUnknown:
    """Unit tests for the synthetic unknown clip generator."""

    def test_output_shape(self):
        model = _make_dummy_model()
        n = 6
        embs = _embed_synthetic_unknown(n, model, torch.device("cpu"))
        assert embs.shape == (n, EMB_DIM), f"Expected ({n}, {EMB_DIM}), got {embs.shape}"

    def test_deterministic(self):
        """Same seed → same embeddings (for reproducibility)."""
        model = _make_dummy_model()
        embs1 = _embed_synthetic_unknown(6, model, torch.device("cpu"))
        embs2 = _embed_synthetic_unknown(6, model, torch.device("cpu"))
        assert torch.allclose(embs1, embs2)

    def test_no_nan(self):
        model = _make_dummy_model()
        embs = _embed_synthetic_unknown(6, model, torch.device("cpu"))
        assert not torch.isnan(embs).any()


# ===========================================================================
# Integration tests: compute_centroids  (require real data + checkpoint)
# ===========================================================================

class TestComputeCentroids:
    """Integration tests for the full centroid-computation pipeline."""

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT,
        reason="No trained checkpoint at outputs/checkpoints/best_model.pt",
    )
    def test_centroids_pt_created(self, tmp_path):
        """compute_centroids() must write a .pt file."""
        out = str(tmp_path / "centroids_test.pt")
        # Use synth data if real data is absent; test only output-file creation
        # with checkpoint-only path (will fail if no data dir – that's expected
        # and acceptable since data dir is not part of this test's scope)
        try:
            compute_centroids(out_path=out)
        except RuntimeError as exc:
            if "No audio files" in str(exc):
                pytest.skip("No audio data dir available – skipping integration test")
            raise
        assert os.path.exists(out), f"Expected {out} to be created"

    @pytest.mark.skipif(
        not CENTROIDS_PRESENT,
        reason="centroids.pt not found at outputs/centroids.pt",
    )
    def test_centroids_shape(self):
        """Pre-computed centroids.pt must have shape (num_classes, 128)."""
        payload = torch.load(DEFAULT_CENTROIDS, map_location="cpu", weights_only=False)
        centroids = payload["centroids"]
        class_names = payload["class_names"]
        assert centroids.shape[0] == len(class_names), (
            f"Centroid rows ({centroids.shape[0]}) != num classes ({len(class_names)})"
        )
        assert centroids.shape[1] == EMB_DIM, (
            f"Embedding dim must be {EMB_DIM}, got {centroids.shape[1]}"
        )

    @pytest.mark.skipif(
        not CENTROIDS_PRESENT,
        reason="centroids.pt not found at outputs/centroids.pt",
    )
    def test_centroids_finite(self):
        """All centroid values must be finite (no NaN, no Inf)."""
        payload = torch.load(DEFAULT_CENTROIDS, map_location="cpu", weights_only=False)
        centroids = payload["centroids"]
        assert torch.isfinite(centroids).all(), "Centroids contain NaN or Inf values"

    @pytest.mark.skipif(
        not CENTROIDS_PRESENT,
        reason="centroids.pt not found at outputs/centroids.pt",
    )
    def test_centroids_non_negative(self):
        """After fc_embedding + ReLU, centroids must be ≥ 0."""
        payload = torch.load(DEFAULT_CENTROIDS, map_location="cpu", weights_only=False)
        centroids = payload["centroids"]
        assert (centroids >= 0).all(), "Centroids must be non-negative (ReLU activation)"


# ===========================================================================
# Integration tests: calibrate_unknown  (require checkpoint)
# ===========================================================================

class TestCalibrateUnknown:
    """Integration tests for the threshold-calibration pipeline."""

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_threshold_pt_created_synth(self, tmp_path):
        """calibrate_unknown() with synth clips must write threshold.pt."""
        out = str(tmp_path / "threshold_test.pt")
        calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        assert os.path.exists(out), f"Expected {out} to be created"

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_threshold_payload_keys(self, tmp_path):
        """Output payload must have required keys."""
        out = str(tmp_path / "threshold_test.pt")
        payload = calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        required_keys = {"threshold", "metric", "best_f1", "known_dists", "unknown_dists"}
        assert required_keys.issubset(set(payload.keys())), (
            f"Missing keys: {required_keys - set(payload.keys())}"
        )

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_threshold_is_finite(self, tmp_path):
        out = str(tmp_path / "threshold_test.pt")
        payload = calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        assert np.isfinite(payload["threshold"]), "Threshold must be a finite number"

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_f1_in_range(self, tmp_path):
        out = str(tmp_path / "threshold_test.pt")
        payload = calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        assert 0.0 <= payload["best_f1"] <= 1.0, (
            f"F1 must be in [0, 1], got {payload['best_f1']}"
        )

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_known_dists_shape(self, tmp_path):
        out = str(tmp_path / "threshold_test.pt")
        payload = calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        assert payload["known_dists"].ndim == 1, "known_dists must be a 1-D tensor"
        assert len(payload["known_dists"]) > 0

    @pytest.mark.skipif(
        not CHECKPOINT_PRESENT or not CENTROIDS_PRESENT,
        reason="Requires both best_model.pt and centroids.pt",
    )
    def test_distances_non_negative(self, tmp_path):
        out = str(tmp_path / "threshold_test.pt")
        payload = calibrate_unknown(out_path=out, synth_clips=20, data_dir="nonexistent_dir")
        assert (payload["known_dists"] >= 0).all()
        assert (payload["unknown_dists"] >= 0).all()


# ===========================================================================
# End-to-end "smoke test": inference rule sanity check
# ===========================================================================

class TestInferenceRuleSanity:
    """Verify the inference rule behaves as expected on toy data."""

    def test_centroid_clips_pass_as_known(self):
        """Embeddings very close to centroids must fall below any reasonable threshold."""
        centroids = torch.randn(4, EMB_DIM)
        # Embeddings near centroid 0 (tiny noise)
        near_embs = centroids[0].unsqueeze(0) + torch.randn(5, EMB_DIM) * 0.01
        dists = min_centroid_distances(near_embs, centroids)
        # With noise ~0.01 and dim=128, distance ≈ 0.01 * sqrt(128) ≈ 0.11
        assert (dists < 1.0).all(), "Near-centroid embeddings should have small distances"

    def test_random_noise_clips_are_far(self):
        """Embeddings far from centroids should have large distances."""
        centroids = torch.zeros(4, EMB_DIM)   # all at origin
        far_embs = torch.ones(5, EMB_DIM) * 50.0
        dists = min_centroid_distances(far_embs, centroids)
        # Distance = 50 * sqrt(128) ≈ 565
        assert (dists > 100.0).all(), "Far embeddings should have large distances"

    def test_threshold_flags_unknown_correctly(self):
        """Given a threshold, the flagging logic must work correctly."""
        threshold = 10.0
        # Known: distances below threshold
        known_dists = torch.tensor([2.0, 3.5, 5.0, 7.0, 9.9])
        # Unknown: distances above threshold
        unknown_dists = torch.tensor([12.0, 15.0, 20.0])

        known_flagged = (known_dists > threshold).sum().item()
        unknown_flagged = (unknown_dists > threshold).sum().item()

        assert known_flagged == 0, f"{known_flagged} known clips incorrectly flagged as unknown"
        assert unknown_flagged == len(unknown_dists), (
            f"Only {unknown_flagged}/{len(unknown_dists)} unknown clips correctly flagged"
        )

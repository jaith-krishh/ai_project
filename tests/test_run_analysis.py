"""
tests/test_run_analysis.py
==========================
Unit and integration tests for backend/utils.py and backend/run_analysis.py.

Run with:
    python -m pytest tests/test_run_analysis.py -v
"""

from __future__ import annotations

import os
import sys
import types
import numpy as np
import pytest
import torch

# Make sure the project root is on sys.path when running from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils import (
    SAMPLE_RATE,
    N_MELS,
    TARGET_LENGTH,
    DURATION,
    audio_window_to_mel,
    load_audio,
)
from backend.run_analysis import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HOP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    run_analysis,
    _run_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(duration: float = 6.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a simple 440 Hz sine wave as float32."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_dummy_model(num_classes: int = 4):
    """Instantiate an untrained AudioSEDNet on CPU."""
    from backend.model import AudioSEDNet
    model = AudioSEDNet(num_classes=num_classes)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


CLASS_NAMES = ["bird", "construction", "machinery", "traffic"]


# ---------------------------------------------------------------------------
# audio_window_to_mel  (utils.py)
# ---------------------------------------------------------------------------

class TestAudioWindowToMel:
    def test_output_shape_full_window(self):
        """Full-length input produces (1, N_MELS, T) tensor."""
        audio = _make_sine(DURATION)
        spec = audio_window_to_mel(audio)
        assert spec.shape[0] == 1
        assert spec.shape[1] == N_MELS

    def test_output_dtype(self):
        audio = _make_sine(DURATION)
        spec = audio_window_to_mel(audio)
        assert spec.dtype == torch.float32

    def test_short_audio_padded_to_same_shape(self):
        """Audio shorter than TARGET_LENGTH must be padded to same output shape."""
        audio_short = _make_sine(1.5)   # much shorter than 4 s
        audio_full  = _make_sine(DURATION)
        spec_short = audio_window_to_mel(audio_short)
        spec_full  = audio_window_to_mel(audio_full)
        assert spec_short.shape == spec_full.shape, (
            f"Short audio shape {spec_short.shape} != full audio shape {spec_full.shape}"
        )

    def test_long_audio_trimmed_to_same_shape(self):
        """Audio longer than TARGET_LENGTH must be trimmed."""
        audio_long = _make_sine(10.0)
        spec_long = audio_window_to_mel(audio_long)
        spec_ref  = audio_window_to_mel(_make_sine(DURATION))
        assert spec_long.shape == spec_ref.shape

    def test_values_are_normalised(self):
        """Spectrogram values should be z-scored (roughly zero-mean, unit std)."""
        audio = _make_sine(DURATION)
        spec = audio_window_to_mel(audio).numpy()
        # With per-instance z-score the result may not be exactly 0/1 due to
        # float rounding, but should be close-ish.
        assert abs(spec.mean()) < 1.0   # mean near zero
        assert 0.5 < spec.std() < 2.0   # std in reasonable range

    def test_no_channel_dim_explosion(self):
        """Output must have exactly one channel (first dim)."""
        audio = _make_sine(DURATION)
        spec = audio_window_to_mel(audio)
        assert spec.shape[0] == 1


# ---------------------------------------------------------------------------
# load_audio  (utils.py)
# ---------------------------------------------------------------------------

class TestLoadAudio:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_audio(str(tmp_path / "nonexistent.wav"))

    def test_returns_float32(self, tmp_path):
        import soundfile as sf
        wav_path = str(tmp_path / "test.wav")
        data = _make_sine(2.0)
        sf.write(wav_path, data, SAMPLE_RATE)
        audio = load_audio(wav_path)
        assert audio.dtype == np.float32

    def test_returns_1d(self, tmp_path):
        import soundfile as sf
        wav_path = str(tmp_path / "test.wav")
        data = _make_sine(2.0)
        sf.write(wav_path, data, SAMPLE_RATE)
        audio = load_audio(wav_path)
        assert audio.ndim == 1


# ---------------------------------------------------------------------------
# Sliding-window logic (run_analysis.py internals)
# ---------------------------------------------------------------------------

class TestWindowTimestamps:
    """Verify timestamp correctness without real checkpoint."""

    def _mock_run_window(self, *args, start_sec, end_sec, **kwargs):
        return {"start": start_sec, "end": end_sec, "probs": {}, "embedding": []}

    def test_window_count_6s_audio(self):
        """6 s audio with 4 s window / 2 s hop -> 2 windows: [0-4], [2-6]."""
        total = 6.0
        ws    = 4.0
        hs    = 2.0
        sr    = SAMPLE_RATE
        waveform = _make_sine(total)
        total_samples = len(waveform)
        window_samples = int(ws * sr)
        hop_samples    = int(hs * sr)

        starts = []
        ends = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = start_sample + window_samples
            actual_end_sample = min(end_sample, total_samples)
            starts.append(start_sample / sr)
            ends.append(actual_end_sample / sr)
            if actual_end_sample >= total_samples:
                break
            start_sample += hop_samples

        assert len(starts) == 2
        assert starts == [pytest.approx(0.0), pytest.approx(2.0)]
        assert ends == [pytest.approx(4.0), pytest.approx(6.0)]

    def test_first_window_end(self):
        """First window of a long audio has end = window_seconds."""
        ws = 4.0
        total_duration = 10.0
        sr = SAMPLE_RATE
        total_samples = int(total_duration * sr)
        window_samples = int(ws * sr)
        start_sample = 0
        actual_end = min(start_sample + window_samples, total_samples)
        end_sec = actual_end / sr
        assert end_sec == pytest.approx(ws)

    def test_final_partial_window_end_is_audio_end(self):
        """Final short window end == actual audio length / sr."""
        total_duration = 6.0
        ws = 4.0
        hs = 2.0
        sr = SAMPLE_RATE
        total_samples = int(total_duration * sr)
        window_samples = int(ws * sr)
        hop_samples    = int(hs * sr)

        ends = []
        start_sample = 0
        while start_sample < total_samples:
            actual_end = min(start_sample + window_samples, total_samples)
            ends.append(actual_end / sr)
            start_sample += hop_samples

        # Last window ends at 6.0 s (real audio end), not 8.0 s (padded end)
        assert ends[-1] == pytest.approx(total_duration)


# ---------------------------------------------------------------------------
# _run_window output contract
# ---------------------------------------------------------------------------

class TestRunWindowContract:
    def test_output_keys(self):
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        assert set(result.keys()) == {"start", "end", "probs", "embedding"}

    def test_probs_contains_all_classes(self):
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        assert set(result["probs"].keys()) == set(CLASS_NAMES)

    def test_probs_in_range_0_1(self):
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        for cls, prob in result["probs"].items():
            assert 0.0 <= prob <= 1.0, f"Prob for {cls!r} out of [0,1]: {prob}"

    def test_embedding_is_list_of_floats(self):
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        emb = result["embedding"]
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_embedding_not_cuda_tensor(self):
        """Embedding must be a plain Python list (serialisable)."""
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        assert not isinstance(result["embedding"], torch.Tensor)

    def test_no_gradients_computed(self):
        """Inference must not accumulate gradients."""
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=0.0,
            end_sec=4.0,
        )
        for p in model.parameters():
            assert p.grad is None, "Gradients should not be computed during inference"

    def test_start_end_types(self):
        model = _make_dummy_model(len(CLASS_NAMES))
        device = torch.device("cpu")
        audio = _make_sine(DURATION)
        result = _run_window(
            model=model,
            class_names=CLASS_NAMES,
            audio_segment=audio,
            device=device,
            start_sec=2.0,
            end_sec=6.0,
        )
        assert isinstance(result["start"], float)
        assert isinstance(result["end"], float)
        assert result["start"] == pytest.approx(2.0)
        assert result["end"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# run_analysis  (end-to-end with real checkpoint if present)
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Integration tests.  Skipped if checkpoint or test audio are absent."""

    CHECKPOINT = DEFAULT_CHECKPOINT

    @pytest.fixture
    def sine_wav(self, tmp_path):
        import soundfile as sf
        wav_path = str(tmp_path / "test_audio.wav")
        data = _make_sine(6.0)   # 6 s audio -> 3 windows at 4s/2s
        sf.write(wav_path, data, SAMPLE_RATE)
        return wav_path

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_multiple_overlapping_windows(self, sine_wav):
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        assert len(results) >= 2, "Expected multiple windows for 6 s audio"

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_windows_chronological(self, sine_wav):
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        starts = [w["start"] for w in results]
        assert starts == sorted(starts), "Windows must be chronologically ordered"

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_output_schema(self, sine_wav):
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        for w in results:
            assert "start" in w
            assert "end" in w
            assert "probs" in w
            assert "embedding" in w
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)
            assert isinstance(w["probs"], dict)
            assert isinstance(w["embedding"], list)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_probs_all_classes_present(self, sine_wav):
        import torch
        from backend.utils import load_model
        _, class_names = load_model(self.CHECKPOINT)
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        for w in results:
            assert set(w["probs"].keys()) == set(class_names)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_probs_range(self, sine_wav):
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        for w in results:
            for cls, p in w["probs"].items():
                assert 0.0 <= p <= 1.0, f"Prob for {cls!r} = {p} out of [0,1]"

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_embedding_serialisable(self, sine_wav):
        import json
        results = run_analysis(sine_wav, checkpoint_path=self.CHECKPOINT)
        for w in results:
            # Must not raise
            json.dumps(w)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_short_audio_handled(self, tmp_path):
        """Audio shorter than one window must produce exactly one window."""
        import soundfile as sf
        wav_path = str(tmp_path / "short.wav")
        data = _make_sine(1.5)   # shorter than 4 s window
        sf.write(wav_path, data, SAMPLE_RATE)
        results = run_analysis(wav_path, checkpoint_path=self.CHECKPOINT)
        assert len(results) == 1
        assert results[0]["start"] == pytest.approx(0.0)
        assert results[0]["end"]   == pytest.approx(1.5, abs=0.01)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_4s_window_2s_hop_timestamps(self, tmp_path):
        """Exact timestamp check: 10 s audio, 4 s window, 2 s hop -> [0, 2, 4, 6]."""
        import soundfile as sf
        wav_path = str(tmp_path / "ten_sec.wav")
        data = _make_sine(10.0)
        sf.write(wav_path, data, SAMPLE_RATE)
        results = run_analysis(
            wav_path,
            checkpoint_path=self.CHECKPOINT,
            window_seconds=4.0,
            hop_seconds=2.0,
        )
        expected_starts = [0.0, 2.0, 4.0, 6.0]
        actual_starts = [w["start"] for w in results]
        assert actual_starts == pytest.approx(expected_starts, abs=0.01)
        expected_ends = [4.0, 6.0, 8.0, 10.0]
        actual_ends = [w["end"] for w in results]
        assert actual_ends == pytest.approx(expected_ends, abs=0.01)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_partial_window_7s_audio(self, tmp_path):
        """7 s audio, 4 s window, 2 s hop -> 3 windows with final partial window ending at 7.0s."""
        import soundfile as sf
        wav_path = str(tmp_path / "seven_sec.wav")
        data = _make_sine(7.0)
        sf.write(wav_path, data, SAMPLE_RATE)
        results = run_analysis(
            wav_path,
            checkpoint_path=self.CHECKPOINT,
            window_seconds=4.0,
            hop_seconds=2.0,
        )
        assert len(results) == 3
        assert [w["start"] for w in results] == pytest.approx([0.0, 2.0, 4.0], abs=0.01)
        assert [w["end"] for w in results] == pytest.approx([4.0, 6.0, 7.0], abs=0.01)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_CHECKPOINT),
        reason="No trained checkpoint found at outputs/checkpoints/best_model.pt",
    )
    def test_cpu_inference_works(self, sine_wav):
        results = run_analysis(
            sine_wav,
            checkpoint_path=self.CHECKPOINT,
            device=torch.device("cpu"),
        )
        assert len(results) > 0

    def test_missing_audio_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_analysis(str(tmp_path / "missing.wav"))

    def test_missing_checkpoint_raises(self, tmp_path):
        import soundfile as sf
        wav_path = str(tmp_path / "audio.wav")
        sf.write(wav_path, _make_sine(2.0), SAMPLE_RATE)
        with pytest.raises(FileNotFoundError):
            run_analysis(wav_path, checkpoint_path=str(tmp_path / "no_ckpt.pt"))

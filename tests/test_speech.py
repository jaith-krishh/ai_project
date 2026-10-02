"""Tests for backend/speech.py (speech detection alongside the sound model)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.speech import SPEECH_LABEL, suppress_voice_like


def _pred(label, start, end):
    return {"label": label, "start": start, "end": end, "confidence": 0.5, "embedding": [0.0]}


def test_voice_like_and_unknown_dropped_under_speech():
    speech = [{"label": SPEECH_LABEL, "start": 0.0, "end": 4.0, "confidence": 0.9, "similar_to": None}]
    labelled = [
        _pred("us8k_children_playing", 0.0, 2.0),   # voice-like, fully under speech -> dropped
        _pred("Unknown", 2.0, 4.0),                 # unknown, fully under speech -> dropped
        _pred("esc_rain", 2.0, 4.0),                # not voice-like -> kept
        _pred("esc_laughing", 6.0, 8.0),            # voice-like but no speech there -> kept
        _pred("Unknown", 3.5, 7.5),                 # only 12.5% covered by speech -> kept
    ]
    kept = suppress_voice_like(labelled, speech)
    assert [(p["label"], p["start"]) for p in kept] == [("esc_rain", 2.0), ("esc_laughing", 6.0), ("Unknown", 3.5)]


def test_no_speech_keeps_everything():
    labelled = [_pred("us8k_children_playing", 0.0, 2.0)]
    assert suppress_voice_like(labelled, []) == labelled


def test_vad_finds_no_speech_in_noise(tmp_path):
    pytest.importorskip("silero_vad")
    import soundfile as sf
    from backend.speech import detect_speech

    path = str(tmp_path / "noise.wav")
    sf.write(path, (0.05 * np.random.default_rng(0).standard_normal(16000 * 3)).astype(np.float32), 16000)
    assert detect_speech(path) == []

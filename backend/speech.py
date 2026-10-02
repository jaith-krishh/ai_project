"""
backend/speech.py
=================
Speech detection alongside the 60-class sound model.

ESC-50 and UrbanSound8K have no "speech" class, so talking used to be labelled
with the nearest human sound the model knows (children playing, laughing, ...)
or flagged Unknown. This module runs Silero VAD (a small pretrained
voice-activity detector, bundled in the ``silero-vad`` pip package, no download
at runtime) to find speech segments, then:

* adds "Speech" events to the report, and
* drops voice-like guesses (children playing, laughing, crying baby) and
  Unknown predictions from windows that are mostly speech, since the speech
  explains them.

If ``silero-vad`` is not installed, speech detection is skipped with a warning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

SPEECH_LABEL = "Speech"
VAD_SAMPLE_RATE = 16000

# Sound names (see unknown_scoring.sound_name) the model uses for human voices
# when it hears talking. Dropped where speech is detected.
VOICE_LIKE = {"children_playing", "laughing", "crying_baby"}

_vad_model = None


def _load_vad():
    global _vad_model
    if _vad_model is None:
        import warnings
        from silero_vad import load_silero_vad
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation notice
            _vad_model = load_silero_vad()
    return _vad_model


def detect_speech(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_s: float = 0.3,
    merge_gap_s: float = 1.0,
) -> List[Dict[str, Any]]:
    """Return speech events [{"label": "Speech", "start", "end", "confidence", "similar_to": None}].

    Segments closer than merge_gap_s are joined so a sentence with short pauses is
    one event. confidence is the mean speech probability inside the segment.
    """
    import librosa
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_vad()
    audio, _ = librosa.load(audio_path, sr=VAD_SAMPLE_RATE, mono=True)
    if len(audio) == 0:
        return []
    wav = torch.from_numpy(audio.astype(np.float32))

    segments = get_speech_timestamps(
        wav, model, threshold=threshold, sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=int(min_speech_s * 1000), return_seconds=False,
    )
    if not segments:
        return []

    # Per-chunk speech probabilities for confidence values
    chunk = 512
    model.reset_states()
    with torch.no_grad():
        probs = np.array([
            float(model(torch.nn.functional.pad(wav[i:i + chunk], (0, max(0, chunk - len(wav[i:i + chunk])))),
                        VAD_SAMPLE_RATE))
            for i in range(0, len(wav), chunk)
        ])
    model.reset_states()

    merged: List[Tuple[int, int]] = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        if merged and s - merged[-1][1] <= merge_gap_s * VAD_SAMPLE_RATE:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    events = []
    for s, e in merged:
        p = probs[s // chunk: max(s // chunk + 1, e // chunk)]
        events.append({
            "label": SPEECH_LABEL,
            "start": round(s / VAD_SAMPLE_RATE, 3),
            "end": round(e / VAD_SAMPLE_RATE, 3),
            "confidence": round(float(p.mean()) if len(p) else threshold, 4),
            "similar_to": None,
        })
    return events


def _overlap(a_start: float, a_end: float, segs: List[Dict[str, Any]]) -> float:
    return sum(max(0.0, min(a_end, s["end"]) - max(a_start, s["start"])) for s in segs)


def suppress_voice_like(
    labelled: List[Dict[str, Any]],
    speech_events: List[Dict[str, Any]],
    min_fraction: float = 0.5,
) -> List[Dict[str, Any]]:
    """Drop voice-like and Unknown window predictions that are mostly covered by speech.

    labelled: P3 window predictions (from label_windows), each with start/end/label.
    """
    from backend.unknown_scoring import sound_name

    if not speech_events:
        return labelled
    kept = []
    for pred in labelled:
        span = pred["end"] - pred["start"]
        mostly_speech = span > 0 and _overlap(pred["start"], pred["end"], speech_events) / span >= min_fraction
        voice_like = pred["label"].lower() == "unknown" or sound_name(pred["label"]) in VOICE_LIKE
        if not (mostly_speech and voice_like):
            kept.append(pred)
    return kept

"""
backend/run_analysis.py
=======================
Sliding-window inference core for the Acoustic Sound Analyzer.

P3 Integration Contract
-----------------------
P3 consumes the output of ``run_analysis()`` and can assume::

    results = run_analysis(audio_path)
    for window in results:
        start     = window["start"]      # float  -- window start in seconds
        end       = window["end"]        # float  -- window end   in seconds
        probs     = window["probs"]      # dict[str, float]  class -> sigmoid prob
        embedding = window["embedding"]  # list[float]  128-dim feature vector

P3 does NOT need to know about waveform preprocessing, librosa internals,
PyTorch device handling, mel-spectrogram construction, or tensor dimensions.

Window output contract
----------------------
Each window dict contains exactly::

    {
        "start":     float,          # seconds from beginning of audio
        "end":       float,          # seconds (= start + window_seconds for full
                                     #           windows; actual audio end for the
                                     #           final short window)
        "probs":     dict[str, float],  # ALL known classes, sigmoid probabilities
        "embedding": list[float],       # 128-dim CPU-side serialisable vector
    }

Padding note
------------
The final window may be shorter than ``window_seconds``.  The raw waveform is
zero-padded to ``TARGET_LENGTH`` before mel-spectrogram computation so the
model always receives a fixed-size input.  ``end`` reports the *actual* audio
time (not the padded end), so callers know the true temporal coverage.

Default sliding-window settings
--------------------------------
window_seconds = 4.0 s  (matches model training duration)
hop_seconds    = 2.0 s  (50 % overlap)

At sr=22 050:
  window = 4 * 22050 = 88 200 samples
  hop    = 2 * 22050 = 44 100 samples
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend.utils import (
    SAMPLE_RATE,
    TARGET_LENGTH,
    audio_window_to_mel,
    load_audio,
    load_model,
)

# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = os.path.join("outputs", "checkpoints", "best_model.pt")

# Default sliding-window parameters
DEFAULT_WINDOW_SECONDS: float = 4.0
DEFAULT_HOP_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_window(
    model: torch.nn.Module,
    class_names: List[str],
    audio_segment: np.ndarray,
    device: torch.device,
    start_sec: float,
    end_sec: float,
) -> Dict[str, Any]:
    """Run inference on a single waveform window and return the window dict.

    Parameters
    ----------
    model:
        ``AudioSEDNet`` in eval mode with gradients disabled.
    class_names:
        Ordered list of class names matching the model output dimension.
    audio_segment:
        1-D float32 waveform segment (may be shorter than TARGET_LENGTH;
        will be zero-padded inside ``audio_window_to_mel``).
    device:
        Device the model lives on.
    start_sec, end_sec:
        Temporal position of this window in the full audio (seconds).

    Returns
    -------
    dict with keys ``start``, ``end``, ``probs``, ``embedding``.
    """
    # --- Preprocessing: waveform -> (1, 1, n_mels, time_steps) tensor -------
    spec = audio_window_to_mel(audio_segment)   # (1, n_mels, T)
    spec = spec.unsqueeze(0).to(device)         # (1, 1, n_mels, T)

    with torch.no_grad():
        # Extract embedding (128-dim, after fc_embedding + ReLU)
        emb = model.extract_embeddings(spec)    # (1, 128)

        # Compute logits (reuse emb to avoid running CNN backbone twice; in eval mode dropout is a no-op)
        if hasattr(model, "classifier") and hasattr(model, "dropout"):
            logits = model.classifier(model.dropout(emb))
        else:
            logits = model(spec)                # (1, num_classes)

    # Sigmoid probabilities (model returns raw logits)
    probs_tensor = torch.sigmoid(logits)        # (1, num_classes)

    # Move to CPU and convert
    probs_np = probs_tensor.squeeze(0).cpu().numpy()   # (num_classes,)
    emb_np = emb.squeeze(0).cpu().numpy()              # (128,)

    if len(probs_np) != len(class_names):
        raise RuntimeError(
            f"Model output dim ({len(probs_np)}) != number of class labels "
            f"({len(class_names)}).  Checkpoint and class_names are mismatched."
        )

    probs_dict: Dict[str, float] = {
        name: float(prob) for name, prob in zip(class_names, probs_np)
    }

    return {
        "start": float(start_sec),
        "end": float(end_sec),
        "probs": probs_dict,
        "embedding": emb_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    audio_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Analyse an audio file with an overlapping sliding window and return
    per-window inference results.

    Parameters
    ----------
    audio_path:
        Path to the input audio file (WAV, FLAC, OGG, MP3, …).
    checkpoint_path:
        Path to the trained ``AudioSEDNet`` checkpoint (``.pt`` file).
        Defaults to ``outputs/checkpoints/best_model.pt``.
    window_seconds:
        Duration of each analysis window in seconds.  Default 4.0 s
        (matches training duration).
    hop_seconds:
        Hop (step) size between consecutive windows in seconds.  Default 2.0 s
        (50 % overlap).
    device:
        PyTorch device.  If *None*, CUDA is used when available, otherwise CPU.

    Returns
    -------
    list[dict]
        Chronologically ordered list of window result dicts.  Each dict::

            {
                "start":     float,            # window start (seconds)
                "end":       float,            # window end   (seconds, actual audio time)
                "probs":     dict[str, float], # ALL classes, sigmoid [0, 1]
                "embedding": list[float],      # 128-dim feature vector
            }

    Raises
    ------
    FileNotFoundError
        If *audio_path* or *checkpoint_path* does not exist.
    RuntimeError
        If the audio cannot be decoded, the checkpoint cannot be loaded,
        or model output dimensions do not match class labels.
    ValueError
        If *window_seconds* or *hop_seconds* are not positive numbers.
    """
    # --- Validate parameters ------------------------------------------------
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be > 0, got {hop_seconds}")

    # --- Device selection ---------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model (once) --------------------------------------------------
    model, class_names = load_model(checkpoint_path, device=device)

    # --- Load audio (once) --------------------------------------------------
    waveform = load_audio(audio_path, sr=SAMPLE_RATE)
    total_samples = len(waveform)
    total_duration = total_samples / SAMPLE_RATE

    # --- Compute window / hop sizes in samples ------------------------------
    window_samples = int(window_seconds * SAMPLE_RATE)
    hop_samples = int(hop_seconds * SAMPLE_RATE)

    # --- Sliding window -----------------------------------------------------
    results: List[Dict[str, Any]] = []

    # Handle audio shorter than one window
    if total_samples == 0:
        raise RuntimeError(f"Audio file {audio_path!r} produced an empty waveform.")

    start_sample = 0
    while start_sample < total_samples:
        end_sample = start_sample + window_samples

        # Actual audio end time (may be before the padded window end)
        actual_end_sample = min(end_sample, total_samples)
        start_sec = start_sample / SAMPLE_RATE
        end_sec = actual_end_sample / SAMPLE_RATE  # reports real audio coverage

        segment = waveform[start_sample:actual_end_sample]

        window_result = _run_window(
            model=model,
            class_names=class_names,
            audio_segment=segment,
            device=device,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        results.append(window_result)

        # If this window reached or passed the end of the audio, all audio has
        # been processed; do not produce further redundant padded windows.
        if actual_end_sample >= total_samples:
            break

        start_sample += hop_samples

    # Results are already chronological (start_sample increments monotonically)
    return results

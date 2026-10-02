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
        probs     = window["probs"]      # dict[str, float]  class -> softmax prob
        embedding = window["embedding"]  # list[float]  128-dim feature vector
        rms_db    = window["rms_db"]     # float  loudness in dBFS (silence ~ -60 or lower)

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
        "probs":     dict[str, float],  # ALL known classes, softmax probabilities (sum to 1)
        "embedding": list[float],       # 128-dim CPU-side serialisable vector
        "rms_db":    float,             # RMS loudness of the real (unpadded) audio, dBFS
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
    dict with keys ``start``, ``end``, ``probs``, ``embedding``, ``rms_db``.
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

    # Softmax probabilities: the model was trained with CrossEntropyLoss
    # (single-label), so softmax is the calibrated output, not sigmoid.
    probs_tensor = F.softmax(logits, dim=1)     # (1, num_classes)

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

    rms = float(np.sqrt(np.mean(np.square(audio_segment, dtype=np.float64)))) if len(audio_segment) else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-10))

    return {
        "start": float(start_sec),
        "end": float(end_sec),
        "probs": probs_dict,
        "embedding": emb_np.tolist(),
        "rms_db": float(rms_db),
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
                "probs":     dict[str, float], # ALL classes, softmax [0, 1]
                "embedding": list[float],      # 128-dim feature vector
                "rms_db":    float,            # loudness, dBFS
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


# ---------------------------------------------------------------------------
# Full pipeline (P1 + P2 + P3 + P4)
# ---------------------------------------------------------------------------

DEFAULT_CENTROIDS = os.path.join("outputs", "centroids.pt")
DEFAULT_THRESHOLD = os.path.join("outputs", "threshold.pt")


def analyze_file(
    audio_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    centroids_path: str = DEFAULT_CENTROIDS,
    threshold_path: str = DEFAULT_THRESHOLD,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    prob_threshold: float = 0.3,
    top_k: int = 3,
    min_confidence: float = 0.3,
    silence_db: float = -50.0,
    similarity_temperature: float = 0.5,
) -> Dict[str, Any]:
    """Run the complete analysis on one audio file.

    Steps: sliding-window inference (P2) -> per-window labels with Unknown
    detection against P1's centroids/threshold -> event merging and similarity
    lists (P3) -> location classification and text report (P4).

    If ``centroids.pt`` or ``threshold.pt`` is missing, a warning is printed and
    the analysis runs without Unknown detection.

    Returns
    -------
    dict with keys ``events`` (P3 event list), ``aggregation`` (P4 result),
    ``duration`` (seconds) and ``report`` (formatted text).
    """
    from backend.aggregate_report import aggregate_events, format_report
    from backend.p3_event_merging import (
        label_windows,
        load_centroids,
        load_class_radii,
        load_unknown_config,
        postprocess_predictions,
    )

    centroids: Dict[str, Any] = {}
    radii = None
    unknown_cfg: Dict[str, Any] = {"method": "distance", "threshold": None}
    if os.path.exists(centroids_path) and os.path.exists(threshold_path):
        centroids = load_centroids(centroids_path)
        radii = load_class_radii(centroids_path)
        unknown_cfg = load_unknown_config(threshold_path)
    else:
        print(
            f"[run_analysis] WARNING: {centroids_path!r} or {threshold_path!r} not found; "
            "Unknown-sound detection is disabled. Run backend.compute_centroids and "
            "backend.calibrate_unknown first (see README.md)."
        )

    windows = run_analysis(
        audio_path,
        checkpoint_path=checkpoint_path,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    duration = max(w["end"] for w in windows)

    labelled = label_windows(
        windows,
        centroids,
        unknown_cfg["threshold"],
        prob_threshold=prob_threshold,
        method=unknown_cfg["method"],
        radii=radii,
        min_confidence=min_confidence,
        silence_db=silence_db,
    )
    events = postprocess_predictions(
        labelled, centroids=centroids, top_k=top_k, similarity_temperature=similarity_temperature
    )
    aggregation = aggregate_events(events, total_duration=duration)
    report = format_report(events, aggregation)

    return {
        "events": events,
        "aggregation": aggregation,
        "duration": duration,
        "report": report,
    }


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Acoustic Sound Analyzer: detect sounds, flag unknowns and classify the location of an audio file."
    )
    parser.add_argument("--input", required=True, help="Path to the audio file to analyse.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--centroids", default=DEFAULT_CENTROIDS, help=f"Class centroids from compute_centroids (default: {DEFAULT_CENTROIDS})")
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD, help=f"Unknown threshold from calibrate_unknown (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SECONDS, help="Window length in seconds (default: 4.0)")
    parser.add_argument("--hop", type=float, default=DEFAULT_HOP_SECONDS, help="Hop between windows in seconds (default: 2.0)")
    parser.add_argument("--prob-threshold", type=float, default=0.3,
                        help="Extra classes in a window are reported when their probability is at least this (default: 0.3)")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Windows whose most likely class is below this are treated as background (default: 0.3)")
    parser.add_argument("--silence-db", type=float, default=-50.0,
                        help="Windows quieter than this (dBFS) are treated as silence (default: -50)")
    parser.add_argument("--top-k", type=int, default=3, help="Closest known classes listed for Unknown sounds (default: 3)")
    parser.add_argument("--similarity-temperature", type=float, default=0.5,
                        help="Sharpness of Unknown 'closest matches' percentages; smaller = sharper (default: 0.5)")
    parser.add_argument("--json", metavar="PATH", help="Also save events and aggregation as JSON to PATH.")
    args = parser.parse_args(argv)

    result = analyze_file(
        args.input,
        checkpoint_path=args.checkpoint,
        centroids_path=args.centroids,
        threshold_path=args.threshold,
        window_seconds=args.window,
        hop_seconds=args.hop,
        prob_threshold=args.prob_threshold,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        silence_db=args.silence_db,
        similarity_temperature=args.similarity_temperature,
    )
    print(result["report"])

    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "report"}, f, indent=2)
        print(f"\n[run_analysis] JSON saved -> {args.json}")


if __name__ == "__main__":
    main()

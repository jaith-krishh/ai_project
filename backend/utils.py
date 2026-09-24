"""
backend/utils.py
================
Mel-spectrogram preprocessing helpers for the Acoustic Sound Analyzer.

Training preprocessing configuration (discovered from backend/dataset.py):
  - sample rate    : 22 050 Hz
  - channels       : mono
  - target duration: 4.0 s  ->  target_length = 88 200 samples
  - padding        : zero-pad (end) when shorter than target_length
  - trimming       : hard-trim when longer than target_length
  - n_mels         : 64
  - n_fft          : 1 024
  - hop_length     : 512
  - fmin / fmax    : librosa defaults (0 Hz / sr/2)
  - power spectrum : librosa.feature.melspectrogram  (power=2.0 default)
  - log conversion : librosa.power_to_db(S, ref=np.max)
  - normalisation  : per-instance z-score  (log_S - mean) / (std + 1e-6)
  - tensor shape   : (1, n_mels, time_steps)  -- one channel, float32
  - dtype          : torch.float32
  - augmentation   : NONE at inference (train-only: time-shift, noise, SpecAugment)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import librosa
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Training preprocessing constants -- MUST match backend/dataset.py exactly
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22_050          # Hz used during training
N_MELS: int = 64                   # Mel frequency bins
N_FFT: int = 1_024                 # FFT window size
HOP_LENGTH: int = 512              # STFT hop length (samples per step)
DURATION: float = 4.0              # Expected clip duration in seconds
TARGET_LENGTH: int = int(SAMPLE_RATE * DURATION)  # 88 200 samples


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to *sr*.

    Parameters
    ----------
    audio_path:
        Path to the audio file.  Must exist and be decodable by librosa /
        soundfile (WAV, FLAC, OGG, MP3, etc.).
    sr:
        Target sample rate.  Defaults to the training sample rate (22 050 Hz).

    Returns
    -------
    np.ndarray
        1-D float32 waveform array at *sr* Hz.

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist on disk.
    RuntimeError
        If the file cannot be decoded.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path!r}")

    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode audio file {audio_path!r}: {exc}"
        ) from exc

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Mel-spectrogram preprocessing (MUST mirror dataset.py exactly)
# ---------------------------------------------------------------------------

def audio_window_to_mel(
    audio_segment: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_length: int = TARGET_LENGTH,
) -> torch.Tensor:
    """Convert a raw waveform window into a normalised log-Mel spectrogram tensor.

    Reproduces the *inference* path of ``AudioDataset.__getitem__``
    in ``backend/dataset.py`` without any training-only augmentations
    (time-shift, Gaussian noise, and SpecAugment are intentionally omitted).

    Processing steps
    ----------------
    1. Zero-pad (at end) if shorter than *target_length*; hard-trim if longer.
    2. Compute power mel-spectrogram with librosa (n_fft=1024, hop_length=512).
    3. Convert to dB: ``librosa.power_to_db(S, ref=np.max)``.
    4. Per-instance z-score: ``(log_S - mean) / (std + 1e-6)``.
    5. Wrap as ``torch.float32`` tensor of shape ``(1, n_mels, time_steps)``.

    Parameters
    ----------
    audio_segment:
        1-D float32 waveform at *sr* Hz.  May be shorter than *target_length*
        (will be zero-padded to produce a fixed-size spectrogram).
    sr, n_mels, n_fft, hop_length:
        Must match the values used during training (defaults do so).
    target_length:
        Expected number of waveform samples.  Short segments are zero-padded;
        long segments are trimmed.

    Returns
    -------
    torch.Tensor
        Shape ``(1, n_mels, time_steps)``, dtype ``torch.float32``.
    """
    # Step 1 -- Pad or trim to fixed waveform length (matches dataset.py) ----
    if len(audio_segment) < target_length:
        pad_len = target_length - len(audio_segment)
        audio_segment = np.pad(audio_segment, (0, pad_len), mode="constant")
    else:
        audio_segment = audio_segment[:target_length]

    # Step 2 -- Power mel-spectrogram (matches dataset.py) -------------------
    S = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Step 3 -- Log conversion (matches dataset.py) --------------------------
    log_S = librosa.power_to_db(S, ref=np.max)

    # Step 4 -- Per-instance z-score normalisation (matches dataset.py) ------
    mean = np.mean(log_S)
    std = np.std(log_S) + 1e-6
    norm_S = (log_S - mean) / std

    # Step 5 -- To tensor (1, n_mels, time_steps) (matches dataset.py) -------
    spec_tensor = torch.tensor(norm_S, dtype=torch.float32).unsqueeze(0)

    return spec_tensor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load a trained ``AudioSEDNet`` checkpoint and return ``(model, class_names)``.

    The checkpoint is expected to contain the keys written by ``train.py``::

        {
            "model_state_dict": ...,
            "class_names":      [...],   # list[str]
            ...
        }

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` checkpoint file produced by ``train.py``.
    device:
        Target device.  If *None*, the best available device is chosen
        automatically (CUDA > CPU).

    Returns
    -------
    model : AudioSEDNet
        Model in ``eval()`` mode with gradients disabled, on *device*.
    class_names : list[str]
        Ordered list of class label strings (length == model output dim).

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    KeyError
        If required keys are absent from the checkpoint.
    RuntimeError
        If the state-dict cannot be loaded (e.g. architecture mismatch).
    """
    from backend.model import AudioSEDNet  # local import avoids circular deps

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load checkpoint {checkpoint_path!r}: {exc}"
        ) from exc

    if "class_names" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} is missing the 'class_names' key."
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} is missing the 'model_state_dict' key."
        )

    class_names: List[str] = checkpoint["class_names"]
    num_classes = len(class_names)

    model = AudioSEDNet(num_classes=num_classes)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"State-dict mismatch when loading {checkpoint_path!r}: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    # Disable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    return model, class_names

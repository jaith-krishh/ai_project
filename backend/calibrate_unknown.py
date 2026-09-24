"""
backend/calibrate_unknown.py
=============================
P1 Task – Step 2: Unknown-Threshold Calibration

Uses pre-computed class centroids (outputs/centroids.pt) to find a
distance threshold that separates:
  • Known-class validation clips  → small distance to nearest centroid
  • Unknown-class clips           → large distance to every centroid

Strategy
--------
1. Load centroids.pt produced by compute_centroids.py.
2. Compute the minimum Euclidean distance from each validation clip's
   embedding to all known-class centroids  → "known" distance set.
3. Load the held-out "unknown" class (default: synthesised white-noise
   clips *or* real audio files supplied via --unknown-dir).  Compute
   the same minimum-centroid distance → "unknown" distance set.
4. Sweep candidate thresholds and pick the one that maximises the
   F1-score for the binary task "is this unknown?", using equal weight
   for precision and recall.
5. Save threshold.pt with the scalar threshold and diagnostic metadata.

Output
------
outputs/threshold.pt — dict with:
    "threshold"      : float   — the calibrated distance cutoff
    "metric"         : str     — optimisation target ("f1")
    "best_f1"        : float   — F1 at the chosen threshold
    "known_dists"    : Tensor  — min-centroid distances for known clips
    "unknown_dists"  : Tensor  — min-centroid distances for unknown clips

Usage (from project root)
-------------------------
    # 1. First generate centroids (if not already done):
    python -m backend.compute_centroids

    # 2. Calibrate using synthesised unknown clips (no audio data needed):
    python -m backend.calibrate_unknown

    # 3. Calibrate using a directory of real unknown-class audio files:
    python -m backend.calibrate_unknown --unknown-dir path/to/unknown_audio

    # 4. Calibrate using a specific held-out class from the known dataset:
    python -m backend.calibrate_unknown --holdout-class esc_siren \\
        --data-dir data --holdout-as-unknown

Inference usage
---------------
At inference time, flag a sound as "Unknown" when:
    min_dist_to_any_centroid  >  threshold
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from backend.utils import audio_window_to_mel, load_model, SAMPLE_RATE, TARGET_LENGTH

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_CENTROIDS = os.path.join("outputs", "centroids.pt")
DEFAULT_CHECKPOINT = os.path.join("outputs", "checkpoints", "best_model.pt")
DEFAULT_OUTPUT = os.path.join("outputs", "threshold.pt")
DEFAULT_DATA_DIR = "data"

# Number of synthetic clips to generate when no real unknown audio is available
DEFAULT_SYNTH_CLIPS = 200
# Fraction of the known dataset to use as validation set for known distances
DEFAULT_VAL_FRACTION = 0.2
# Random seed for reproducibility
SEED = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the unknown-sound detection threshold."
    )
    parser.add_argument(
        "--centroids",
        type=str,
        default=DEFAULT_CENTROIDS,
        help=f"Path to centroids.pt (default: {DEFAULT_CENTROIDS})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to best_model.pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--unknown-dir",
        type=str,
        default=None,
        help="Optional directory of real unknown-class audio files. "
             "If omitted, synthetic white-noise clips are used.",
    )
    parser.add_argument(
        "--holdout-class",
        type=str,
        default=None,
        help="If set, treat this class from the training dataset as "
             "the unknown class (requires --holdout-as-unknown flag).",
    )
    parser.add_argument(
        "--holdout-as-unknown",
        action="store_true",
        help="Enable held-out-class mode. Requires --holdout-class.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"Fraction of known clips to use as validation (default: {DEFAULT_VAL_FRACTION})",
    )
    parser.add_argument(
        "--synth-clips",
        type=int,
        default=DEFAULT_SYNTH_CLIPS,
        help=f"Number of synthetic unknown clips to generate when no "
             f"real unknown audio is provided (default: {DEFAULT_SYNTH_CLIPS})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for threshold.pt (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def min_centroid_distances(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum Euclidean distance from each embedding to any centroid.

    Parameters
    ----------
    embeddings : Tensor shape (N, D)
    centroids  : Tensor shape (C, D)

    Returns
    -------
    Tensor shape (N,) — min distance to the nearest centroid for each embedding.
    """
    # Efficient pairwise Euclidean distance via cdist
    dists = torch.cdist(embeddings.float(), centroids.float(), p=2)  # (N, C)
    min_dists, _ = dists.min(dim=1)                                  # (N,)
    return min_dists


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------

def _embed_audio_array(
    audio: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Convert a 1-D waveform to a 128-dim embedding tensor (CPU, shape (128,))."""
    spec = audio_window_to_mel(audio)          # (1, n_mels, T)
    spec = spec.unsqueeze(0).to(device)        # (1, 1, n_mels, T)
    with torch.no_grad():
        emb = model.extract_embeddings(spec)   # (1, 128)
    return emb.squeeze(0).cpu()                # (128,)


def _embed_file_list(
    file_paths: List[str],
    model: torch.nn.Module,
    device: torch.device,
    desc: str = "Embedding",
) -> torch.Tensor:
    """Embed a list of audio files and return a (N, 128) tensor."""
    import librosa

    embs: List[torch.Tensor] = []
    for fp in tqdm(file_paths, desc=desc):
        try:
            audio, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True, duration=5.0)
        except Exception as exc:
            print(f"  [skip] {fp}: {exc}")
            continue
        emb = _embed_audio_array(audio.astype(np.float32), model, device)
        embs.append(emb)

    if not embs:
        raise RuntimeError(f"No audio files could be embedded from the list provided.")

    return torch.stack(embs, dim=0)  # (N, 128)


def _embed_dataset_class(
    data_dir: str,
    target_class: str,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Extract embeddings for all clips of *target_class* from the dataset."""
    import librosa
    from backend.dataset import AudioDataset

    ds = AudioDataset(data_dir=data_dir, is_train=False)

    if target_class not in ds.class_to_idx:
        raise ValueError(
            f"Class '{target_class}' not found in dataset. "
            f"Available classes: {ds.class_names}"
        )

    # Collect file paths for the target class
    target_paths = [fp for fp, lbl in ds.samples if lbl == target_class]
    if not target_paths:
        raise RuntimeError(f"No audio files found for class '{target_class}'.")

    print(f"[calibrate_unknown] Found {len(target_paths)} clips for '{target_class}'")
    return _embed_file_list(target_paths, model, device, desc=f"Embedding {target_class}")


def _embed_directory(
    audio_dir: str,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Embed all audio files in *audio_dir*."""
    AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff", ".au"}
    file_paths = [
        os.path.join(audio_dir, fn)
        for fn in os.listdir(audio_dir)
        if os.path.splitext(fn)[1].lower() in AUDIO_EXTS
    ]
    if not file_paths:
        raise RuntimeError(f"No audio files found in unknown-dir: {audio_dir!r}")

    print(f"[calibrate_unknown] Found {len(file_paths)} unknown audio files in {audio_dir}")
    return _embed_file_list(file_paths, model, device, desc="Embedding unknown files")


def _embed_synthetic_unknown(
    n_clips: int,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Generate *n_clips* synthetic white-noise clips and extract their embeddings.

    White noise sits uniformly far from any learned audio class and is a
    reliable worst-case stand-in when no real unknown data is available.
    We also mix in pink noise and silence for diversity.
    """
    rng = np.random.default_rng(SEED)
    embs: List[torch.Tensor] = []

    for i in tqdm(range(n_clips), desc="Embedding synthetic unknowns"):
        # Cycle through: white noise, pink-ish noise, near-silence
        variant = i % 3
        if variant == 0:
            # White noise
            audio = rng.standard_normal(TARGET_LENGTH).astype(np.float32) * 0.3
        elif variant == 1:
            # Pink-ish noise (sum of filtered bands via cumsum trick)
            white = rng.standard_normal(TARGET_LENGTH).astype(np.float32)
            audio = (np.cumsum(white) * 0.0005).astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)
        else:
            # Near-silence with tiny dither
            audio = rng.standard_normal(TARGET_LENGTH).astype(np.float32) * 0.001

        emb = _embed_audio_array(audio, model, device)
        embs.append(emb)

    return torch.stack(embs, dim=0)  # (n_clips, 128)


# ---------------------------------------------------------------------------
# Known-class validation embeddings
# ---------------------------------------------------------------------------

def _embed_known_val(
    data_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    val_fraction: float,
    exclude_class: Optional[str] = None,
) -> torch.Tensor:
    """Extract embeddings for a random validation subset of the known dataset.

    Parameters
    ----------
    exclude_class:
        Class name to exclude from the known set (used when that class
        is being treated as the held-out unknown).
    """
    import librosa
    from backend.dataset import AudioDataset

    ds = AudioDataset(data_dir=data_dir, is_train=False)

    # Filter out the held-out class if supplied
    eligible = [
        fp for fp, lbl in ds.samples
        if lbl != exclude_class
    ]

    random.seed(SEED)
    n_val = max(1, int(len(eligible) * val_fraction))
    val_paths = random.sample(eligible, n_val)

    print(
        f"[calibrate_unknown] Using {len(val_paths)} known-class validation clips "
        f"(val_fraction={val_fraction:.0%})"
    )
    return _embed_file_list(val_paths, model, device, desc="Embedding known-val clips")


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def _find_best_threshold(
    known_dists: torch.Tensor,
    unknown_dists: torch.Tensor,
    n_thresholds: int = 500,
) -> Tuple[float, float]:
    """Sweep *n_thresholds* candidate values and pick the one with highest F1.

    Convention:
        - A clip is POSITIVE  (unknown) when min-distance > threshold.
        - A clip is NEGATIVE  (known)   when min-distance ≤ threshold.

    Returns
    -------
    (threshold, best_f1)
    """
    all_dists = torch.cat([known_dists, unknown_dists])
    d_min = float(all_dists.min())
    d_max = float(all_dists.max())

    # Ground-truth labels: 0 = known, 1 = unknown
    y_true = torch.cat([
        torch.zeros(len(known_dists)),
        torch.ones(len(unknown_dists)),
    ])

    best_thresh = (d_min + d_max) / 2.0
    best_f1 = 0.0

    candidates = np.linspace(d_min, d_max, n_thresholds)
    for tau in candidates:
        y_pred = (all_dists > tau).float()

        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(tau)

    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def calibrate_unknown(
    centroids_path: str = DEFAULT_CENTROIDS,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    data_dir: str = DEFAULT_DATA_DIR,
    unknown_dir: Optional[str] = None,
    holdout_class: Optional[str] = None,
    holdout_as_unknown: bool = False,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    synth_clips: int = DEFAULT_SYNTH_CLIPS,
    out_path: str = DEFAULT_OUTPUT,
    batch_size: int = 8,
) -> dict:
    """Calibrate the unknown-detection distance threshold.

    Parameters
    ----------
    centroids_path:
        Path to ``centroids.pt`` produced by ``compute_centroids.py``.
    checkpoint_path:
        Path to the trained model checkpoint (``.pt``).
    data_dir:
        Root data directory.  Required when using holdout-class mode or
        for extracting known-class validation embeddings from real data.
    unknown_dir:
        Optional directory with real audio files for the unknown class.
    holdout_class:
        Name of a known class to treat as the unknown class (for demo /
        evaluation).  E.g. ``"esc_siren"``.
    holdout_as_unknown:
        Must be ``True`` to activate holdout-class mode.
    val_fraction:
        Fraction of the known dataset to use as the validation set.
    synth_clips:
        Number of synthetic unknown clips to generate if no real unknown
        audio is available.
    out_path:
        Output path for ``threshold.pt``.

    Returns
    -------
    dict with ``"threshold"``, ``"best_f1"``, ``"known_dists"``,
    ``"unknown_dists"``, ``"class_names"``.
    """
    # --- Device ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate_unknown] Using device: {device}")

    # --- Load centroids -------------------------------------------------------
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(
            f"Centroids file not found: {centroids_path!r}\n"
            f"Run `python -m backend.compute_centroids` first."
        )
    print(f"[calibrate_unknown] Loading centroids from: {centroids_path}")
    centroid_payload = torch.load(centroids_path, map_location="cpu", weights_only=False)
    centroids: torch.Tensor = centroid_payload["centroids"]   # (C, 128)
    class_names: List[str] = centroid_payload["class_names"]
    print(f"[calibrate_unknown] Centroids shape: {centroids.shape}  ({len(class_names)} classes)")

    # --- Load model (needed for embedding extraction) -------------------------
    print(f"[calibrate_unknown] Loading model from: {checkpoint_path}")
    model, _ = load_model(checkpoint_path, device=device)

    # --- Extract UNKNOWN embeddings ------------------------------------------
    if holdout_as_unknown and holdout_class:
        print(f"[calibrate_unknown] Using held-out class '{holdout_class}' as unknown.")
        # Remove the held-out class's centroid so distances are measured only
        # to the *remaining* known classes
        if holdout_class in class_names:
            holdout_idx = class_names.index(holdout_class)
            mask = torch.ones(len(class_names), dtype=torch.bool)
            mask[holdout_idx] = False
            centroids_for_eval = centroids[mask]
            class_names_for_eval = [c for c in class_names if c != holdout_class]
            print(
                f"[calibrate_unknown] Removed '{holdout_class}' centroid; "
                f"{len(class_names_for_eval)} centroids remain for distance measurement."
            )
        else:
            print(f"[calibrate_unknown] WARNING: '{holdout_class}' not in centroids; "
                  f"using all centroids.")
            centroids_for_eval = centroids
            class_names_for_eval = class_names

        unknown_embs = _embed_dataset_class(data_dir, holdout_class, model, device)

    elif unknown_dir is not None:
        print(f"[calibrate_unknown] Using real unknown audio from: {unknown_dir}")
        centroids_for_eval = centroids
        class_names_for_eval = class_names
        unknown_embs = _embed_directory(unknown_dir, model, device)

    else:
        print(f"[calibrate_unknown] No real unknown audio provided. "
              f"Generating {synth_clips} synthetic white-noise clips.")
        centroids_for_eval = centroids
        class_names_for_eval = class_names
        unknown_embs = _embed_synthetic_unknown(synth_clips, model, device)

    # --- Extract KNOWN-class VALIDATION embeddings ----------------------------
    data_available = os.path.exists(data_dir)
    if data_available:
        exclude = holdout_class if holdout_as_unknown else None
        try:
            known_embs = _embed_known_val(
                data_dir, model, device, val_fraction, exclude_class=exclude
            )
        except Exception as exc:
            print(f"[calibrate_unknown] Could not load known-val from dataset ({exc}). "
                  f"Falling back to centroid-proximity sampling.")
            data_available = False

    if not data_available:
        # Fallback: generate "known-like" clips by perturbing centroids
        # with small Gaussian noise (reasonable stand-in without real data)
        rng = np.random.default_rng(SEED)
        noise = rng.standard_normal((200, 128)).astype(np.float32) * 2.0
        random_centroids = centroids[np.random.randint(0, len(centroids), size=200)]
        known_embs = random_centroids + torch.tensor(noise)
        print(
            f"[calibrate_unknown] Generated {len(known_embs)} centroid-perturbation "
            f"clips as known-class proxies."
        )

    # --- Compute min-centroid distances ---------------------------------------
    print(f"[calibrate_unknown] Computing min-centroid distances …")
    known_dists = min_centroid_distances(known_embs, centroids_for_eval)
    unknown_dists = min_centroid_distances(unknown_embs, centroids_for_eval)

    print(f"  Known   distances — mean: {known_dists.mean():.4f}  "
          f"std: {known_dists.std():.4f}  "
          f"max: {known_dists.max():.4f}")
    print(f"  Unknown distances — mean: {unknown_dists.mean():.4f}  "
          f"std: {unknown_dists.std():.4f}  "
          f"min: {unknown_dists.min():.4f}")

    # --- Find optimal threshold -----------------------------------------------
    print("[calibrate_unknown] Sweeping thresholds …")
    threshold, best_f1 = _find_best_threshold(known_dists, unknown_dists)

    print(f"\n{'='*60}")
    print(f"  ✅ Calibrated threshold : {threshold:.6f}")
    print(f"  📊 Best F1 score        : {best_f1:.4f}")
    print(f"{'='*60}\n")
    print(
        "Inference rule: flag sound as 'Unknown' when\n"
        f"  min_distance_to_any_centroid  >  {threshold:.6f}"
    )

    # --- Save threshold.pt ----------------------------------------------------
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    payload = {
        "threshold": threshold,
        "metric": "f1",
        "best_f1": best_f1,
        "known_dists": known_dists,
        "unknown_dists": unknown_dists,
        "class_names": class_names_for_eval,
        "known_mean_dist": float(known_dists.mean()),
        "unknown_mean_dist": float(unknown_dists.mean()),
    }
    torch.save(payload, out_path)
    print(f"[calibrate_unknown] Threshold saved → {out_path}")

    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.holdout_as_unknown and not args.holdout_class:
        raise ValueError("--holdout-as-unknown requires --holdout-class to be specified.")

    calibrate_unknown(
        centroids_path=args.centroids,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        unknown_dir=args.unknown_dir,
        holdout_class=args.holdout_class,
        holdout_as_unknown=args.holdout_as_unknown,
        val_fraction=args.val_fraction,
        synth_clips=args.synth_clips,
        out_path=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

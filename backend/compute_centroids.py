"""
backend/compute_centroids.py
============================
P1 Task – Step 1: Centroid Computation

Runs the trained AudioSEDNet model over every training clip, extracts
the 128-dim penultimate-layer (fc_embedding + ReLU) embedding, and
computes the mean embedding ("centroid") for each known class.

Output
------
outputs/centroids.pt  — torch.Tensor of shape (num_classes, 128)
                        alongside a ``class_names`` list saved in the
                        same dict so calibrate_unknown.py can reload
                        both together.

Usage (from project root)
-------------------------
    python -m backend.compute_centroids                        # defaults
    python -m backend.compute_centroids --data-dir data        # explicit path
    python -m backend.compute_centroids --batch-size 16        # larger GPU batch
    python -m backend.compute_centroids --out outputs/centroids.pt
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from backend.dataset import AudioDataset
from backend.utils import load_model

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = os.path.join("outputs", "checkpoints", "best_model.pt")
DEFAULT_OUTPUT = os.path.join("outputs", "centroids.pt")
DEFAULT_DATA_DIR = "data"
DEFAULT_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-class embedding centroids from the training set."
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
        help=f"Root data directory for ESC-50 / UrbanSound8K (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for centroids.pt (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 = main process only)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_centroids(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    out_path: str = DEFAULT_OUTPUT,
    num_workers: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute per-class centroids and save them to *out_path*.

    Parameters
    ----------
    checkpoint_path:
        Path to the trained ``AudioSEDNet`` ``.pt`` checkpoint.
    data_dir:
        Root directory containing ESC-50 and/or UrbanSound8K data.
    batch_size:
        Mini-batch size for inference (no gradients, so can be larger than training).
    out_path:
        File path to save the resulting ``centroids.pt``.
    num_workers:
        Number of DataLoader worker processes.

    Returns
    -------
    dict with:
        ``"centroids"``   : torch.Tensor shape (num_classes, 128)
        ``"class_names"`` : list[str]  length num_classes
    """
    # --- Device selection ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute_centroids] Using device: {device}")

    # --- Load model ---------------------------------------------------------
    print(f"[compute_centroids] Loading model from: {checkpoint_path}")
    model, class_names = load_model(checkpoint_path, device=device)
    num_classes = len(class_names)
    print(f"[compute_centroids] {num_classes} known classes loaded.")

    # --- Build training dataset (no augmentation at inference time) ---------
    print(f"[compute_centroids] Loading training dataset from: {data_dir}")
    dataset = AudioDataset(data_dir=data_dir, is_train=False)  # no augmentation

    # Use the full dataset (all folds) to get best centroid estimates
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[compute_centroids] Dataset size: {len(dataset)} clips")

    # --- Accumulate embeddings per class ------------------------------------
    # class_sum[class_idx]   → running sum of embeddings (128-dim)
    # class_count[class_idx] → number of clips processed for that class
    class_sum: Dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(128))
    class_count: Dict[int, int] = defaultdict(int)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Extracting embeddings", unit="batch")
        for specs, labels in pbar:
            specs = specs.to(device)   # (B, 1, n_mels, T)
            embs = model.extract_embeddings(specs)  # (B, 128)
            embs_cpu = embs.cpu()

            for emb, label_idx in zip(embs_cpu, labels.tolist()):
                class_sum[label_idx] = class_sum[label_idx] + emb
                class_count[label_idx] += 1

    # --- Compute mean centroid per class -----------------------------------
    centroids = torch.zeros(num_classes, 128)
    missing_classes: List[str] = []

    for class_idx in range(num_classes):
        count = class_count.get(class_idx, 0)
        if count == 0:
            missing_classes.append(class_names[class_idx])
            # Centroid stays as zero vector; caller can decide how to handle
        else:
            centroids[class_idx] = class_sum[class_idx] / count

    if missing_classes:
        print(
            f"[compute_centroids] WARNING: No training clips found for "
            f"{len(missing_classes)} class(es): {missing_classes}"
        )

    # --- Save to disk -------------------------------------------------------
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    payload = {
        "centroids": centroids,    # (num_classes, 128)
        "class_names": class_names,
    }
    torch.save(payload, out_path)
    print(f"[compute_centroids] Centroids saved → {out_path}")
    print(f"[compute_centroids] Shape: {centroids.shape}  (num_classes={num_classes}, emb_dim=128)")

    # Per-class clip count summary
    clips_per_class = {class_names[k]: v for k, v in sorted(class_count.items())}
    print(f"[compute_centroids] Clips per class (first 5): "
          f"{dict(list(clips_per_class.items())[:5])} ...")

    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    compute_centroids(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        out_path=args.out,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

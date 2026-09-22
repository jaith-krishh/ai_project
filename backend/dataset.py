import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import librosa

class AudioDataset(Dataset):
    """
    PyTorch Dataset for ESC-50 and UrbanSound8K sound event audio clips.
    Extracts 64-bin log-Mel Spectrogram features.
    """
    def __init__(self, data_dir: str = "data", sr: int = 22050, duration: float = 4.0, n_mels: int = 64, is_train: bool = True):
        self.sr = sr
        self.target_length = int(sr * duration)
        self.n_mels = n_mels
        self.is_train = is_train
        self.samples = []
        self.class_names = []

        esc50_csv = os.path.join(data_dir, "ESC-50", "meta", "esc50.csv")
        esc50_audio_dir = os.path.join(data_dir, "ESC-50", "audio")

        us8k_csv = os.path.join(data_dir, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
        us8k_audio_dir = os.path.join(data_dir, "UrbanSound8K", "audio")

        classes = set()

        # Load ESC-50 if present
        if os.path.exists(esc50_csv) and os.path.exists(esc50_audio_dir):
            df_esc = pd.read_csv(esc50_csv)
            for _, row in df_esc.iterrows():
                file_path = os.path.join(esc50_audio_dir, row["filename"])
                label = f"esc_{row['category']}"
                if os.path.exists(file_path):
                    self.samples.append((file_path, label))
                    classes.add(label)

        # Load UrbanSound8K if present
        if os.path.exists(us8k_csv) and os.path.exists(us8k_audio_dir):
            df_us8k = pd.read_csv(us8k_csv)
            for _, row in df_us8k.iterrows():
                fold_dir = f"fold{row['fold']}"
                file_path = os.path.join(us8k_audio_dir, fold_dir, row["slice_file_name"])
                label = f"us8k_{row['class']}"
                if os.path.exists(file_path):
                    self.samples.append((file_path, label))
                    classes.add(label)

        if not self.samples:
            raise RuntimeError(f"No audio files found in directory: {data_dir}")

        self.class_names = sorted(list(classes))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]

        try:
            audio, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=5.0)
        except Exception:
            audio = np.zeros(self.target_length, dtype=np.float32)

        # Random time shift augmentation during training
        if self.is_train and len(audio) > 0:
            shift = random.randint(-int(self.sr * 0.3), int(self.sr * 0.3))
            audio = np.roll(audio, shift)

        # Pad or trim to target length
        if len(audio) < self.target_length:
            pad_len = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        else:
            audio = audio[:self.target_length]

        # Gaussian Noise Augmentation during training
        if self.is_train and random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        # Compute log-Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels, n_fft=1024, hop_length=512
        )
        log_S = librosa.power_to_db(S, ref=np.max)

        # Standardize spectrogram values
        mean = np.mean(log_S)
        std = np.std(log_S) + 1e-6
        norm_S = (log_S - mean) / std

        # SpecAugment: Frequency masking
        if self.is_train:
            num_freq_masks = random.randint(1, 2)
            for _ in range(num_freq_masks):
                f = random.randint(1, min(8, self.n_mels // 8))
                f0 = random.randint(0, self.n_mels - f)
                norm_S[f0:f0 + f, :] = 0.0

            # SpecAugment: Time masking
            num_time_masks = random.randint(1, 2)
            time_steps = norm_S.shape[1]
            for _ in range(num_time_masks):
                t = random.randint(1, min(15, time_steps // 8))
                t0 = random.randint(0, time_steps - t)
                norm_S[:, t0:t0 + t] = 0.0

        # Convert to Tensor (1, n_mels, time_steps)
        spec_tensor = torch.tensor(norm_S, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(label_idx, dtype=torch.long)

        return spec_tensor, target_tensor


def get_dataloaders(data_dir: str = "data", batch_size: int = 8, val_split: float = 0.2, num_workers: int = 2):
    full_dataset = AudioDataset(data_dir=data_dir, is_train=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_dataset.dataset.is_train = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full_dataset.class_names

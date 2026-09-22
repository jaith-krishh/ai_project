import librosa
import numpy as np

def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Load an audio file and resample to `sr`.
    Returns a 1‑D NumPy array.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def mel_spectrogram(audio: np.ndarray, sr: int = 22050, n_mels: int = 128,
                    hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    """Convert raw audio to a log‑Mel spectrogram.
    Output shape: (n_mels, time_frames)
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    audio = load_audio(path)
    spec = mel_spectrogram(audio)
    print(f"Spectrogram shape: {spec.shape}")

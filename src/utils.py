import os
import sys
import torch
import torchaudio
import soundfile as sf
from src.models import AudioFeatureExtractor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import SAMPLE_RATE


def load_audio_file(filepath: str):
    """Loads an audio file safely using soundfile to bypass torchaudio torchcodec dependency."""
    data, sr = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.T
    return tensor, sr


def load_and_preprocess_audio(audio_path: str, device: torch.device) -> torch.Tensor:
    """Preprocesses audio file into a 3-channel spectrogram tensor."""
    waveform, sr = load_audio_file(audio_path)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Normalize duration to 1.0 second (16,000 samples)
    target_len = SAMPLE_RATE
    current_len = waveform.shape[1]

    if current_len < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - current_len))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # Extract 3-channel features (Log-Mel + Delta + Delta-Delta)
    extractor = AudioFeatureExtractor().to(device)
    extractor.eval()

    waveform = waveform.to(device)
    with torch.no_grad():
        features = extractor(waveform)

    return features
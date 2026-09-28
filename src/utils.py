import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import SAMPLE_RATE, TARGET_SAMPLES, N_FFT, HOP_LENGTH, N_MELS

def load_and_preprocess_audio(audio_path, device="cpu"):
    """Loads an audio file, resamples to 16kHz mono, pads/crops to 1s, and converts to 3-channel Log-Mel + Deltas."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or crop to target sample count (16,000 samples)
    num_samples = waveform.shape[1]
    if num_samples < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - num_samples
        waveform = F.pad(waveform, (0, padding))
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]

    # Compute Log-Mel Spectrogram (Channel 1)
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-6)

    # Compute Delta (Channel 2) and Delta-Delta (Channel 3) - v2.0.0
    compute_deltas = T.ComputeDeltas()
    delta_spec = compute_deltas(log_mel_spec)
    delta2_spec = compute_deltas(delta_spec)

    # Concatenate along channel dimension -> Shape: [3, 64, 63]
    three_channel_spec = torch.cat([log_mel_spec, delta_spec, delta2_spec], dim=0)

    # Output shape: [1, 3, 64, 63]
    return three_channel_spec.unsqueeze(0).to(device)
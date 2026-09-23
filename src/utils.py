import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from configs.config import SAMPLE_RATE, TARGET_SAMPLES, N_FFT, HOP_LENGTH, N_MELS

def load_and_preprocess_audio(audio_path, device="cpu"):
    """Loads an audio file, resamples to 16kHz mono, pads/crops to 1s, and converts to Log-Mel."""
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

    # Compute Log-Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-6)

    # Output shape: [1, 1, 64, 63]
    return log_mel_spec.unsqueeze(0).to(device)
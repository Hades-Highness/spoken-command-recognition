import torch
import torch.nn as nn
import torchaudio.transforms as T
from configs.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

class AudioFeatureExtractor(nn.Module):
    """GPU-accelerated Log-Mel + Delta + Delta-Delta feature extractor."""
    def __init__(self):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.compute_deltas = T.ComputeDeltas()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        log_mel = torch.log(self.mel_spectrogram(x) + 1e-6)
        delta = self.compute_deltas(log_mel)
        delta2 = self.compute_deltas(delta)
        
        return torch.cat([log_mel, delta, delta2], dim=1)


class CommandSense(nn.Module):
    """Standard CNN architecture for CommandSense v2.0 (3-channel input)."""
    def __init__(self, num_classes=35, in_channels=3):
        super(CommandSense, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc(x)
        return x
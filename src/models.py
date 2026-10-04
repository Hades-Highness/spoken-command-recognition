import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, NUM_CLASSES


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


class ResidualBlock(nn.Module):
    """Standard residual block for 2D (time x frequency) spectrograms."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut, projected when the shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CommandSense(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)    # -> [64, 32, 32]
        self.layer2 = ResidualBlock(64, 128, stride=2)   # -> [128, 16, 16]
        self.layer3 = ResidualBlock(128, 256, stride=2)  # -> [256, 8, 8]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3)

    dummy_input = torch.randn(4, 3, 64, 63)
    dummy_output = model(dummy_input)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("--- CommandSense architecture check ---")
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {dummy_output.shape} (expected: [4, {NUM_CLASSES}])")
    print(f"Trainable parameters: {num_params:,}  (expected ~1,216,000)")
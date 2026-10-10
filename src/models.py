"""Audio front-end and residual classifier for every CommandSense release.

The classifier is a compact 2D ResNet over Log-Mel + Delta + Delta-Delta
spectrograms. From v4.0 on, every residual block is followed by a
Squeeze-and-Excitation (SE) module that recalibrates its channels after the
residual sum. The SE stage adds under 2% of the total parameters, keeps the
``[B, 3, 64, 63] -> [B, num_classes]`` contract untouched and stays fully
ONNX-exportable. It is optional: ``use_se=False`` reproduces the exact v3.0
graph, parameter count and ``state_dict()``.
"""

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


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., 2018) for 2D feature maps.

    On an input ``[B, C, H, W]`` the block performs three steps:

    1. *Squeeze* - a global average pool collapses the spatial dimensions into
       a ``[B, C]`` channel descriptor.
    2. *Excitation* - a two-layer bottleneck MLP (``C -> C // reduction_ratio
       -> C``) learns a per-channel gate, squashed into ``(0, 1)`` by a sigmoid.
    3. *Scale* - the gate multiplies the input channel-wise and broadcasts over
       the spatial dimensions, returning ``[B, C, H, W]``.

    Only the two linear layers carry parameters, so the block contributes a
    negligible fraction of the total weight budget. It introduces no buffer,
    which keeps ``state_dict()`` clean, and its ops (GlobalAveragePool, Gemm,
    Relu, Sigmoid, Mul) are all natively supported by ONNX.
    """

    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        # Guard the bottleneck against a reduction ratio larger than C.
        hidden_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        squeezed = self.avg_pool(x).flatten(1)
        # Excitation: [B, C] -> [B, C // reduction_ratio] -> [B, C] in (0, 1)
        weights = self.sigmoid(self.fc2(self.relu(self.fc1(squeezed))))
        # Un-squeeze: [B, C] -> [B, C, 1, 1] so the gate broadcasts over H and W.
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights


class ResidualBlock(nn.Module):
    """Standard residual block for 2D (time x frequency) spectrograms."""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True, se_reduction=8):
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

        # Channel recalibration applied to the fused residual output. Kept as a
        # plain attribute (``None`` when disabled) so the v3.0-compatible
        # configuration keeps the exact same ``state_dict()`` as before.
        self.se = (
            SqueezeExcitation(out_channels, reduction_ratio=se_reduction)
            if use_se else None
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # Recalibrate the channels once the shortcut has been merged in, then
        # apply the block's final non-linearity.
        if self.se is not None:
            out = self.se(out)
        return F.relu(out)


class CommandSense(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3, use_se=True, se_reduction=8):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-Excitation (v4.0) is wired into every residual block via
        # the two shared switches, so ``use_se=False`` restores the v3.0 graph.
        self.layer1 = ResidualBlock(32, 64, stride=2, use_se=use_se, se_reduction=se_reduction)    # -> [64, 32, 32]
        self.layer2 = ResidualBlock(64, 128, stride=2, use_se=use_se, se_reduction=se_reduction)   # -> [128, 16, 16]
        self.layer3 = ResidualBlock(128, 256, stride=2, use_se=use_se, se_reduction=se_reduction)  # -> [256, 8, 8]

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
    # Side-by-side check: v3.0 (no SE) vs the default v4.0 (SE) configuration.
    model_v3 = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=False)
    model_v4 = CommandSense(num_classes=NUM_CLASSES, in_channels=3, use_se=True)

    dummy_input = torch.randn(2, 3, 64, 63)

    with torch.no_grad():
        output_v3 = model_v3(dummy_input)
        output_v4 = model_v4(dummy_input)

    params_v3 = sum(p.numel() for p in model_v3.parameters() if p.requires_grad)
    params_v4 = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)
    delta = params_v4 - params_v3

    print("--- CommandSense architecture check ---")
    print(f"Input shape  : {tuple(dummy_input.shape)}  (expected: [2, 3, 64, 63])")
    print(f"Output shape : v3.0 {tuple(output_v3.shape)} | v4.0 {tuple(output_v4.shape)}"
          f"  (expected: [2, {NUM_CLASSES}])")
    print(f"v3.0 params (use_se=False): {params_v3:,}")
    print(f"v4.0 params (use_se=True) : {params_v4:,}")
    print(f"SE overhead               : +{delta:,} (+{100.0 * delta / params_v3:.2f}% of v3.0)")
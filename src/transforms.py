"""GPU-side 3-channel feature extraction and dynamic SpecAugment (CommandSense v3.0).

The v3.0 pipeline keeps raw ``int16`` audio in system RAM and performs every
spectral operation on the accelerator, so the CPU never materializes a
spectrogram. One ``nn.Module`` owns the whole front-end:

1. cast ``int16`` -> ``float32`` and normalize by ``/ 32768.0`` (restores [-1, 1]),
2. Mel-Spectrogram (channel 1, log-scaled to match ``AudioFeatureExtractor``),
3. :func:`torchaudio.functional.compute_deltas` (channel 2, Delta),
4. a second ``compute_deltas`` (channel 3, Delta-Delta),
5. stack into ``[Batch, 3, n_mels, time_frames]``,
6. dynamic SpecAugment (independent frequency + time masks per clip).

Every step is a plain tensor op, so the module runs on whatever device it is
moved to - CUDA in production, CPU for the smoke test. It holds no learnable
parameters (only the Mel filterbank buffers), so it never receives gradients.
"""

import os
import sys

import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as AT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE  # noqa: E402

# Division applied to the int16 cache. 32768 (2**15) maps the full int16 range
# back onto [-1, 1). The dataset quantizes with 32767, so the round-trip error
# stays below one int16 step (~3e-5) - far below the mel filterbank's resolution.
INT16_NORMALIZER = 32768.0

# Dynamic SpecAugment (Park et al., 2019). Widths are drawn per clip and per
# call, so no two epochs mask the same bins. The values are deliberately mild
# for 1.0 s commands: 64 mel bins / 63 frames leave little room for large holes.
FREQ_MASK_PARAM = 8
TIME_MASK_PARAM = 10
N_FREQ_MASKS = 2
N_TIME_MASKS = 2


class AudioToThreeChannelMel(nn.Module):
    """Raw int16 audio -> GPU Log-Mel + Delta + Delta-Delta (with SpecAugment).

    Input : ``[Batch, 1, T]`` (``torch.int16``, or ``[Batch, 1, T]`` float32).
    Output: ``[Batch, 3, n_mels, time_frames]`` (``torch.float32``).

    SpecAugment is applied only in ``train()`` mode; call ``.eval()`` for the
    deterministic features used by validation, export and inference.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        augment: bool = True,
        freq_mask_param: int = FREQ_MASK_PARAM,
        time_mask_param: int = TIME_MASK_PARAM,
        n_freq_masks: int = N_FREQ_MASKS,
        n_time_masks: int = N_TIME_MASKS,
    ):
        super().__init__()
        self.augment = augment
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

        # Registered buffers move with ``.to(device)`` / ``.cuda()``.
        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # iid_masks=True draws an independent mask for every clip, which is what
        # makes the augmentation dynamic per sample instead of sharing one hole.
        self.freq_masking = nn.ModuleList(
            AT.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            for _ in range(n_freq_masks)
        )
        self.time_masking = nn.ModuleList(
            AT.TimeMasking(
                time_mask_param=time_mask_param, iid_masks=True
            )
            for _ in range(n_time_masks)
        )

    @property
    def spec_augment_enabled(self) -> bool:
        """SpecAugment runs only while the module is in training mode."""
        return self.augment and self.training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. int16 -> float32 and normalize.
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x / INT16_NORMALIZER

        if x.ndim == 2:  # [B, T] is accepted as single-channel audio
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"expected audio shaped [Batch, 1, T], got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:  # defensive: downmix any stray channels
            x = x.mean(dim=1, keepdim=True)

        # MelSpectrogram consumes ``(..., time)``. Dropping the mono channel keeps
        # the output ``[Batch, n_mels, frames]`` so stacking yields ``[Batch, 3, ...]``.
        x = x.squeeze(1)

        # 2. Log-Mel spectrogram (channel 1).
        log_mel = torch.log(self.mel_spectrogram(x) + 1e-6)

        # 3./4. Delta and Delta-Delta along the time axis (channels 2 and 3).
        delta = AF.compute_deltas(log_mel)
        delta2 = AF.compute_deltas(delta)

        # 5. Stack channels: [Batch, 3, n_mels, time_frames].
        features = torch.stack((log_mel, delta, delta2), dim=1)

        # 6. Dynamic SpecAugment, directly on the features' device.
        if self.spec_augment_enabled:
            features = self._apply_spec_augment(features)

        return features

    def _apply_spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        for masking in self.freq_masking:
            features = masking(features)
        for masking in self.time_masking:
            features = masking(features)
        return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AudioToThreeChannelMel().to(device).eval()
    dummy = torch.randint(
        -32768, 32767, (4, 1, SAMPLE_RATE), dtype=torch.int16
    ).to(device)
    with torch.no_grad():
        output = extractor(dummy)
    print("--- AudioToThreeChannelMel check ---")
    print(f"device       : {device}")
    print(f"input shape  : {tuple(dummy.shape)} ({dummy.dtype})")
    print(f"output shape : {tuple(output.shape)} ({output.dtype})")

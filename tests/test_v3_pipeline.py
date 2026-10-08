"""Smoke test for the v3.0 memory-optimized pipeline (no training, no dataset).

Runs one forward pass of ``AudioToThreeChannelMel`` on a dummy int16 batch and
asserts the documented ``[Batch, 3, n_mels, time_frames]`` contract. It also
checks the loader-level standardization helper (mono / 16 kHz / fixed length /
int16), so the raw-audio cache format is verified without touching the Speech
Commands tree on disk.

Run directly:  python tests/test_v3_pipeline.py
"""

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import HOP_LENGTH, N_MELS, SAMPLE_RATE, TARGET_SAMPLES  # noqa: E402
from src.dataset import waveform_to_int16  # noqa: E402
from src.transforms import AudioToThreeChannelMel  # noqa: E402

BATCH_SIZE = 8
EXPECTED_TIME_FRAMES = TARGET_SAMPLES // HOP_LENGTH + 1  # 63 frames for 1.0 s


def _dummy_int16_batch(batch_size=BATCH_SIZE, device="cpu"):
    """Random int16 audio shaped like the dataset cache: [B, 1, TARGET_SAMPLES]."""
    batch = torch.randint(
        -32768, 32767, (batch_size, 1, TARGET_SAMPLES), dtype=torch.int16
    )
    return batch.to(device)


def test_transform_output_shape():
    """One forward pass must yield [Batch, 3, n_mels, time] on the active device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AudioToThreeChannelMel().to(device).eval()
    batch = _dummy_int16_batch(device=device)

    with torch.no_grad():
        features = extractor(batch)

    expected = (BATCH_SIZE, 3, N_MELS, EXPECTED_TIME_FRAMES)
    assert tuple(features.shape) == expected, (
        f"feature shape {tuple(features.shape)} != expected {expected}"
    )
    assert features.dtype == torch.float32, features.dtype
    assert torch.isfinite(features).all(), "features contain non-finite values"
    print(f"[OK] forward pass: {tuple(batch.shape)} -> {tuple(features.shape)} on {device}")
    return features


def test_spec_augment_is_dynamic_and_train_only():
    """eval() is deterministic; train() injects zero-filled masks."""
    torch.manual_seed(0)
    extractor = AudioToThreeChannelMel().eval()
    batch = _dummy_int16_batch()

    with torch.no_grad():
        eval_first = extractor(batch)
        eval_second = extractor(batch)
    assert torch.equal(eval_first, eval_second), "eval() must be deterministic"

    extractor.train()
    with torch.no_grad():
        train_features = extractor(batch)

    assert train_features.shape == eval_first.shape
    train_zeros = int((train_features == 0).sum())
    eval_zeros = int((eval_first == 0).sum())
    assert train_zeros > eval_zeros, "SpecAugment injected no masks during train()"
    print(f"[OK] SpecAugment: eval() deterministic, train() zeroed {train_zeros - eval_zeros} bins")


def test_waveform_standardization_to_int16():
    """Loading-time standardization: mono / 16 kHz / exactly 1.0 s / int16."""
    # 44.1 kHz stereo, shorter than 1.0 s -> resample, downmix and zero-pad.
    stereo = torch.randn(2, int(44100 * 0.6))
    audio = waveform_to_int16(stereo, 44100)
    assert audio.dtype == torch.int16, audio.dtype
    assert tuple(audio.shape) == (1, TARGET_SAMPLES), tuple(audio.shape)

    # Over-long mono clips are cropped, short mono clips are padded.
    long_clip = waveform_to_int16(torch.randn(1, TARGET_SAMPLES * 2), SAMPLE_RATE)
    short_clip = waveform_to_int16(torch.randn(1, 1000), SAMPLE_RATE)
    assert tuple(long_clip.shape) == (1, TARGET_SAMPLES)
    assert tuple(short_clip.shape) == (1, TARGET_SAMPLES)
    assert int(short_clip.abs().sum()) > 0, "real samples were lost before the pad"
    print("[OK] standardization: mono / 16 kHz / fixed length / int16")


def main():
    test_transform_output_shape()
    test_spec_augment_is_dynamic_and_train_only()
    test_waveform_standardization_to_int16()
    print("\n[OK] v3 pipeline smoke test passed")


if __name__ == "__main__":
    main()

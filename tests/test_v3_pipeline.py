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


def test_waveform_augmentation_is_train_only_and_masked():
    """eval() is a no-op; in train() only the eligible rows are touched."""
    from src.transforms import WaveformAugmentation

    torch.manual_seed(0)
    augmenter = WaveformAugmentation(p_pitch=1.0, p_time=1.0)
    audio = torch.randn(4, 1, TARGET_SAMPLES) * 0.1
    mask = torch.tensor([True, False, True, False])

    augmenter.eval()
    with torch.no_grad():
        assert torch.equal(augmenter(audio, mask), audio), "eval() must be a no-op"

    augmenter.train()
    with torch.no_grad():
        out = augmenter(audio, mask)

    assert tuple(out.shape) == tuple(audio.shape), tuple(out.shape)
    # Rows flagged False are never touched; rows flagged True always change
    # because both augmentations are forced on (p_pitch = p_time = 1.0).
    assert torch.equal(out[~mask], audio[~mask]), "masked-out rows were modified"
    assert not torch.equal(out[mask], audio[mask]), "eligible rows were left untouched"
    print("[OK] WaveformAugmentation: eval() no-op, train() only on the mask")


def test_silence_clips_are_never_augmented():
    """train() must leave the '_silence_' rows identical to the eval() features."""
    torch.manual_seed(0)
    silence_index = 0  # '_silence_' sorts first in configs/labels.json
    extractor = AudioToThreeChannelMel(silence_index=silence_index)
    batch = _dummy_int16_batch()
    targets = torch.tensor([0, 1, 2, 0, 3, 1, 2, 0], dtype=torch.long)
    targets = targets[: batch.shape[0]]
    silence_rows = targets == silence_index
    assert bool(silence_rows.any()) and bool((~silence_rows).any())

    extractor.eval()
    with torch.no_grad():
        reference = extractor(batch, targets)

    extractor.train()
    with torch.no_grad():
        augmented = extractor(batch, targets)

    assert torch.equal(
        augmented[silence_rows], reference[silence_rows]
    ), "'_silence_' rows changed in train() (waveform and/or SpecAugment leaked)"
    assert not torch.equal(
        augmented[~silence_rows], reference[~silence_rows]
    ), "non-silence rows were not augmented in train()"
    print("[OK] conditional augmentation: '_silence_' rows untouched in train()")


def main():
    test_transform_output_shape()
    test_spec_augment_is_dynamic_and_train_only()
    test_waveform_standardization_to_int16()
    test_waveform_augmentation_is_train_only_and_masked()
    test_silence_clips_are_never_augmented()
    print("\n[OK] v3 pipeline smoke test passed")


if __name__ == "__main__":
    main()

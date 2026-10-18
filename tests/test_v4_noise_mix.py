"""Smoke test for the v4.0 SNR noise-mixing augmentation (no dataset, no training).

Verifies the contracts of ``WaveformNoiseMixer``: ``eval()`` is a no-op,
'_silence_' is never mixed, the mixture lands on the requested SNR (with the
command kept louder than background speech), and the output stays inside
[-1, 1]. The last test drives the full ``AudioToThreeChannelMel`` with a
synthetic noise bank to prove the mixer is wired into the GPU front-end.

Run directly:  python tests/test_v4_noise_mix.py
"""

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    SNR_AMBIENT_MAX_DB,
    SNR_AMBIENT_MIN_DB,
    SNR_SPEECH_MAX_DB,
    SNR_SPEECH_MIN_DB,
    TARGET_SAMPLES,
)
from src.transforms import AudioToThreeChannelMel, WaveformNoiseMixer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
SILENCE_INDEX = 0


def _int16_pool(count=16, amplitude=3000, seed=0):
    """Synthetic int16 noise pool in the dataset's ``[N, 1, T]`` cache layout."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -amplitude,
        amplitude,
        (count, 1, TARGET_SAMPLES),
        dtype=torch.int16,
        generator=generator,
    )


def _quiet_clean(batch=BATCH_SIZE, scale=0.05, seed=0):
    """Clean audio small enough that no peak scaling is needed (gain == 1)."""
    generator = torch.Generator().manual_seed(seed)
    audio = (torch.rand(batch, 1, TARGET_SAMPLES, generator=generator) - 0.5) * 2.0
    return (audio * scale).to(DEVICE)


def _snr_db(clean, mixed):
    residual = mixed - clean
    return 10.0 * torch.log10(clean.pow(2).mean() / residual.pow(2).mean())


def test_eval_is_a_no_op():
    """Validation, testing and export run in eval(): the mixer must vanish."""
    mixer = WaveformNoiseMixer(ambient_noise=_int16_pool(), p=1.0).to(DEVICE)
    audio = _quiet_clean()

    mixer.eval()
    with torch.no_grad():
        assert torch.equal(mixer(audio), audio), "eval() must be a no-op"

    mixer.train()
    with torch.no_grad():
        assert not torch.equal(mixer(audio), audio), "train() did not mix"
    print("[OK] WaveformNoiseMixer: eval() no-op, train() mixes")


def test_probability_zero_disables_mixing():
    mixer = WaveformNoiseMixer(ambient_noise=_int16_pool(), p=0.0).to(DEVICE).train()
    audio = _quiet_clean()
    with torch.no_grad():
        assert torch.equal(mixer(audio), audio), "p=0 must disable mixing"
    print("[OK] WaveformNoiseMixer: p=0 is a no-op")


def test_silence_is_never_mixed():
    """'_silence_' keeps its pure background noise, mask and targets agreeing."""
    mixer = WaveformNoiseMixer(
        ambient_noise=_int16_pool(),
        speech_noise=_int16_pool(seed=1),
        p=1.0,
        silence_index=SILENCE_INDEX,
    ).to(DEVICE)
    mixer.train()

    audio = _quiet_clean()
    targets = torch.tensor([0, 1, 2, 0, 3, 4, 0, 5], device=DEVICE)
    silence = targets == SILENCE_INDEX

    with torch.no_grad():
        mixed = mixer(audio, targets != SILENCE_INDEX, targets)

    assert torch.equal(mixed[silence], audio[silence]), "'_silence_' was mixed"
    assert not torch.equal(mixed[~silence], audio[~silence]), "commands were not mixed"
    print("[OK] WaveformNoiseMixer: '_silence_' rows untouched")


def test_ambient_snr_matches_target():
    """The measured SNR of an ambient mix equals the requested dB value."""
    clean = _quiet_clean()
    for target_db in (SNR_AMBIENT_MIN_DB, SNR_AMBIENT_MAX_DB):
        mixer = WaveformNoiseMixer(
            ambient_noise=_int16_pool(),
            p=1.0,
            ambient_snr_db=(target_db, target_db),
        ).to(DEVICE)
        mixer.train()
        with torch.no_grad():
            mixed = mixer(clean)
        measured = _snr_db(clean, mixed).item()
        assert abs(measured - target_db) < 0.01, (
            f"requested {target_db} dB, measured {measured:.4f} dB"
        )
        assert bool(mixed.abs().max() <= 1.0), "mixture left the [-1, 1] range"
    print(
        f"[OK] ambient SNR: {SNR_AMBIENT_MIN_DB:.0f} dB and "
        f"{SNR_AMBIENT_MAX_DB:.0f} dB reproduced exactly"
    )


def test_speech_snr_keeps_command_louder():
    """Overlapping speech never gets closer than SNR_SPEECH_MIN_DB to a command."""
    mixer = WaveformNoiseMixer(speech_noise=_int16_pool(seed=2), p=1.0).to(DEVICE)
    mixer.train()

    clean = _quiet_clean()
    with torch.no_grad():
        mixed = mixer(clean)

    residual = mixed - clean
    per_row = 10.0 * torch.log10(
        clean.pow(2).mean(dim=-1) / residual.pow(2).mean(dim=-1)
    )
    assert per_row.min().item() >= SNR_SPEECH_MIN_DB - 1e-3, per_row
    assert per_row.max().item() <= SNR_SPEECH_MAX_DB + 1e-3, per_row
    print(
        f"[OK] speech SNR: {per_row.min().item():.2f}-{per_row.max().item():.2f} dB "
        f"(range {SNR_SPEECH_MIN_DB:.0f}-{SNR_SPEECH_MAX_DB:.0f} dB)"
    )


def test_loud_input_is_bounded():
    """A loud command plus loud noise must still fit the int16 / Mel path."""
    mixer = WaveformNoiseMixer(
        ambient_noise=_int16_pool(amplitude=20000),
        speech_noise=_int16_pool(amplitude=20000, seed=3),
        p=1.0,
    ).to(DEVICE)
    mixer.train()

    generator = torch.Generator().manual_seed(9)
    loud = (torch.rand(BATCH_SIZE, 1, TARGET_SAMPLES, generator=generator) - 0.5) * 1.8
    loud = loud.to(DEVICE)

    with torch.no_grad():
        mixed = mixer(loud)

    assert torch.isfinite(mixed).all(), "mixture contains non-finite values"
    assert bool(mixed.abs().max() <= 1.0), "mixture exceeds [-1, 1]"
    print(f"[OK] bounded output: max |x| = {mixed.abs().max().item():.4f}")


def test_shape_contract_both_layouts():
    """``[B, T]`` and ``[B, 1, T]`` in, the same shape and dtype out."""
    mixer = WaveformNoiseMixer(ambient_noise=_int16_pool(), p=1.0).to(DEVICE).train()
    flat = _quiet_clean().squeeze(1)
    with torch.no_grad():
        out_flat = mixer(flat)
    assert tuple(out_flat.shape) == tuple(flat.shape)
    assert out_flat.dtype == torch.float32
    print("[OK] shape contract: [B, T] and [B, 1, T] preserved")


def test_frontend_integration():
    """The mixer is wired into AudioToThreeChannelMel and stays train-only."""
    bank = {"ambient": _int16_pool(), "speech": _int16_pool(seed=5)}
    extractor = AudioToThreeChannelMel(
        silence_index=SILENCE_INDEX,
        noise_bank=bank,
        noise_mix_prob=1.0,
        pitch_shift_prob=0.0,
        time_shift_prob=0.0,
        n_freq_masks=0,
        n_time_masks=0,
    ).to(DEVICE)
    assert extractor.noise_mixer is not None, "noise bank was not injected"
    assert extractor.noise_mixer.ambient_noise.device.type == DEVICE.type

    generator = torch.Generator().manual_seed(11)
    batch = torch.randint(
        -32768,
        32767,
        (BATCH_SIZE, 1, TARGET_SAMPLES),
        dtype=torch.int16,
        generator=generator,
    ).to(DEVICE)
    targets = torch.tensor([0, 1, 2, 0, 3, 4, 0, 5], device=DEVICE)
    silence = targets == SILENCE_INDEX

    extractor.eval()
    with torch.no_grad():
        reference = extractor(batch, targets)

    extractor.train()
    with torch.no_grad():
        mixed = extractor(batch, targets)

    assert torch.equal(mixed[silence], reference[silence]), (
        "'_silence_' features changed in train()"
    )
    assert not torch.equal(mixed[~silence], reference[~silence]), (
        "command features were not noise-mixed in train()"
    )
    assert torch.isfinite(mixed).all()

    # Inference, quantization and tests pass no bank: the v3.0 front-end is kept.
    assert AudioToThreeChannelMel().noise_mixer is None
    print("[OK] front-end integration: train() mixes, eval() and bank-less stay clean")


def main():
    test_eval_is_a_no_op()
    test_probability_zero_disables_mixing()
    test_silence_is_never_mixed()
    test_ambient_snr_matches_target()
    test_speech_snr_keeps_command_louder()
    test_loud_input_is_bounded()
    test_shape_contract_both_layouts()
    test_frontend_integration()
    print("\n[OK] v4 noise mixing smoke test passed")


if __name__ == "__main__":
    main()

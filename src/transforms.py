"""GPU-side 3-channel feature extraction with train-only conditional augmentation.

Raw ``int16`` audio lives in system RAM and every spectral operation runs on the
accelerator, so the CPU never materializes a spectrogram: cast to ``float32`` and
normalize, optionally mix in SNR-controlled noise, augment the waveform, then
Log-Mel + Delta + Delta-Delta into ``[Batch, 3, n_mels, time_frames]``.

Augmentation is *conditional* and train-only: ``_silence_`` teaches the reject
(OOD) boundary from raw background noise, so shifting, masking or mixing it
would only blur that boundary, and ``eval()`` short-circuits every augmentation,
which keeps validation metrics stable and the exported ONNX graph deterministic.
The module holds no learnable parameters, only the Mel filterbank buffers and,
for v4.0, the preloaded noise pools.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (  # noqa: E402
    FREQ_MASK_PARAM,
    HOP_LENGTH,
    N_FFT,
    N_FREQ_MASKS,
    N_MELS,
    N_TIME_MASKS,
    NOISE_MIX_PROB,
    NOISE_SPEECH_SOURCE_PROB,
    PITCH_SHIFT_BINS_PER_OCTAVE,
    PITCH_SHIFT_MODE,
    PITCH_SHIFT_N_FFT,
    PITCH_SHIFT_PROB,
    PITCH_SHIFT_REFERENCE_HZ,
    PITCH_SHIFT_SEMITONES,
    SAMPLE_RATE,
    SILENCE_CLASS_INDEX,
    SNR_AMBIENT_MAX_DB,
    SNR_AMBIENT_MIN_DB,
    SNR_SPEECH_MAX_DB,
    SNR_SPEECH_MIN_DB,
    TARGET_SAMPLES,
    TIME_MASK_PARAM,
    TIME_SHIFT_PROB,
    TIME_SHIFT_RATIO,
)

# 32768 (2**15) maps the full int16 range back onto [-1, 1). The dataset quantizes
# with 32767, so the round-trip error stays below one int16 step (~3e-5).
INT16_NORMALIZER = 32768.0


class WaveformAugmentation(nn.Module):
    """GPU-native per-clip Time-Shift + Pitch-Shift for the v3.0 front-end.

    Input : ``[Batch, T]`` or ``[Batch, 1, T]`` float audio in [-1, 1] (the
            front-end normalizes the int16 cache before calling this module).
    Output: same shape, dtype and device as the input.

    Everything here is strictly ``train()``-only and runs under
    ``torch.no_grad()`` (enforced by the decorator below), so ``eval()``
    (validation, ONNX export, inference) stays bit-for-bit deterministic and no
    autograd graph is ever built for data preparation. ``augment_mask`` (bool
    ``[Batch]``, ``True`` = may be perturbed) lets the caller shield the
    ``_silence_`` clips, which keep the raw background noise the reject boundary
    is learnt from.

    Both shifts are drawn from a small discrete set, so clips sharing a value go
    through a single batched call: the loop count is bounded by the number of
    distinct values (2 for ``PITCH_SHIFT_SEMITONES = (-1, 1)``), never by the
    batch size. The default spectral pitch shift is ~180x cheaper than the phase
    vocoder; see ``PITCH_SHIFT_MODE`` in ``configs/config.py``.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        p_pitch: float = PITCH_SHIFT_PROB,
        pitch_shift_semitones=PITCH_SHIFT_SEMITONES,
        pitch_mode: str = PITCH_SHIFT_MODE,
        pitch_reference_hz: float = PITCH_SHIFT_REFERENCE_HZ,
        pitch_n_fft: int = PITCH_SHIFT_N_FFT,
        bins_per_octave: int = PITCH_SHIFT_BINS_PER_OCTAVE,
        p_time: float = TIME_SHIFT_PROB,
        time_shift_ratio: float = TIME_SHIFT_RATIO,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.p_pitch = p_pitch
        self.pitch_shift_semitones = tuple(pitch_shift_semitones)
        self.pitch_mode = str(pitch_mode).lower()
        if self.pitch_mode not in ("spectral", "vocoder", "off"):
            raise ValueError(
                "pitch_mode must be 'spectral', 'vocoder' or 'off', "
                f"got {pitch_mode!r}"
            )
        self.pitch_reference_hz = pitch_reference_hz
        self.pitch_n_fft = pitch_n_fft
        self.pitch_hop_length = pitch_n_fft // 4
        self.bins_per_octave = bins_per_octave
        self.p_time = p_time
        self.time_shift_ratio = time_shift_ratio

        # Semitone -> STFT bin is a fixed mapping solved once here, so it stays out
        # of the per-batch hot path.
        self.pitch_bin_shifts = tuple(
            self._semitone_to_bins(semitone)
            for semitone in self.pitch_shift_semitones
        )
        # Non-persistent: absent from state_dict, so checkpointing and the ONNX
        # export are untouched.
        self.register_buffer(
            "_pitch_window",
            torch.hann_window(pitch_n_fft, periodic=True),
            persistent=False,
        )

    @property
    def enabled(self) -> bool:
        """Augmentation runs only while the module is in training mode."""
        return self.training

    @torch.no_grad()
    def forward(
        self, waveform: torch.Tensor, augment_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if not self.enabled:
            return waveform

        squeeze = waveform.ndim == 3 and waveform.shape[1] == 1
        audio = waveform.squeeze(1) if squeeze else waveform
        if audio.ndim != 2:
            raise ValueError(
                "WaveformAugmentation expects [Batch, T] or [Batch, 1, T], "
                f"got {tuple(waveform.shape)}"
            )

        if augment_mask is None:
            allowed = torch.ones(
                audio.shape[0], dtype=torch.bool, device=audio.device
            )
        else:
            allowed = augment_mask.reshape(-1).to(torch.bool)
        if not bool(allowed.any()):
            return waveform

        audio = self._time_shift(audio, allowed)
        audio = self._pitch_shift(audio, allowed)
        return audio.unsqueeze(1) if squeeze else audio

    @torch.no_grad()
    def _time_shift(
        self, audio: torch.Tensor, allowed: torch.Tensor
    ) -> torch.Tensor:
        """One native :func:`torch.roll`, with the wrap-around left in place.

        ``torch.roll`` is a slice-concat that the accelerator runs at memory
        bandwidth (~3 ms for a 256 x 16000 batch) where the previous
        ``gather`` + zero-fill needed a ``[batch, length]`` int64 index, a
        clamp, a mask multiply and a mask cast every single batch (~24 ms). The
        artifact it trades in is a few wrapped samples at the window edge, which
        is negligible for a fixed 1.0 s command clip.

        A single shift is drawn per batch and shared by every eligible clip.
        That is not a loss of diversity: the loader shuffles, so each clip still
        draws a fresh shift once per epoch, and it is what keeps the cost at one
        kernel launch. ``_silence_`` rows are never touched (they are excluded
        from the roll target, not merely zeroed afterwards).
        """
        _, length = audio.shape
        max_shift = int(round(length * self.time_shift_ratio))
        if self.p_time <= 0.0 or max_shift == 0:
            return audio

        device = audio.device
        if float(torch.rand((), device=device)) >= self.p_time:
            return audio
        shift = int(torch.randint(-max_shift, max_shift + 1, (), device=device))
        if shift == 0:
            return audio

        if bool(allowed.all()):
            return torch.roll(audio, shifts=shift, dims=-1)
        shifted = audio.clone()
        shifted[allowed] = torch.roll(audio[allowed], shifts=shift, dims=-1)
        return shifted

    @torch.no_grad()
    def _pitch_shift(
        self, audio: torch.Tensor, allowed: torch.Tensor
    ) -> torch.Tensor:
        """Batched pitch shift, grouped by the discrete shift values in use.

        Gating is per clip (``p_pitch``) and a clip draws one of
        ``pitch_shift_semitones``; clips that drew the same value share a single
        batched call, so the number of calls equals the number of distinct values
        (2 with ``(-1, 1)``) whatever the batch size. What the call itself does
        depends on ``pitch_mode`` - see :meth:`_spectral_shift` and
        :meth:`_vocoder_shift`.
        """
        if self.p_pitch <= 0.0 or not self.pitch_shift_semitones:
            return audio
        if self.pitch_mode == "off":
            return audio

        if self.pitch_mode == "vocoder":
            steps = self.pitch_shift_semitones
        else:
            steps = tuple(b for b in self.pitch_bin_shifts if b != 0)
        if not steps:
            return audio

        batch, length = audio.shape
        device = audio.device
        choices = torch.tensor(steps, dtype=torch.long, device=device)
        draw = choices[torch.randint(len(steps), (batch,), device=device)]
        gated = (torch.rand(batch, device=device) < self.p_pitch) & allowed
        chosen = torch.where(gated, draw, torch.zeros_like(draw))
        if not bool((chosen != 0).any()):
            return audio

        shift_fn = (
            self._vocoder_shift if self.pitch_mode == "vocoder" else self._spectral_shift
        )
        augmented = audio
        for step in steps:
            rows = torch.nonzero(chosen == step, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            shifted = shift_fn(audio[rows], int(step), length)
            if augmented is audio:
                augmented = audio.clone()
            augmented[rows] = shifted
        return augmented

    @torch.no_grad()
    def _vocoder_shift(
        self, audio: torch.Tensor, step: int, length: int
    ) -> torch.Tensor:
        """Phase-vocoder shift (``PITCH_SHIFT_MODE = "vocoder"``).

        Highest quality, but proportionally the most expensive: the cost is
        dominated by the internal ``AF.resample`` and is nearly flat per call
        (~3.4 s for 8 clips, ~16.8 s for 256 on CPU), so it cannot ride along in
        the per-batch hot path without multiplying the epoch time.
        """
        shifted = AF.pitch_shift(
            audio,
            self.sample_rate,
            step,
            bins_per_octave=self.bins_per_octave,
            n_fft=self.pitch_n_fft,
        )
        # The vocoder preserves length for our frame-aligned clips, but stay
        # defensive about a possible off-by-a-few-samples output.
        return self._fit_length(shifted, length)

    @torch.no_grad()
    def _spectral_shift(
        self, audio: torch.Tensor, bins: int, length: int
    ) -> torch.Tensor:
        """Shift the spectrum by ``bins`` STFT bins (default, ~180x cheaper).

        The phase vocoder is skipped entirely: harmonics and spectral envelope
        translate together along the frequency axis, which is the effect the
        regularizer is after. Vacated bins are zeroed instead of wrapped so no
        Nyquist fold-back is introduced, and ``length`` is pinned on the inverse
        transform so the front-end frame math never sees a moving target.
        """
        spec = torch.stft(
            audio,
            self.pitch_n_fft,
            self.pitch_hop_length,
            window=self._pitch_window,
            center=True,
            return_complex=True,
        )
        shifted = torch.roll(spec, shifts=bins, dims=-2)
        if bins > 0:
            shifted[..., :bins, :] = 0
        else:
            shifted[..., bins:, :] = 0
        return torch.istft(
            shifted,
            self.pitch_n_fft,
            self.pitch_hop_length,
            window=self._pitch_window,
            length=length,
        )

    def _semitone_to_bins(self, semitone: int) -> int:
        """Map a semitone offset to STFT bins, anchored at
        ``pitch_reference_hz``.

        A true pitch shift is frequency dependent (one semitone is ~59 Hz at
        1 kHz but ~12 Hz at 200 Hz) and a single fixed bin offset cannot follow
        that, so the anchor is made explicit and reproducible: one semitone at the
        reference frequency is ``ref * (2 ** (1/12) - 1)`` Hz, compared against the
        bin width ``sample_rate / pitch_n_fft``. Never returns 0, so a nonzero
        semitone can never degrade into a no-op.
        """
        if semitone == 0:
            return 0
        semitone_hz = self.pitch_reference_hz * (2.0 ** (semitone / 12.0) - 1.0)
        bins = int(round(semitone_hz / (self.sample_rate / self.pitch_n_fft)))
        if bins == 0:
            bins = 1 if semitone > 0 else -1
        return bins

    @staticmethod
    def _fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
        current = audio.shape[-1]
        if current == length:
            return audio
        if current > length:
            return audio[..., :length]
        return F.pad(audio, (0, length - current))


def _noise_pool_tensor(bank: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a noise pool to a contiguous float32 ``[N, T]`` in [-1, 1].

    ``int16`` pools are divided by ``INT16_NORMALIZER``, exactly the conversion
    the front-end applies to the audio cache, so a noise clip and a clean clip
    live on the same scale and the SNR arithmetic below is meaningful. ``None``
    (or an empty pool) yields ``None``, which switches that source off.
    """
    if bank is None:
        return None
    if bank.ndim == 3 and bank.shape[1] == 1:
        bank = bank.squeeze(1)
    if bank.ndim != 2:
        raise ValueError(
            f"'{name}' noise pool must be [N, T] or [N, 1, T], got {tuple(bank.shape)}"
        )
    if bank.shape[0] == 0:
        return None
    if bank.dtype == torch.int16:
        bank = bank.to(torch.float32) / INT16_NORMALIZER
    else:
        bank = bank.to(torch.float32)
    return bank.contiguous()


class WaveformNoiseMixer(nn.Module):
    """SNR-controlled ambient / overlapping-speech mixing for the v4.0 front-end.

    Input : ``[Batch, T]`` or ``[Batch, 1, T]`` float audio in [-1, 1] (the
            front-end normalizes the int16 cache before calling this module),
            the boolean ``eligible`` mask and, defensively, ``targets``.
    Output: same shape, dtype and device as the input; every row that was not
            selected is returned bit-for-bit untouched.

    Three properties keep it safe inside the GPU pipeline:

    * ``train()`` only, like :class:`WaveformAugmentation`. ``eval()`` -
      validation, testing, ONNX export, inference - is an exact no-op, so no
      split outside training ever sees a mixed waveform.
    * ``_silence_`` is never mixed, not even with probability ``p``: the class has
      to stay pure background noise for the OOD rejection boundary. It is
      excluded twice, by ``eligible`` and by ``targets != silence_index``.
    * Both pools are non-persistent buffers preloaded once as ``int16`` and
      promoted to the accelerator by ``.to(device)``, so the hot path performs no
      I/O, allocates no CPU audio path and never leaves PyTorch.

    Mixing is ``y = y_clean + alpha * y_noise`` with

        ``alpha = (rms(y_clean) / rms(y_noise)) * 10 ** (-SNR_dB / 20)``

    The leading RMS ratio puts the two signals on a common level - without it
    ``10 ** (-SNR_dB / 20)`` alone would only define an SNR for clips that happen
    to share an RMS - and the exponential is the requested SNR as a linear gain.
    One target SNR is drawn per eligible row, uniformly inside the ambient or the
    speech range. When the mixture would exceed [-1, 1] both terms are scaled by
    the same factor, which leaves the SNR intact and keeps the downstream int16 /
    Mel path overflow-free.
    """

    def __init__(
        self,
        ambient_noise: torch.Tensor = None,
        speech_noise: torch.Tensor = None,
        p: float = NOISE_MIX_PROB,
        ambient_snr_db=(SNR_AMBIENT_MIN_DB, SNR_AMBIENT_MAX_DB),
        speech_snr_db=(SNR_SPEECH_MIN_DB, SNR_SPEECH_MAX_DB),
        speech_source_prob: float = NOISE_SPEECH_SOURCE_PROB,
        silence_index: int = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.p = float(p)
        self.ambient_snr_db = self._check_snr_range(ambient_snr_db, "ambient_snr_db")
        self.speech_snr_db = self._check_snr_range(speech_snr_db, "speech_snr_db")
        self.speech_source_prob = float(speech_source_prob)
        self.silence_index = silence_index
        self.eps = float(eps)

        # Non-persistent: the pools never enter a state_dict, so checkpoints and
        # the ONNX export stay byte-compatible with the weights they describe.
        self.register_buffer(
            "ambient_noise",
            _noise_pool_tensor(ambient_noise, "ambient"),
            persistent=False,
        )
        self.register_buffer(
            "speech_noise",
            _noise_pool_tensor(speech_noise, "speech"),
            persistent=False,
        )

    @staticmethod
    def _check_snr_range(value, name: str):
        low, high = (float(v) for v in value)
        if high < low:
            raise ValueError(f"{name} must be (min, max) with max >= min, got {value!r}")
        return low, high

    @property
    def has_noise(self) -> bool:
        return self.ambient_noise is not None or self.speech_noise is not None

    @property
    def enabled(self) -> bool:
        """Mixing runs only in training mode, only with a pool, only if p > 0."""
        return self.training and self.p > 0.0 and self.has_noise

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        eligible: torch.Tensor = None,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return waveform

        squeeze = waveform.ndim == 3 and waveform.shape[1] == 1
        audio = waveform.squeeze(1) if squeeze else waveform
        if audio.ndim != 2:
            raise ValueError(
                "WaveformNoiseMixer expects [Batch, T] or [Batch, 1, T], "
                f"got {tuple(waveform.shape)}"
            )

        batch, device = audio.shape[0], audio.device
        if eligible is None:
            allowed = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            allowed = eligible.reshape(-1).to(torch.bool)
        # Defence in depth: '_silence_' is never mixed, whatever the caller
        # passed as ``eligible``.
        if targets is not None and self.silence_index is not None:
            allowed = allowed & (targets.reshape(-1) != self.silence_index)
        if self.p < 1.0:
            allowed = allowed & (torch.rand(batch, device=device) < self.p)
        if not bool(allowed.any()):
            return waveform

        rows = torch.nonzero(allowed, as_tuple=False).flatten()
        if self.ambient_noise is not None and self.speech_noise is not None:
            use_speech = (
                torch.rand(rows.numel(), device=device) < self.speech_source_prob
            )
            ambient_rows = rows[~use_speech]
            speech_rows = rows[use_speech]
        elif self.speech_noise is not None:
            ambient_rows = rows[:0]
            speech_rows = rows
        else:
            ambient_rows = rows
            speech_rows = rows[:0]

        mixed = audio.clone()
        if ambient_rows.numel():
            mixed[ambient_rows] = self._mix_rows(
                audio[ambient_rows], self.ambient_noise, self.ambient_snr_db
            )
        if speech_rows.numel():
            mixed[speech_rows] = self._mix_rows(
                audio[speech_rows], self.speech_noise, self.speech_snr_db
            )
        return mixed.unsqueeze(1) if squeeze else mixed

    @torch.no_grad()
    def _mix_rows(
        self, clean: torch.Tensor, pool: torch.Tensor, snr_range
    ) -> torch.Tensor:
        """Mix ``clean [N, T]`` with one independently drawn pool clip per row."""
        batch, length = clean.shape
        indices = torch.randint(0, pool.shape[0], (batch,), device=pool.device)
        noise = pool[indices].to(clean.device)
        if noise.shape[-1] != length:
            if noise.shape[-1] > length:
                noise = noise[..., :length]
            else:
                noise = F.pad(noise, (0, length - noise.shape[-1]))

        clean_rms = clean.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        noise_rms = noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()

        low, high = snr_range
        snr_db = torch.rand((batch, 1), device=clean.device) * (high - low) + low
        alpha = (clean_rms / noise_rms) * torch.pow(10.0, -snr_db / 20.0)

        return self._limit_peak(clean + alpha * noise)

    def _limit_peak(self, audio: torch.Tensor) -> torch.Tensor:
        """Peak-normalize anything above 1.0, then hard-clamp as a backstop.

        One factor is applied to the whole mixture, so ``clean`` and ``noise``
        keep their ratio and the requested SNR survives; only the absolute level
        changes, which is what keeps the int16 / Mel path in range.
        """
        peak = audio.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            peak > 1.0, 1.0 / peak.clamp_min(self.eps), torch.ones_like(peak)
        )
        return (audio * scale).clamp(-1.0, 1.0)


class AudioToThreeChannelMel(nn.Module):
    """Raw int16 audio -> GPU Log-Mel + Delta + Delta-Delta, conditionally augmented.

    Input : ``[Batch, 1, T]`` (``torch.int16``, or ``[Batch, 1, T]`` float32) and,
             optionally, the batch ``targets`` used to shield ``_silence_`` clips.
    Output: ``[Batch, 3, n_mels, time_frames]`` (``torch.float32``).

    Every augmentation (SNR noise mixing, waveform pitch/time shift *and*
    SpecAugment) is applied only in ``train()`` mode; call ``.eval()`` for the
    deterministic features used by validation, export and inference.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        augment: bool = True,
        waveform_augment: bool = True,
        silence_index: int = SILENCE_CLASS_INDEX,
        freq_mask_param: int = FREQ_MASK_PARAM,
        time_mask_param: int = TIME_MASK_PARAM,
        n_freq_masks: int = N_FREQ_MASKS,
        n_time_masks: int = N_TIME_MASKS,
        pitch_shift_prob: float = PITCH_SHIFT_PROB,
        pitch_shift_semitones=PITCH_SHIFT_SEMITONES,
        pitch_mode: str = PITCH_SHIFT_MODE,
        time_shift_prob: float = TIME_SHIFT_PROB,
        time_shift_ratio: float = TIME_SHIFT_RATIO,
        noise_bank: dict = None,
        noise_mix_prob: float = NOISE_MIX_PROB,
    ):
        super().__init__()
        self.augment = augment
        self.waveform_augment = waveform_augment
        self.silence_index = silence_index
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Waveform augmentation (pitch + time shift) runs first, on the normalized
        # [-1, 1] audio. Its ``training`` flag mirrors this module's, so ``eval()``
        # here also switches the waveform augmentation off.
        self.waveform_augmentation = WaveformAugmentation(
            sample_rate=sample_rate,
            p_pitch=pitch_shift_prob,
            pitch_shift_semitones=pitch_shift_semitones,
            pitch_mode=pitch_mode,
            p_time=time_shift_prob,
            time_shift_ratio=time_shift_ratio,
        )

        # v4.0 SNR noise mixing. ``noise_bank`` carries the two int16 pools that
        # the dataset preloaded in RAM; they become non-persistent buffers, so
        # ``.to(device)`` promotes them to the accelerator exactly once and the
        # training loop never reads noise from disk. Without a bank (inference,
        # quantization, tests) the mixer stays absent and the front-end is the
        # v3.0 pipeline.
        ambient_noise = None
        speech_noise = None
        if noise_bank is not None:
            ambient_noise = noise_bank.get("ambient")
            speech_noise = noise_bank.get("speech")
        self.noise_mixer = None
        if noise_mix_prob > 0.0 and (ambient_noise is not None or speech_noise is not None):
            self.noise_mixer = WaveformNoiseMixer(
                ambient_noise=ambient_noise,
                speech_noise=speech_noise,
                p=noise_mix_prob,
                silence_index=silence_index,
            )

        # iid_masks=True draws an independent mask per clip, so the augmentation
        # stays per-sample instead of sharing one hole across the batch.
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

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor = None
    ) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x / INT16_NORMALIZER

        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"expected audio shaped [Batch, 1, T], got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:  # defensive: downmix any stray channels
            x = x.mean(dim=1, keepdim=True)

        # Train-only, and the '_silence_' rows are excluded by ``augment_mask``
        # (``None`` when no labels were supplied, i.e. every clip is eligible).
        # Noise mixing comes first, so the pitch/time shift perturbs the whole
        # scene (command + background), and it re-checks '_silence_' itself.
        augment_mask = self._augment_mask(targets)
        if self.augment and self.noise_mixer is not None:
            x = self.noise_mixer(x, augment_mask, targets)
        if self.augment and self.waveform_augment:
            x = self.waveform_augmentation(x, augment_mask)

        # MelSpectrogram consumes ``(..., time)``, so the mono channel is dropped.
        x = x.squeeze(1)

        log_mel = torch.log(self.mel_spectrogram(x) + 1e-6)
        delta = AF.compute_deltas(log_mel)
        delta2 = AF.compute_deltas(delta)
        features = torch.stack((log_mel, delta, delta2), dim=1)

        # Gated so the '_silence_' rows stay exactly as they were extracted.
        if self.spec_augment_enabled:
            features = self._apply_spec_augment(features, augment_mask)

        return features

    def _augment_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Bool ``[Batch]`` mask (``True`` = augmentation allowed), or ``None``.

        ``_silence_`` clips are excluded so the reject boundary keeps being learnt
        from untouched background noise.
        """
        if targets is None or self.silence_index is None:
            return None
        return targets.reshape(-1) != self.silence_index

    def _apply_spec_augment(
        self, features: torch.Tensor, augment_mask: torch.Tensor = None
    ) -> torch.Tensor:
        original = features
        augmented = features
        for masking in self.freq_masking:
            augmented = masking(augmented)
        for masking in self.time_masking:
            augmented = masking(augmented)
        # Fast path: nothing to shield, keep the freshly masked tensor as is.
        if augment_mask is None or bool(augment_mask.all()):
            return augmented
        keep = augment_mask.reshape(-1, 1, 1, 1)
        return torch.where(keep, augmented, original)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AudioToThreeChannelMel().to(device)
    dummy = torch.randint(
        -32768, 32767, (4, 1, TARGET_SAMPLES), dtype=torch.int16
    ).to(device)
    targets = torch.tensor([0, 7, 12, 0], device=device)  # 0 == '_silence_'

    extractor.eval()
    with torch.no_grad():
        features = extractor(dummy, targets)
        features_again = extractor(dummy, targets)

    extractor.train()
    with torch.no_grad():
        augmented = extractor(dummy, targets)

    silence_rows = targets == extractor.silence_index
    augmentation = extractor.waveform_augmentation
    print("--- AudioToThreeChannelMel check ---")
    print(f"device         : {device}")
    print(f"input shape    : {tuple(dummy.shape)} ({dummy.dtype})")
    print(f"output shape   : {tuple(features.shape)} ({features.dtype})")
    print(f"eval determin. : {torch.equal(features, features_again)}")
    print(f"silence intact : {torch.equal(augmented[silence_rows], features[silence_rows])}")
    print(f"others augm.   : {not torch.equal(augmented[~silence_rows], features[~silence_rows])}")
    print(f"pitch mode     : {augmentation.pitch_mode} "
          f"(semitone {augmentation.pitch_shift_semitones} -> bins "
          f"{augmentation.pitch_bin_shifts})")
    print(f"shift probs    : pitch={augmentation.p_pitch} time={augmentation.p_time}")

    # Timing on a realistically sized batch: the augmentation budget has to stay
    # a small fraction of the mel front-end, not a multiple of it.
    timed = torch.randint(
        -32768, 32767, (256, 1, TARGET_SAMPLES), dtype=torch.int16
    ).to(device)
    timed_targets = torch.randint(1, 36, (256,), device=device)
    runs = 5
    extractor.eval()
    with torch.no_grad():
        for _ in range(runs):
            extractor(timed, timed_targets)
        start = time.perf_counter()
        for _ in range(runs):
            extractor(timed, timed_targets)
        eval_ms = (time.perf_counter() - start) / runs * 1000.0

    extractor.train()
    with torch.no_grad():
        for _ in range(runs):
            extractor(timed, timed_targets)
        start = time.perf_counter()
        for _ in range(runs):
            extractor(timed, timed_targets)
        train_ms = (time.perf_counter() - start) / runs * 1000.0

    print("--- per-batch timing (256 x 16000) ---")
    print(f"eval  forward  : {eval_ms:8.1f} ms  (no augmentation)")
    print(f"train forward  : {train_ms:8.1f} ms  (augmented)")
    print(f"augmentation   : {train_ms - eval_ms:8.1f} ms  overhead")

    # v4.0 noise mixing check on a synthetic int16 bank: the measured SNR of the
    # mixture must match the requested one, '_silence_' must survive untouched,
    # and the mixed waveform must stay inside [-1, 1].
    bank = {
        "ambient": torch.randint(
            -3000, 3000, (32, 1, TARGET_SAMPLES), dtype=torch.int16
        ),
        "speech": torch.randint(
            -3000, 3000, (32, 1, TARGET_SAMPLES), dtype=torch.int16
        ),
    }
    mixer = WaveformNoiseMixer(
        ambient_noise=bank["ambient"],
        speech_noise=bank["speech"],
        p=1.0,
        speech_source_prob=0.0,
        ambient_snr_db=(10.0, 10.0),
        silence_index=0,
    ).to(device)

    clean = (torch.rand(8, 1, TARGET_SAMPLES, device=device) - 0.5) * 0.2
    probe_targets = torch.tensor([0, 1, 2, 3, 0, 4, 5, 6], device=device)
    mixer.train()
    with torch.no_grad():
        mixed = mixer(clean, probe_targets != 0, probe_targets)
    residual = mixed - clean
    silent = probe_targets == 0
    # Measure on eligible rows only: '_silence_' rows add exact zeros to the
    # residual and would bias the ratio upwards (0.75x power -> +1.25 dB).
    measured_snr = 10.0 * torch.log10(
        clean[~silent].pow(2).mean() / residual[~silent].pow(2).mean()
    )

    print("--- WaveformNoiseMixer check ---")
    print(f"shape preserved : {tuple(mixed.shape) == tuple(clean.shape)}")
    print(f"silence intact  : {torch.equal(mixed[silent], clean[silent])}")
    print(f"measured SNR    : {measured_snr.item():.3f} dB (requested 10.000 dB)")
    print(f"bounded [-1, 1] : {bool(mixed.abs().max() <= 1.0)}")
    print(f"snr ranges      : ambient={mixer.ambient_snr_db} speech={mixer.speech_snr_db}")

    extractor_with_noise = AudioToThreeChannelMel(noise_bank=bank).to(device)
    extractor_with_noise.train()
    with torch.no_grad():
        for _ in range(runs):
            extractor_with_noise(timed, timed_targets)
        start = time.perf_counter()
        for _ in range(runs):
            extractor_with_noise(timed, timed_targets)
        noise_ms = (time.perf_counter() - start) / runs * 1000.0
    print(f"noise mixing   : {noise_ms - train_ms:8.1f} ms  additional overhead")


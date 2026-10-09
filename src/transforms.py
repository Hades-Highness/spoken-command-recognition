"""GPU-side 3-channel feature extraction with train-only conditional augmentation.

The v3.0 pipeline keeps raw ``int16`` audio in system RAM and performs every
spectral operation on the accelerator, so the CPU never materializes a
spectrogram. One ``nn.Module`` owns the whole front-end, and every step below
runs on whatever device it is moved to (CUDA in production, CPU for the smoke
test):

1. cast ``int16`` -> ``float32`` and normalize by ``/ 32768.0`` (restores [-1, 1]),
2. waveform augmentation (:class:`WaveformAugmentation`): per-clip pitch shift and
   time shift, applied only in ``train()`` mode and never to ``_silence_`` clips,
3. Mel-Spectrogram (channel 1, log-scaled to match ``AudioFeatureExtractor``),
4. :func:`torchaudio.functional.compute_deltas` (channel 2, Delta),
5. a second ``compute_deltas`` (channel 3, Delta-Delta),
6. stack into ``[Batch, 3, n_mels, time_frames]``,
7. dynamic SpecAugment (independent frequency + time masks per clip), also gated
   away from ``_silence_`` clips.

The augmentation is deliberately *conditional*: ``_silence_`` teaches the reject
(OOD) boundary from raw background noise, so shifting or masking it would only
blur that boundary. It is also entirely train-time - ``eval()`` short-circuits
both augmentations, which keeps validation metrics stable and the exported ONNX
graph deterministic.

The module holds no learnable parameters (only the Mel filterbank buffers), so it
never receives gradients.
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
    PITCH_SHIFT_BINS_PER_OCTAVE,
    PITCH_SHIFT_MODE,
    PITCH_SHIFT_N_FFT,
    PITCH_SHIFT_PROB,
    PITCH_SHIFT_REFERENCE_HZ,
    PITCH_SHIFT_SEMITONES,
    SAMPLE_RATE,
    SILENCE_CLASS_INDEX,
    TARGET_SAMPLES,
    TIME_MASK_PARAM,
    TIME_SHIFT_PROB,
    TIME_SHIFT_RATIO,
)

# Division applied to the int16 cache. 32768 (2**15) maps the full int16 range
# back onto [-1, 1). The dataset quantizes with 32767, so the round-trip error
# stays below one int16 step (~3e-5) - far below the mel filterbank's resolution.
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

    Two augmentations, both chosen to be cheap enough for the per-batch hot path
    (measured on a 256 x 16000 cpu batch, vs a 80 ms mel+delta baseline):

    * **Time-Shift** - one native :func:`torch.roll` (3 ms) rather than a
      ``gather`` + zero-fill (24 ms). The wrap-around is left in place: over a
      fixed 1.0 s window its artifact is a few samples of cross-fade at the
      window edge, far cheaper than the index math it replaces.
    * **Pitch-Shift** - ``PITCH_SHIFT_MODE = "spectral"`` (default) shifts the
      STFT bins along the frequency axis and inverts (``stft -> roll -> istft``,
      90 ms total). It is ~180x cheaper than the
      :func:`torchaudio.functional.pitch_shift` phase vocoder, whose cost is
      dominated by an internal ``AF.resample`` (~16.4 s per 256-clip batch).
      ``PITCH_SHIFT_MODE = "vocoder"`` restores that higher-quality path, and
      ``"off"`` disables pitch shifting entirely.

    Both shifts are drawn from a small discrete set, so clips sharing a value go
    through a single batched call: the loop count is bounded by the number of
    distinct values (2 for ``PITCH_SHIFT_SEMITONES = (-1, 1)``), never by the
    batch size.
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

        # Semitone -> STFT bin is a fixed, frequency-anchored mapping: resolving
        # it once here keeps it out of the per-batch hot path.
        self.pitch_bin_shifts = tuple(
            self._semitone_to_bins(semitone)
            for semitone in self.pitch_shift_semitones
        )
        # Non-persistent (absent from state_dict, so checkpointing and the ONNX
        # export are untouched) but still follows .to(device) with the module.
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

        Gating is per clip (``p_pitch``), and a clip draws one of
        ``pitch_shift_semitones``; clips that drew the same value then share a
        single batched call, so the number of calls equals the number of distinct
        values (2 with ``(-1, 1)``) whatever the batch size. What the call itself
        does depends on ``pitch_mode`` - see :meth:`_spectral_shift` and
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
        1 kHz but ~12 Hz at 200 Hz); a single fixed bin offset cannot follow
        that, so the anchor is made explicit and reproducible: one semitone at the
        reference frequency is ``ref * (2 ** (1/12) - 1)`` Hz, compared against
        the bin width ``sample_rate / pitch_n_fft``. Never returns 0 so a nonzero
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


class AudioToThreeChannelMel(nn.Module):
    """Raw int16 audio -> GPU Log-Mel + Delta + Delta-Delta, conditionally augmented.

    Input : ``[Batch, 1, T]`` (``torch.int16``, or ``[Batch, 1, T]`` float32) and,
             optionally, the batch ``targets`` used to shield ``_silence_`` clips.
    Output: ``[Batch, 3, n_mels, time_frames]`` (``torch.float32``).

    Every augmentation (waveform pitch/time shift *and* SpecAugment) is applied
    only in ``train()`` mode; call ``.eval()`` for the deterministic features used
    by validation, export and inference.
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
    ):
        super().__init__()
        self.augment = augment
        self.waveform_augment = waveform_augment
        self.silence_index = silence_index
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

        # Waveform-level augmentation (pitch + time shift) runs first, on the
        # normalized [-1, 1] audio. Its ``training`` flag mirrors this module's,
        # so ``eval()`` here also switches the waveform augmentation off.
        self.waveform_augmentation = WaveformAugmentation(
            sample_rate=sample_rate,
            p_pitch=pitch_shift_prob,
            pitch_shift_semitones=pitch_shift_semitones,
            pitch_mode=pitch_mode,
            p_time=time_shift_prob,
            time_shift_ratio=time_shift_ratio,
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

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor = None
    ) -> torch.Tensor:
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

        # 2. Waveform-level augmentation (train only, never on '_silence_').
        #    ``augment_mask`` is ``None`` when no labels were supplied, in which
        #    case every clip is eligible.
        augment_mask = self._augment_mask(targets)
        if self.augment and self.waveform_augment:
            x = self.waveform_augmentation(x, augment_mask)

        # MelSpectrogram consumes ``(..., time)``. Dropping the mono channel keeps
        # the output ``[Batch, n_mels, frames]`` so stacking yields ``[Batch, 3, ...]``.
        x = x.squeeze(1)

        # 3. Log-Mel spectrogram (channel 1).
        log_mel = torch.log(self.mel_spectrogram(x) + 1e-6)

        # 4./5. Delta and Delta-Delta along the time axis (channels 2 and 3).
        delta = AF.compute_deltas(log_mel)
        delta2 = AF.compute_deltas(delta)

        # 6. Stack channels: [Batch, 3, n_mels, time_frames].
        features = torch.stack((log_mel, delta, delta2), dim=1)

        # 7. Dynamic SpecAugment, directly on the features' device and gated so
        #    the '_silence_' rows stay exactly as they were extracted.
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

"""Audio I/O and feature extraction - single source of truth.

Both training (``src/dataset.py``), evaluation (``evaluate.py``) and inference
(``src/inference.py``) build their features through :class:`AudioFeatureExtractor`
and :func:`prepare_waveform` defined here. Nothing else may re-implement the
pipeline: duplicated preprocessing is how a model silently degrades between
training and serving.
"""

import logging
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import SAMPLE_RATE, TARGET_SAMPLES  # noqa: E402
from src.models import AudioFeatureExtractor  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torchaudio.load compatibility shim
# ---------------------------------------------------------------------------
# Recent torchaudio releases route ``load`` through TorchCodec, which is not
# installed everywhere; and clip formats produced by browsers (WebM/Opus) are
# not readable by libsndfile either. The wrapper below keeps the native loader
# when it works and falls back to soundfile otherwise, so that dataset reading
# and file inference keep working in both situations.
def _install_load_fallback() -> None:
    if getattr(torchaudio, "_commandsense_load_patch", False):
        return

    native_load = torchaudio.load

    def _safe_load(filepath, *args, **kwargs):
        try:
            return native_load(filepath, *args, **kwargs)
        except Exception as exc:  # TorchCodec missing, unsupported container, ...
            logger.debug("torchaudio.load failed on %s (%s); using soundfile.", filepath, exc)

        data, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
        # soundfile returns [frames, channels]; torchaudio expects [channels, frames].
        tensor = torch.from_numpy(data.T.copy())
        return tensor, sample_rate

    torchaudio.load = _safe_load
    torchaudio._commandsense_load_patch = True


_install_load_fallback()


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def _load_with_pyav(filepath: str):
    """Decode containers libsndfile cannot open (WebM/Opus from a browser)."""
    import av  # optional dependency, imported lazily

    with av.open(filepath) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE
        )
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(torch.from_numpy(resampled.to_ndarray().reshape(-1).copy()))
        for resampled in resampler.resample(None):  # flush
            chunks.append(torch.from_numpy(resampled.to_ndarray().reshape(-1).copy()))

    if not chunks:
        raise RuntimeError(f"PyAV decoded no audio samples from '{filepath}'.")

    return torch.cat(chunks).unsqueeze(0), SAMPLE_RATE


def load_audio_file(filepath: str):
    """Return ``(waveform, sample_rate)`` with waveform shaped ``[channels, time]``."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate
    except Exception as exc:
        logger.debug("torchaudio/soundfile could not read %s (%s); trying PyAV.", filepath, exc)

    try:
        return _load_with_pyav(filepath)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not decode '{filepath}'. Install 'av' (pip install av) for "
            "browser-recorded formats such as WebM/Opus."
        ) from exc


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to 16 kHz, downmix to mono and fit to exactly 1.0 s.

    Returns a ``[1, TARGET_SAMPLES]`` tensor.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    current = waveform.shape[1]
    if current < TARGET_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_SAMPLES - current))
    elif current > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]

    return waveform.contiguous()


def extract_features(waveform: torch.Tensor, device=None, extractor=None) -> torch.Tensor:
    """Build the 3-channel feature tensor ``[1, 3, N_MELS, frames]``.

    ``waveform`` must already be shaped ``[1, TARGET_SAMPLES]`` at 16 kHz.
    """
    extractor = extractor if extractor is not None else AudioFeatureExtractor()
    if device is not None:
        extractor = extractor.to(device)
    extractor.eval()

    waveform = waveform if device is None else waveform.to(device)
    with torch.no_grad():
        return extractor(waveform)


def load_and_preprocess_audio(audio_path: str, device=None, extractor=None) -> torch.Tensor:
    """Load a file and return its 3-channel feature tensor."""
    waveform, sample_rate = load_audio_file(audio_path)
    waveform = prepare_waveform(waveform, sample_rate)
    return extract_features(waveform, device=device, extractor=extractor)

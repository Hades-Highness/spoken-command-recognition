"""Raw-audio dataset for the CommandSense v3.0 memory-optimized pipeline.

The v3.0 front-end moves spectral feature extraction to the GPU
(``src/transforms.py``), so the dataset no longer stores 3-channel spectrograms
in RAM. Instead every clip is standardized at load time and cached as
``torch.int16``:

* stereo is downmixed to mono (``[1, T]``),
* audio is resampled to 16 kHz when necessary,
* clips are zero-padded or cropped to exactly 1.0 s (``TARGET_SAMPLES``).

16000 samples * 2 bytes = 32 KB per clip, so the whole training split stays well
under 4 GB - roughly a quarter of the previous float32 spectrogram cache. The
pad/crop policy itself still lives in :func:`src.utils.prepare_waveform`, so
training and serving cannot drift apart.

``.wav`` and ``.flac`` are handled seamlessly through
:func:`src.utils.load_audio_file` (a ``torchaudio`` wrapper with a soundfile /
PyAV fallback), so container support is identical for the Speech Commands and
LibriSpeech sources.

``return_audio=False`` preserves the legacy contract used by ``evaluate.py``: the
dataset then serves the pre-computed ``[3, n_mels, frames]`` features produced by
:class:`src.models.AudioFeatureExtractor`. Those features are derived from the
same int16 cache, so evaluation and training see identical inputs.

Dataset *provisioning* is centralized here too. Before anything is indexed or
cached, :func:`ensure_speech_commands` and :func:`ensure_librispeech` make sure
the raw ``.tar.gz`` archives have been downloaded (with ``aria2c`` when it is on
``PATH``, otherwise ``urllib``) and extracted under ``./data/``. The helpers check
their target directory first, so every entry point is idempotent and no caller
(``train.py``, ``evaluate.py``) has to know where the bytes come from.
"""

import gc
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from urllib.parse import urlparse

import torch
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import DATA_DIR, NUM_CLASSES, TARGET_SAMPLES  # noqa: E402
from src.models import AudioFeatureExtractor  # noqa: E402
from src.utils import load_audio_file, prepare_waveform  # noqa: E402

VALID_SUBSETS = ("training", "validation", "testing")

# ---------------------------------------------------------------------------
# Centralized dataset provisioning
# ---------------------------------------------------------------------------
# Upstream archives. Speech Commands is a single tarball; LibriSpeech ships one
# archive per subset (dev-clean, train-clean-100, ...).
SPEECH_COMMANDS_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
LIBRISPEECH_URL_TEMPLATE = "https://www.openslr.org/resources/12/{subset}.tar.gz"

# Extraction roots. Both archives already carry their own top-level directory, so
# extracting Speech Commands under ``./data/SpeechCommands`` and LibriSpeech under
# ``./data`` reproduces exactly the layout torchaudio expects with download=False:
#   data/SpeechCommands/speech_commands_v0.02   (SPEECHCOMMANDS)
#   data/LibriSpeech/<subset>                   (LIBRISPEECH)
SPEECH_COMMANDS_ROOT = os.path.join(str(DATA_DIR), "SpeechCommands")
SPEECH_COMMANDS_DIR = os.path.join(SPEECH_COMMANDS_ROOT, "speech_commands_v0.02")
LIBRISPEECH_ROOT = os.path.join(str(DATA_DIR), "LibriSpeech")

# LibriSpeech subset backing the ``_unknown_`` rejection samples for each Speech
# Commands split: training uses the large clean-100 set, validation/testing reuse
# the small dev-clean set.
LIBRISPEECH_SUBSETS = ("dev-clean", "train-clean-100")
LIBRISPEECH_SPLIT_BY_SUBSET = {
    "training": "train-clean-100",
    "validation": "dev-clean",
    "testing": "dev-clean",
}

# Containers the loader understands. Both are decoded natively by
# torchaudio/soundfile; keeping the set explicit lets directory scans stay in
# sync with the background-noise and LibriSpeech sources.
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac")

# int16 quantization constants. The GPU front-end divides by INT16_NORMALIZER
# (see ``src/transforms.py``) to restore the [-1, 1] scale.
INT16_SCALE = 32767.0
INT16_NORMALIZER = 32768.0


def _download_with_urllib(url: str, destination: str) -> None:
    """Single-connection fallback used when ``aria2c`` is unavailable.

    Standard-library :func:`urllib.request.urlretrieve` with a ``tqdm`` progress
    bar so the fallback is not noticeably worse to watch than ``aria2c``.

    The transfer is retried once over an unverified TLS context when - and only
    when - the failure is a certificate validation error.
    ``download.tensorflow.org`` intermittently serves a chain the local trust
    store cannot verify, so the retry keeps dataset provisioning usable instead
    of aborting on a known, transient CDN issue. Any other
    :class:`urllib.error.URLError` is re-raised untouched, so genuine network
    problems are never masked.
    """
    # Imported locally: ``aria2c`` is the primary path and most runs never reach
    # this fallback, so the SSL plumbing stays out of the module import cost.
    import ssl
    import urllib.error

    progress = {"bar": None}

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if progress["bar"] is None:
            total = total if total and total > 0 else None
            progress["bar"] = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(destination),
            )
        progress["bar"].update(block_size)

    try:
        try:
            urllib.request.urlretrieve(url, destination, reporthook=_hook)
        except urllib.error.URLError as exc:
            # Only a TLS trust failure triggers the retry; a DNS error, timeout
            # or HTTP status error is re-raised so real network problems stay
            # visible instead of being silently swallowed.
            reason = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" not in reason and "ssl.c" not in reason:
                raise

            print(
                "[!] SSL certificate verification failed for "
                f"{urlparse(url).netloc} (known intermittent CDN issue):\n"
                f"    {reason}\n"
                "[!] Retrying once with an unverified SSL context."
            )
            # A certificate error aborts the connection before any block is read,
            # so the bar is normally still unset; close it defensively anyway so
            # the retry starts from a clean progress bar.
            if progress["bar"] is not None:
                progress["bar"].close()
                progress["bar"] = None

            # Unverified context + globally installed opener, as a last resort
            # for this known CDN certificate problem.
            context = ssl._create_unverified_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=context)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, destination, reporthook=_hook)
            print("[+] Download succeeded using the unverified SSL context.")
    finally:
        if progress["bar"] is not None:
            progress["bar"].close()


def _extract_archive(archive_path: str, output_dir: str) -> None:
    """Extract a ``.tar.gz``/``.tgz`` archive into ``output_dir`` in place."""
    if not archive_path.endswith((".tar.gz", ".tgz")):
        return

    print(f"[*] Extracting {os.path.basename(archive_path)} -> {output_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            # Python >= 3.12 requires an explicit extraction filter (and warns
            # otherwise); ``data`` is the safe default for trusted datasets.
            tar.extractall(output_dir, filter="data")
        except TypeError:
            # Older interpreters do not know the ``filter`` keyword.
            tar.extractall(output_dir)


def _download_archive(url: str, output_dir: str) -> str:
    """Download ``url`` into ``output_dir`` and extract the ``.tar.gz`` there.

    ``aria2c -x 16 -s 16 -d <output_dir> <url>`` is preferred because the 16
    parallel connections saturate the link far better than a single stream; when
    ``aria2c`` is missing (or fails) the download degrades to
    :func:`_download_with_urllib`. The archive is kept on disk so reruns can skip
    the transfer, and the ``.tar.gz`` is extracted immediately afterwards.

    Returns the path of the downloaded archive.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    archive_path = os.path.join(output_dir, filename)

    if os.path.isfile(archive_path):
        print(f"[=] Archive already present, skipping download: {archive_path}")
    else:
        aria2c = shutil.which("aria2c")
        downloaded = False
        if aria2c:
            print(f"[*] aria2c ({aria2c}) downloading {url}")
            try:
                subprocess.run(
                    [aria2c, "-x", "16", "-s", "16", "-d", output_dir, url],
                    check=True,
                )
                downloaded = True
            except (subprocess.CalledProcessError, OSError) as exc:
                print(f"[!] aria2c failed ({exc}); falling back to urllib.")
        else:
            print("[*] aria2c not found on PATH; using the urllib fallback.")

        if not downloaded:
            print(f"[*] Downloading {url} -> {archive_path}")
            _download_with_urllib(url, archive_path)

    _extract_archive(archive_path, output_dir)
    return archive_path


def ensure_speech_commands() -> str:
    """Return ``data/SpeechCommands/speech_commands_v0.02``, fetching it if absent.

    Idempotent: checks the target directory before touching the network, so a
    second call is a no-op.
    """
    if os.path.isdir(SPEECH_COMMANDS_DIR):
        print(f"[=] Speech Commands already present: {SPEECH_COMMANDS_DIR}")
        return SPEECH_COMMANDS_DIR

    print("[*] Speech Commands v0.02 not found; downloading (~2.3 GB)...")
    _download_archive(SPEECH_COMMANDS_URL, SPEECH_COMMANDS_ROOT)

    if not os.path.isdir(SPEECH_COMMANDS_DIR):
        raise RuntimeError(
            f"Speech Commands extraction did not produce '{SPEECH_COMMANDS_DIR}'. "
            "Check the archive and the ./data/ tree before retrying."
        )
    return SPEECH_COMMANDS_DIR


def ensure_librispeech(subset: str = "train-clean-100") -> str:
    """Return ``data/LibriSpeech/<subset>``, fetching the split if absent.

    Idempotent: checks the target directory before touching the network. The
    archive is downloaded into ``./data`` and extracted there, which yields the
    ``data/LibriSpeech/<subset>`` path torchaudio's ``LIBRISPEECH`` walks.
    """
    if subset not in LIBRISPEECH_SUBSETS:
        raise ValueError(
            f"Unsupported LibriSpeech subset '{subset}'; expected one of "
            f"{LIBRISPEECH_SUBSETS}."
        )

    target = os.path.join(LIBRISPEECH_ROOT, subset)
    if os.path.isdir(target):
        print(f"[=] LibriSpeech '{subset}' already present: {target}")
        return target

    print(f"[*] LibriSpeech '{subset}' not found; downloading into ./data/...")
    _download_archive(LIBRISPEECH_URL_TEMPLATE.format(subset=subset), str(DATA_DIR))

    if not os.path.isdir(target):
        raise RuntimeError(
            f"LibriSpeech extraction did not produce '{target}'. Check the "
            "archive and the ./data/ tree before retrying."
        )
    return target


def quantize_int16(audio: torch.Tensor) -> torch.Tensor:
    """Clamp a float waveform to [-1, 1] and round it to ``torch.int16``."""
    return torch.round(audio.clamp(-1.0, 1.0) * INT16_SCALE).to(torch.int16)


def waveform_to_int16(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Standardize ``(waveform, sample_rate)`` and quantize it to int16.

    ``prepare_waveform`` owns the downmix / resample / pad-crop policy, so this
    function only adds the storage dtype conversion. Returns ``[1, TARGET_SAMPLES]``.
    """
    return quantize_int16(prepare_waveform(waveform, sample_rate))


def list_audio_files(directory: str) -> list:
    """Sorted ``.wav``/``.flac`` paths in ``directory`` (non-recursive)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)
    )


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v0.02 extended to 37 classes (35 + 2 rejection).

    ``_silence_`` is built from the ``_background_noise_`` recordings and
    ``_unknown_`` from real out-of-vocabulary LibriSpeech speech, exactly as in
    the v2.x releases. Both are standardized and cached as int16 like every other
    clip - no synthetic noise fallback, which would make the rejection class
    meaningless.
    """

    def __init__(
        self,
        subset: str = "training",
        cache_in_ram: bool = True,
        return_audio: bool = True,
        split: str = None,
    ):
        super().__init__()
        # ``split`` is an accepted alias for ``subset`` (torchaudio's own naming);
        # callers may use either keyword.
        if split is not None:
            subset = split
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got '{subset}'.")

        self.subset = subset
        self.cache_in_ram = cache_in_ram
        # v3.0 trains on raw int16 audio; evaluate.py opts into the legacy
        # pre-computed feature contract with return_audio=False.
        self.return_audio = return_audio

        # Centralized, idempotent provisioning. Both datasets are downloaded and
        # extracted under ./data/ *before* indexing or caching, so no split can be
        # built from a half-present tree.
        self.librispeech_subset = LIBRISPEECH_SPLIT_BY_SUBSET[subset]
        ensure_speech_commands()
        ensure_librispeech(self.librispeech_subset)

        print(f"[*] Indexing '{subset}' split from disk...")
        self.raw_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(DATA_DIR),
            url="speech_commands_v0.02",
            download=False,
            subset=subset,
        )

        # Single decode pass: collect labels once and reuse them for both the
        # class list and the target tensor.
        raw_labels = [sample[2] for sample in self.raw_dataset]
        self.base_labels = sorted(set(raw_labels))

        # Extended 37 labels list
        self.labels = sorted(self.base_labels + ["_silence_", "_unknown_"])

        if len(self.labels) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} classes in subset '{subset}' but found "
                f"{len(self.labels)}. Check NUM_CLASSES in configs/config.py."
            )

        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # Extra samples buffer for _silence_ and _unknown_ classes only, already
        # standardized to int16 [1, TARGET_SAMPLES].
        self.extra_samples = []
        self._generate_extra_samples()

        self.num_raw = len(self.raw_dataset)
        self.num_total = self.num_raw + len(self.extra_samples)

        self.targets = torch.tensor(
            [self.label_to_idx[label] for label in raw_labels]
            + [self.label_to_idx[label] for _, label in self.extra_samples],
            dtype=torch.long,
        )

        # int16 audio buffer [num_total, 1, TARGET_SAMPLES] (v3.0). The feature
        # cache [num_total, 3, n_mels, frames] is only built for the legacy path.
        self.audio = None
        self.cached_features = None
        self.feature_extractor = AudioFeatureExtractor()

        if self.cache_in_ram:
            if self.return_audio:
                self._cache_audio_in_ram()
            else:
                self._cache_features_in_ram()

    # ------------------------------------------------------------------ caching
    def _cache_audio_in_ram(self):
        """Preload every clip into a single int16 tensor living in system RAM."""
        gigabytes = self.num_total * TARGET_SAMPLES * 2 / 1e9
        print(
            f"[*] Preloading {self.num_total} clips as int16 audio "
            f"(~{gigabytes:.2f} GB)..."
        )
        self.audio = torch.empty(self.num_total, 1, TARGET_SAMPLES, dtype=torch.int16)

        for idx in tqdm(
            range(self.num_raw),
            desc=f"Loading {self.subset} set",
            unit="files",
            mininterval=0.5,
        ):
            waveform, sample_rate, _ = self._get_raw_audio(idx)
            self.audio[idx] = waveform_to_int16(waveform, sample_rate)

        for offset, (sample, _) in enumerate(self.extra_samples):
            self.audio[self.num_raw + offset] = sample

        # The int16 buffer now owns every clip: drop the per-sample list.
        self.extra_samples = []
        gc.collect()

        print(f"[+] {self.num_total} clips cached in RAM as int16.\n")

    def _cache_features_in_ram(self):
        """Legacy path (``return_audio=False``): pre-compute 3-channel features."""
        print(f"[*] Caching {self.num_total} samples to RAM as 3-channel features...")
        self.cached_features = []
        for idx in tqdm(
            range(self.num_total),
            desc=f"Caching {self.subset} set",
            unit="files",
            mininterval=0.5,
        ):
            features = self._features_from_audio(self._standardized_audio(idx))
            self.cached_features.append(features.squeeze(0))

        del self.extra_samples
        self.extra_samples = []
        gc.collect()

        print(f"[+] {len(self.cached_features)} samples loaded into RAM.\n")

    # -------------------------------------------------- rejection class sources
    def _generate_extra_samples(self):
        """Generates _silence_ and _unknown_ samples using real audio sources."""
        num_extra_target = int(len(self.raw_dataset) * 0.05)  # ~5% of split size

        # 1. Inject _silence_ samples (split noise sources to prevent leakage)
        bg_dir = os.path.join(SPEECH_COMMANDS_DIR, "_background_noise_")
        if not os.path.exists(bg_dir):
            raise RuntimeError(f"[!] Background noise directory not found at '{bg_dir}'.")

        bg_files = list_audio_files(bg_dir)

        if len(bg_files) < 5:
            raise RuntimeError(
                f"Expected at least 5 background noise files in {bg_dir}, "
                f"found {len(bg_files)}."
            )

        # Partition background noise files: 4 files for training, rest for val/test
        if self.subset == "training":
            selected_bg = bg_files[:4]
        else:
            selected_bg = bg_files[4:]

        silence_count = 0
        for _ in range(num_extra_target):
            bg_file = random.choice(selected_bg)
            waveform, sample_rate = load_audio_file(bg_file)

            # Standardize to 16 kHz / mono / 1.0 s, then apply a random gain so
            # the silence class spans a range of quiet levels.
            chunk = prepare_waveform(waveform, sample_rate)
            chunk = chunk * random.uniform(0.1, 1.0)

            self.extra_samples.append((quantize_int16(chunk), "_silence_"))
            silence_count += 1

        # 2. Inject _unknown_ samples from LibriSpeech (real OOV speech)
        ls_subset = self.librispeech_subset
        try:
            librispeech_ds = torchaudio.datasets.LIBRISPEECH(
                root=str(DATA_DIR),
                url=ls_subset,
                download=False,
            )

            indices = list(range(len(librispeech_ds)))
            random.shuffle(indices)

            unknown_count = 0
            for idx in indices:
                if unknown_count >= num_extra_target:
                    break
                waveform, sample_rate, _, _, _, _ = librispeech_ds[idx]
                # prepare_waveform resamples / downmixes / crops the 1.0 s clip.
                self.extra_samples.append(
                    (waveform_to_int16(waveform, sample_rate), "_unknown_")
                )
                unknown_count += 1

        except Exception as e:
            raise RuntimeError(
                f"_unknown_ samples require LibriSpeech '{ls_subset}' but loading "
                f"failed: {e}. Refusing to fall back to synthetic noise, which "
                "would silently make the rejection class meaningless."
            ) from e

        print(
            f"[+] Extra samples generated for split '{self.subset}': "
            f"_silence_={silence_count}, _unknown_={unknown_count}"
        )

    # -------------------------------------------------------------- accessors
    def _get_raw_audio(self, idx: int):
        """Fetch ``(waveform, sample_rate, label)`` for a raw Speech Commands clip."""
        waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
        return waveform, sample_rate, label

    def _standardized_audio(self, idx: int) -> torch.Tensor:
        """Return the int16 clip ``[1, TARGET_SAMPLES]`` for ``idx``.

        Serves the RAM cache when present, otherwise standardizes on demand (and
        reuses the int16 ``_silence_`` / ``_unknown_`` buffer built at init).
        """
        if self.audio is not None:
            return self.audio[idx]
        if idx < self.num_raw:
            waveform, sample_rate, _ = self._get_raw_audio(idx)
            return waveform_to_int16(waveform, sample_rate)
        return self.extra_samples[idx - self.num_raw][0]

    def _features_from_audio(self, audio_int16: torch.Tensor) -> torch.Tensor:
        """Legacy 3-channel features: ``[1, 3, n_mels, frames]`` from int16 audio."""
        waveform = audio_int16.to(torch.float32) / INT16_NORMALIZER
        with torch.no_grad():
            return self.feature_extractor(waveform)

    def __len__(self) -> int:
        return self.num_total

    def __getitem__(self, idx: int):
        target = self.targets[idx]
        if self.return_audio:
            # v3.0: raw int16 [1, TARGET_SAMPLES]; features are built on the GPU.
            return self._standardized_audio(idx), target
        # Legacy contract for evaluate.py: pre-computed [3, n_mels, frames].
        if self.cached_features is not None:
            return self.cached_features[idx], target
        features = self._features_from_audio(self._standardized_audio(idx))
        return features.squeeze(0), target


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=False)
    audio, target = dataset[0]
    print(f"[+] {len(dataset)} samples, {len(dataset.labels)} classes")
    print(
        f"[+] audio dtype {audio.dtype}, shape {tuple(audio.shape)}, "
        f"first target {int(target)} ({dataset.labels[int(target)]})"
    )

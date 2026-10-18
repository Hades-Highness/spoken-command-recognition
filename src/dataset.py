"""Raw-audio dataset for the CommandSense v3.0 / v4.0 pipeline.

Clips are standardized to 16 kHz mono, exactly 1.0 s, and cached as
``torch.int16``; the 3-channel features are extracted on the GPU by
``src/transforms.py``. Dataset provisioning (download and extraction under
``./data/``) is centralized here and idempotent.

For v4.0 the training split additionally exposes ``noise_bank``: two small
``int16`` pools (training-side ``_background_noise_`` chunks and cached
LibriSpeech ``_unknown_`` clips) that ``AudioToThreeChannelMel`` promotes to
non-persistent GPU buffers once, before the first batch. Mixing therefore never
touches the disk and never runs on the CPU, and no other split builds a bank.
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

from configs.config import (  # noqa: E402
    DATA_DIR,
    NOISE_BANK_AMBIENT_CLIPS,
    NOISE_BANK_SPEECH_CLIPS,
    NUM_CLASSES,
    SAMPLE_RATE,
    SEED,
    TARGET_SAMPLES,
)
from src.models import AudioFeatureExtractor  # noqa: E402
from src.utils import load_audio_file, prepare_waveform  # noqa: E402

VALID_SUBSETS = ("training", "validation", "testing")

SPEECH_COMMANDS_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
LIBRISPEECH_URL_TEMPLATE = "https://www.openslr.org/resources/12/{subset}.tar.gz"

# Both archives carry their own top-level directory, so extracting under these
# roots reproduces the layout torchaudio expects with download=False.
SPEECH_COMMANDS_ROOT = os.path.join(str(DATA_DIR), "SpeechCommands")
SPEECH_COMMANDS_DIR = os.path.join(SPEECH_COMMANDS_ROOT, "speech_commands_v0.02")
LIBRISPEECH_ROOT = os.path.join(str(DATA_DIR), "LibriSpeech")

LIBRISPEECH_SUBSETS = ("dev-clean", "train-clean-100")
LIBRISPEECH_SPLIT_BY_SUBSET = {
    "training": "train-clean-100",
    "validation": "dev-clean",
    "testing": "dev-clean",
}

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac")

# INT16_NORMALIZER is the divisor src/transforms.py uses to restore [-1, 1].
INT16_SCALE = 32767.0
INT16_NORMALIZER = 32768.0


def _download_with_urllib(url: str, destination: str) -> None:
    """Single-connection fallback used when ``aria2c`` is unavailable.

    Retried once over an unverified TLS context only for certificate-validation
    failures (download.tensorflow.org intermittently serves an unverifiable
    chain); any other ``URLError`` is re-raised.
    """
    # Imported locally to keep SSL plumbing out of the module import cost.
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
            # Only a TLS trust failure is retried; other errors are re-raised.
            reason = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" not in reason and "ssl.c" not in reason:
                raise

            print(
                "[!] SSL certificate verification failed for "
                f"{urlparse(url).netloc} (known intermittent CDN issue):\n"
                f"    {reason}\n"
                "[!] Retrying once with an unverified SSL context."
            )
            if progress["bar"] is not None:
                progress["bar"].close()
                progress["bar"] = None

            # Last resort for the known CDN certificate issue.
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

    ``aria2c`` is preferred for its parallel connections, with
    :func:`_download_with_urllib` as the fallback. Returns the archive path.
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
    """Return ``data/SpeechCommands/speech_commands_v0.02``, fetching it if absent."""
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
    """Return ``data/LibriSpeech/<subset>``, fetching the split if absent."""
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
    """Standardize ``(waveform, sample_rate)`` into an int16 ``[1, TARGET_SAMPLES]``."""
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


def background_noise_split():
    """Return ``(train_files, heldout_files)`` from ``_background_noise_``.

    The recordings are disjoint between the training split and the validation /
    testing splits, so neither the '_silence_' class nor the v4.0 noise mixer can
    leak held-out background noise into training.
    """
    bg_dir = os.path.join(SPEECH_COMMANDS_DIR, "_background_noise_")
    if not os.path.exists(bg_dir):
        raise RuntimeError(f"[!] Background noise directory not found at '{bg_dir}'.")

    bg_files = list_audio_files(bg_dir)
    if len(bg_files) < 5:
        raise RuntimeError(
            f"Expected at least 5 background noise files in {bg_dir}, "
            f"found {len(bg_files)}."
        )
    return bg_files[:4], bg_files[4:]


def _mono_16k(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to 16 kHz and downmix to mono, keeping the full duration."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def build_ambient_noise_bank(
    files, num_clips: int = NOISE_BANK_AMBIENT_CLIPS, seed: int = SEED
):
    """Slice background recordings into ``[N, 1, TARGET_SAMPLES]`` int16 clips.

    Runs once while the dataset is built, never inside the training loop. Each
    recording contributes the same quota of non-overlapping 1.0 s chunks, so a
    single long file cannot dominate the pool, and every chunk already has the
    cache layout the front-end expects (int16, mono, 16 kHz, fixed length).
    """
    if not files or num_clips <= 0:
        return None

    per_file = max(1, num_clips // len(files))
    chunks = []
    for path in files:
        waveform, sample_rate = load_audio_file(path)
        audio = _mono_16k(waveform, sample_rate)
        taken = 0
        for start in range(0, audio.shape[1] - TARGET_SAMPLES + 1, TARGET_SAMPLES):
            chunks.append(quantize_int16(audio[:, start:start + TARGET_SAMPLES]))
            taken += 1
            if taken >= per_file:
                break
        if len(chunks) >= num_clips:
            break

    if not chunks:
        return None

    bank = torch.cat(chunks, dim=0).unsqueeze(1)
    if bank.shape[0] > num_clips:
        keep = sorted(random.Random(seed).sample(range(bank.shape[0]), num_clips))
        bank = bank[keep]
    return bank.contiguous()


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v0.02 extended to 37 classes (35 + 2 rejection).

    ``_silence_`` comes from the ``_background_noise_`` recordings and
    ``_unknown_`` from real out-of-vocabulary LibriSpeech speech.
    """

    def __init__(
        self,
        subset: str = "training",
        cache_in_ram: bool = True,
        return_audio: bool = True,
        split: str = None,
    ):
        super().__init__()
        if split is not None:
            subset = split
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got '{subset}'.")

        self.subset = subset
        self.cache_in_ram = cache_in_ram
        # evaluate.py sets return_audio=False for the legacy pre-computed features.
        self.return_audio = return_audio

        # Provisioning runs before indexing so no split sees a half-present tree.
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

        raw_labels = [sample[2] for sample in self.raw_dataset]
        self.base_labels = sorted(set(raw_labels))
        self.labels = sorted(self.base_labels + ["_silence_", "_unknown_"])

        if len(self.labels) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} classes in subset '{subset}' but found "
                f"{len(self.labels)}. Check NUM_CLASSES in configs/config.py."
            )

        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        self.extra_samples = []
        self.unknown_extra_indices = []
        self._generate_extra_samples()

        self.num_raw = len(self.raw_dataset)
        self.num_total = self.num_raw + len(self.extra_samples)

        self.targets = torch.tensor(
            [self.label_to_idx[label] for label in raw_labels]
            + [self.label_to_idx[label] for _, label in self.extra_samples],
            dtype=torch.long,
        )

        self.audio = None
        self.cached_features = None
        self.feature_extractor = AudioFeatureExtractor()

        if self.cache_in_ram:
            if self.return_audio:
                self._cache_audio_in_ram()
            else:
                self._cache_features_in_ram()

        # v4.0 noise pools. Built after the audio cache so the LibriSpeech pool is
        # a slice of the clips already in RAM, not a second pass over the corpus.
        # Training-only by construction: validation and test splits stay clean,
        # which is what makes their metrics comparable with v3.x.
        self.noise_bank = None
        if self.return_audio and self.subset == "training":
            self.noise_bank = self._build_noise_bank()

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

        # The int16 buffer owns every clip now; drop the per-sample list.
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

    def _build_noise_bank(self):
        """Preload the v4.0 noise pools as int16 tensors in system RAM.

        Ambient clips are sliced from the training half of ``_background_noise_``
        (the same split the '_silence_' class draws from, so held-out recordings
        stay out of training), and the overlapping-speech pool is sampled from the
        LibriSpeech clips already cached as '_unknown_'. Both are copied once,
        here; the training loop only gathers from the GPU buffers built from them.
        """
        try:
            train_bg, _ = background_noise_split()
        except RuntimeError as exc:
            print(f"[!] Noise mixing disabled: {exc}")
            return None

        ambient = build_ambient_noise_bank(train_bg)
        speech = self._sample_unknown_audio(NOISE_BANK_SPEECH_CLIPS)
        if ambient is None and speech is None:
            print("[!] Noise mixing disabled: no ambient or speech noise clips.")
            return None

        ambient_clips = 0 if ambient is None else int(ambient.shape[0])
        speech_clips = 0 if speech is None else int(speech.shape[0])
        print(
            f"[+] Noise bank ready: ambient={ambient_clips} clips, "
            f"speech={speech_clips} clips (int16, RAM, training split only)"
        )
        return {"ambient": ambient, "speech": speech}

    def _sample_unknown_audio(self, num_clips: int):
        """Random ``[N, 1, TARGET_SAMPLES]`` int16 slice of cached LibriSpeech.

        Selecting before indexing keeps the copy at ``N`` clips instead of
        materializing every '_unknown_' row as a second tensor.
        """
        count = len(self.unknown_extra_indices)
        if count == 0 or num_clips <= 0:
            return None

        picked = random.Random(SEED).sample(range(count), min(num_clips, count))
        if self.audio is not None:
            rows = torch.tensor(
                [self.num_raw + self.unknown_extra_indices[i] for i in picked],
                dtype=torch.long,
            )
            return self.audio[rows].clone()

        return torch.stack(
            [self.extra_samples[self.unknown_extra_indices[i]][0] for i in picked]
        ).clone()

    def _generate_extra_samples(self):
        """Generates _silence_ and _unknown_ samples using real audio sources."""
        num_extra_target = int(len(self.raw_dataset) * 0.05)  # ~5% of split size

        # 1. _silence_ samples; the noise sources are split to prevent leakage.
        #    The same split feeds the v4.0 ambient pool, so the mixer cannot
        #    borrow a held-out recording either.
        train_bg, heldout_bg = background_noise_split()
        selected_bg = train_bg if self.subset == "training" else heldout_bg

        silence_count = 0
        for _ in range(num_extra_target):
            bg_file = random.choice(selected_bg)
            waveform, sample_rate = load_audio_file(bg_file)

            # Random gain so the silence class spans a range of quiet levels.
            chunk = prepare_waveform(waveform, sample_rate)
            chunk = chunk * random.uniform(0.1, 1.0)

            self.extra_samples.append((quantize_int16(chunk), "_silence_"))
            silence_count += 1

        # 2. _unknown_ samples from LibriSpeech (real OOV speech).
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
                self.extra_samples.append(
                    (waveform_to_int16(waveform, sample_rate), "_unknown_")
                )
                # Row of this clip inside ``extra_samples``: the v4.0 speech-noise
                # pool is sampled from exactly these LibriSpeech rows.
                self.unknown_extra_indices.append(len(self.extra_samples) - 1)
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

    def _get_raw_audio(self, idx: int):
        """Fetch ``(waveform, sample_rate, label)`` for a raw Speech Commands clip."""
        waveform, sample_rate, label, _, _ = self.raw_dataset[idx]
        return waveform, sample_rate, label

    def _standardized_audio(self, idx: int) -> torch.Tensor:
        """Return the int16 clip ``[1, TARGET_SAMPLES]`` for ``idx``."""
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
            return self._standardized_audio(idx), target
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

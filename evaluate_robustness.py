"""Noise-robustness evaluation of CommandSense on strictly held-out noise.

The clean ``testing`` split is re-scored at fixed target SNRs, mixed with a noise
bank that the v4.0 training run could never have touched:

* **ambient** - the ``_background_noise_`` recordings that
  :func:`src.dataset.background_noise_split` reserves for validation/testing
  (training only ever sees the first four), sliced into non-overlapping 1.0 s
  chunks by the same :func:`src.dataset.build_ambient_noise_bank` helper;
* **speech** - fresh 1.0 s chunks from LibriSpeech ``dev-clean`` and
  ``test-clean``, while the training noise bank is sampled from
  ``train-clean-100``.

``SpeechCommandsDataset.noise_bank`` is never read here (the evaluator asserts it
stays ``None``), so a robustness number cannot be inflated by a clip the model
memorised. The mixing itself is delegated to
:class:`src.transforms.WaveformNoiseMixer` through :class:`HeldOutSnrMixer`,
which keeps ``alpha = (rms(clean) / rms(noise)) * 10 ** (-SNR / 20)`` and the
peak limiter bit-identical to training and only pins the SNR draw to the level
being evaluated.

Two metrics are reported per condition, on the 35 command classes only:

* **Top-1 accuracy** - raw argmax over every command clip, i.e. the same
  threshold-independent number ``evaluate.py`` reports;
* **rejection rate** - share of those clips whose temperature-scaled top-1
  confidence falls below ``tau`` and is therefore served as ``_unknown_``.

Usage:
    python evaluate_robustness.py
    python evaluate_robustness.py --compare_v3
    python evaluate_robustness.py --snr 0 5 10 --max_samples 512
    python evaluate_robustness.py --download_speech_noise
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NOISE_BANK_AMBIENT_CLIPS,
    NOISE_BANK_SPEECH_CLIPS,
    NOISE_SPEECH_SOURCE_PROB,
    NUM_CLASSES,
    REJECTION_CLASSES,
    SAMPLE_RATE,
    SEED,
    TARGET_SAMPLES,
    checkpoint_dir,
    reports_dir,
)
from src import calibration  # noqa: E402
from src.dataset import (  # noqa: E402
    SpeechCommandsDataset,
    background_noise_split,
    build_ambient_noise_bank,
    ensure_librispeech,
    quantize_int16,
)
from src.models import CommandSense  # noqa: E402
from src.transforms import (  # noqa: E402
    INT16_NORMALIZER,
    AudioToThreeChannelMel,
    WaveformNoiseMixer,
)
import src.utils  # noqa: E402,F401  (installs the torchaudio.load fallback early)
from train import set_seed  # noqa: E402  (shared seeding helper, as in evaluate.py)

EVALUATION_SUBSET = "testing"                    # official clean, speaker-disjoint split
COMPARISON_VERSION = "v3.0"                      # robustness baseline of the comparison table
DEFAULT_SNR_LEVELS_DB = (5.0, 10.0, 15.0, 20.0)  # ascending, as reported

# Held-out bank sizes mirror the training bank so both are equally expressive;
# the sources, not the size, are what makes this evaluation leakage-free.
EVAL_AMBIENT_CLIPS = NOISE_BANK_AMBIENT_CLIPS
EVAL_SPEECH_CLIPS = NOISE_BANK_SPEECH_CLIPS
EVAL_SPEECH_SPLITS = ("dev-clean", "test-clean")  # never a training noise source
TRAINING_SPEECH_SPLITS = ("train-clean-100",)     # what SpeechCommandsDataset samples

REPORT_FILENAME = f"robustness_report_{MODEL_VERSION}.txt"


def resolve_checkpoint(version=MODEL_VERSION):
    """Return the weights of ``version``, or ``None`` when the release is absent.

    Same lookup order as ``evaluate.py`` / ``src/inference.py``: the versioned
    folder first, then the legacy flat layout.
    """
    candidates = [
        checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth",
        CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(candidate)
    return None


def verify_class_mapping(class_names):
    """Fail loudly when ``configs/labels.json`` disagrees with the split order.

    Mirrors ``evaluate.py``: a drifted mapping would silently make every
    reported accuracy meaningless.
    """
    if not os.path.exists(LABELS_PATH):
        print(f"[!] {LABELS_PATH} not found: index order is unverified.")
        return
    with open(LABELS_PATH, "r", encoding="utf-8") as handle:
        declared_labels = json.load(handle)
    if declared_labels != class_names:
        raise RuntimeError(
            f"Class mapping mismatch: '{LABELS_PATH}' differs from the split's "
            "label order, so every predicted index would be mislabelled."
        )
    print(f"[+] Class mapping verified against {LABELS_PATH} ({len(class_names)} classes)")


def load_calibration_values(version=MODEL_VERSION):
    """Return ``(temperature, threshold, source)`` for ``version``.

    ``src/calibration.py`` owns the artifact and the fallback policy: a missing
    or unusable payload degrades to ``T = 1.0`` / ``tau = 0.0`` exactly as the
    served inferencer does, so the evaluator never measures a different
    operating point than the one that ships.
    """
    path = calibration.resolve_calibration_file(version)
    payload = calibration.load_calibration(path) if path is not None else {}

    temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
    threshold = payload.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    try:
        temperature = float(temperature)
        threshold = float(threshold)
    except (TypeError, ValueError):
        temperature, threshold = DEFAULT_TEMPERATURE, DEFAULT_CONFIDENCE_THRESHOLD

    if not (np.isfinite(temperature) and temperature > 0.0):
        print(f"[!] Unusable temperature {temperature!r} for {version}; using {DEFAULT_TEMPERATURE}.")
        temperature = DEFAULT_TEMPERATURE
    if not (np.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        print(f"[!] Unusable threshold {threshold!r} for {version}; using {DEFAULT_CONFIDENCE_THRESHOLD}.")
        threshold = DEFAULT_CONFIDENCE_THRESHOLD

    if path is None:
        print(
            f"[!] No calibration payload for {version}: serving raw logits with "
            f"T={temperature:.4f}, tau={threshold:.2f} (no low-confidence rejection)."
        )
        return temperature, threshold, None

    print(f"[+] Calibration {version}: T={temperature:.6f}, tau={threshold:.2f} <- {path}")
    return temperature, threshold, str(path)


def _librispeech_split_dir(split: str) -> str:
    """Directory ``torchaudio.datasets.LIBRISPEECH`` expects for ``split``."""
    return os.path.join(str(DATA_DIR), "LibriSpeech", split)


def _to_mono_16k(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to 16 kHz and downmix to mono, keeping the whole utterance.

    ``src.dataset._mono_16k`` is private; this repeats its two operations
    locally because the evaluator must not depend on a private helper.
    Cropping happens later, at a random offset.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def _crop_noise_chunk(audio: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Cut one 1.0 s ``int16`` chunk at a random offset of a longer recording.

    A random offset (instead of the first second) spreads the pool over the
    whole corpus, so a handful of utterances cannot determine the bank.
    """
    length = audio.shape[-1]
    if length <= TARGET_SAMPLES:
        chunk = F.pad(audio, (0, TARGET_SAMPLES - length))
    else:
        offset = rng.randrange(0, length - TARGET_SAMPLES + 1)
        chunk = audio[:, offset:offset + TARGET_SAMPLES]
    return quantize_int16(chunk)


def build_heldout_speech_bank(num_clips=EVAL_SPEECH_CLIPS, seed=SEED, allow_download=False):
    """Sample 1.0 s ``int16`` speech-noise chunks from held-out LibriSpeech splits.

    One chunk per utterance is drawn at a random offset, so a handful of
    utterances cannot dominate the pool. ``dev-clean`` and ``test-clean`` are
    used because :data:`TRAINING_SPEECH_SPLITS` is what the v4.0 training noise
    bank samples, so no evaluated noise clip was ever mixed into a training
    batch. Missing splits are skipped with a warning unless ``allow_download``
    asks :func:`src.dataset.ensure_librispeech` to provision them; at least one
    usable split is required.
    """
    for split in EVAL_SPEECH_SPLITS:
        if split in TRAINING_SPEECH_SPLITS:
            raise RuntimeError(
                f"'{split}' feeds the training noise bank; refusing to evaluate with it."
            )

    available, missing = [], []
    for split in EVAL_SPEECH_SPLITS:
        if os.path.isdir(_librispeech_split_dir(split)):
            available.append(split)
        elif allow_download:
            print(f"[*] LibriSpeech '{split}' is absent; provisioning it into ./data/ ...")
            ensure_librispeech(split)
            available.append(split)
        else:
            missing.append(split)

    if missing:
        print(
            f"[!] Held-out speech-noise split(s) absent: {', '.join(missing)}. "
            "Re-run with --download_speech_noise to include them."
        )
    if not available:
        raise RuntimeError(
            "No held-out LibriSpeech split is available, so the speech half of the "
            f"evaluation noise bank cannot be built (expected {EVAL_SPEECH_SPLITS} "
            f"under '{os.path.join(str(DATA_DIR), 'LibriSpeech')}')."
        )

    per_split = max(1, num_clips // len(available))
    chunks, counts = [], {}
    for split in available:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(DATA_DIR), url=split, download=False
        )
        # A string seed keeps each split independent and reproducible across runs.
        rng = random.Random(f"{seed}:{split}")
        picked = sorted(rng.sample(range(len(dataset)), min(per_split, len(dataset))))
        taken = 0
        for index in tqdm(
            picked,
            desc=f"Sampling speech noise ({split})",
            unit="clips",
            mininterval=0.5,
        ):
            waveform, sample_rate, *_ = dataset[index]
            chunks.append(_crop_noise_chunk(_to_mono_16k(waveform, sample_rate), rng))
            taken += 1
        counts[split] = taken
        print(f"[+] Speech noise from '{split}': {taken} chunks of 1.0 s")

    return torch.stack(chunks).contiguous(), counts, missing


def build_eval_noise_bank(
    num_ambient=EVAL_AMBIENT_CLIPS,
    num_speech=EVAL_SPEECH_CLIPS,
    seed=SEED,
    allow_download=False,
):
    """Build the held-out ``eval_noise_bank`` and describe where it came from.

    The ambient half is built by :func:`src.dataset.build_ambient_noise_bank`
    from the held-out background recordings only, and the two file lists are
    checked to be disjoint before a single chunk is cut: a shared recording
    would put training noise into the evaluation bank.
    """
    train_files, heldout_files = background_noise_split()
    overlap = set(train_files) & set(heldout_files)
    if overlap:
        raise RuntimeError(
            "Background-noise split is not disjoint; refusing to evaluate: "
            + ", ".join(sorted(os.path.basename(path) for path in overlap))
        )

    ambient = build_ambient_noise_bank(heldout_files, num_clips=num_ambient, seed=seed)
    if ambient is None:
        raise RuntimeError(
            "Could not slice any 1.0 s ambient chunk from the held-out "
            f"_background_noise_ recordings ({', '.join(heldout_files)})."
        )

    speech, speech_counts, speech_missing = build_heldout_speech_bank(
        num_clips=num_speech, seed=seed, allow_download=allow_download
    )

    return {
        "ambient": ambient,
        "speech": speech,
        "ambient_files": [os.path.basename(path) for path in heldout_files],
        "ambient_excluded": [os.path.basename(path) for path in train_files],
        "ambient_chunks": int(ambient.shape[0]),
        "speech_chunks": int(speech.shape[0]) if speech is not None else 0,
        "speech_splits": speech_counts,
        "speech_absent": speech_missing,
    }


class HeldOutSnrMixer(WaveformNoiseMixer):
    """``WaveformNoiseMixer`` pinned to one SNR and driven outside training mode.

    The training mixer draws one SNR per row from a uniform range and is an
    exact no-op in ``eval()``; an evaluation needs the opposite of both. Rather
    than re-implement the arithmetic, this subclass reuses the parent's
    ``_mix_rows`` and ``_limit_peak`` unchanged, so the gain, the RMS
    equalisation and the peak limiter are the ones the model was trained with,
    and only the SNR draw is forced to ``target_snr_db``.

    A random clip is still drawn from the corresponding pool for every row,
    which is what makes each condition a distribution rather than one sample.
    """

    def __init__(
        self,
        ambient_noise=None,
        speech_noise=None,
        snr_db=10.0,
        speech_source_prob=NOISE_SPEECH_SOURCE_PROB,
        eps=1e-8,
    ):
        snr_db = float(snr_db)
        super().__init__(
            ambient_noise=ambient_noise,
            speech_noise=speech_noise,
            p=0.0,  # the training probability gate is bypassed by mix()
            ambient_snr_db=(snr_db, snr_db),
            speech_snr_db=(snr_db, snr_db),
            speech_source_prob=speech_source_prob,
            silence_index=None,  # only the 35 command classes are mixed
            eps=eps,
        )
        self.target_snr_db = snr_db

    @property
    def snr_range(self):
        """Degenerate ``(min, max)`` range that makes the parent draw exact."""
        return (self.target_snr_db, self.target_snr_db)

    @torch.no_grad()
    def mix(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mix every row at exactly ``target_snr_db``, without an eligibility gate.

        ``waveform`` is ``[Batch, T]`` or ``[Batch, 1, T]`` float audio in
        ``[-1, 1]``; the result has the same shape, dtype and device.
        """
        squeeze = waveform.ndim == 3 and waveform.shape[1] == 1
        audio = waveform.squeeze(1) if squeeze else waveform
        if audio.ndim != 2:
            raise ValueError(
                "HeldOutSnrMixer expects [Batch, T] or [Batch, 1, T], "
                f"got {tuple(waveform.shape)}"
            )
        if not self.has_noise:
            raise RuntimeError("HeldOutSnrMixer needs at least one noise pool.")

        mixed = audio.clone()
        if self.ambient_noise is not None and self.speech_noise is not None:
            draw = torch.rand(audio.shape[0], device=audio.device)
            use_speech = draw < self.speech_source_prob
            ambient_rows = torch.nonzero(~use_speech, as_tuple=False).flatten()
            speech_rows = torch.nonzero(use_speech, as_tuple=False).flatten()
        elif self.speech_noise is not None:
            ambient_rows = audio.new_empty(0, dtype=torch.long)
            speech_rows = torch.arange(audio.shape[0], device=audio.device)
        else:
            ambient_rows = torch.arange(audio.shape[0], device=audio.device)
            speech_rows = audio.new_empty(0, dtype=torch.long)

        if ambient_rows.numel():
            mixed[ambient_rows] = self._mix_rows(
                audio[ambient_rows], self.ambient_noise, self.snr_range
            )
        if speech_rows.numel():
            mixed[speech_rows] = self._mix_rows(
                audio[speech_rows], self.speech_noise, self.snr_range
            )
        return mixed.unsqueeze(1) if squeeze else mixed


EXPECTED_COMMAND_CLIPS = 11005  # 35 command classes of the official 'testing' split


def load_command_split(max_samples=0, seed=SEED):
    """Load the clean ``testing`` split and keep only the 35 command classes.

    Returns ``(audio, targets, rows, class_names, total_clips)`` where ``audio``
    is the dataset's shared ``int16`` cache, ``rows`` selects the command clips
    and ``total_clips`` is the size of the whole split including the two
    rejection classes. The rejection classes are dropped because a robustness
    curve is about commands the model should still recognise, not about how often
    it abstains on silence or on out-of-vocabulary speech.

    ``dataset.noise_bank`` is asserted to be ``None``: the clean split carries no
    training noise, and this evaluator never asks for it.
    """
    dataset = SpeechCommandsDataset(
        subset=EVALUATION_SUBSET, cache_in_ram=True, return_audio=True
    )
    if dataset.audio is None:
        raise RuntimeError(
            "The audio cache is empty; evaluate_robustness.py needs cache_in_ram=True."
        )
    if getattr(dataset, "noise_bank", None) is not None:
        raise RuntimeError(
            f"'{EVALUATION_SUBSET}' unexpectedly exposes a training noise bank; "
            "refusing to evaluate to avoid leakage."
        )

    class_names = list(dataset.labels)
    verify_class_mapping(class_names)

    rejection_indices = {
        class_names.index(label) for label in REJECTION_CLASSES if label in class_names
    }
    targets_all = dataset.targets.cpu().tolist()
    rows = np.asarray(
        [index for index, target in enumerate(targets_all) if target not in rejection_indices],
        dtype=np.int64,
    )

    print(
        f"[+] '{EVALUATION_SUBSET}': {len(targets_all)} clips total, "
        f"{rows.size} command clips across {len(class_names) - len(rejection_indices)} classes"
    )
    if rows.size != EXPECTED_COMMAND_CLIPS:
        print(
            f"[!] Expected {EXPECTED_COMMAND_CLIPS} command clips, found {rows.size}. "
            "Numbers below are only comparable with earlier reports if this matches."
        )

    if 0 < max_samples < rows.size:
        picked = sorted(random.Random(seed).sample(rows.tolist(), max_samples))
        rows = np.asarray(picked, dtype=np.int64)
        print(f"[!] Subset mode: scoring {rows.size} clips only (not a release figure).")

    targets = np.asarray([targets_all[index] for index in rows], dtype=np.int64)
    return dataset.audio, targets, rows, class_names, dataset.num_total


def summarize_condition(probabilities, targets, threshold):
    """Turn one condition's probabilities into the reported figures.

    ``accuracy`` is the threshold-independent top-1 number ``evaluate.py``
    reports; the remaining fields describe the served operating point at
    ``threshold`` (``tau``), so a drop in accuracy can be told apart from a drop
    in usable coverage.
    """
    probabilities = calibration.as_numpy(probabilities).astype(np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    decision = calibration.threshold_metrics(probabilities, targets, threshold)
    confidences, predictions = calibration.top_label_confidence(probabilities)
    correct = predictions == targets

    return {
        "n_total": int(decision["n_total"]),
        "n_correct": int(np.count_nonzero(correct)),
        "accuracy": calibration.accuracy(probabilities, targets),
        "rejection_rate": float(decision["rejection_rate"]),
        "coverage": float(decision["coverage"]),
        "accepted_accuracy": float(decision["retained_accuracy"]),
        "false_rejection_rate": float(decision["frr"]),
        "false_accept_rate": float(decision["far"]),
        "yield": float(decision["accepted_correct"]) / max(1, decision["n_total"]),
        "mean_confidence": float(np.mean(confidences)),
    }


def evaluate_condition(audio, targets, rows, models, extractor, device, batch_size, mixer=None):
    """Score every model on one condition, sharing the features and the mixture.

    The mixture is drawn once per batch and fed to all models, so a comparison
    between two checkpoints cannot be explained by different noise. Features are
    extracted once for the same reason. Mixing happens in float on the
    accelerator and is snapped back onto the ``int16`` grid, the representation
    every cached clip and every served sample uses; the rounding floor is about
    -90 dBFS, i.e. far below the noisiest condition.
    """
    logits_by_version = {model.version: [] for model in models}

    for start in range(0, rows.size, batch_size):
        batch_rows = rows[start:start + batch_size]
        clean = audio[torch.from_numpy(np.ascontiguousarray(batch_rows))].to(device)

        if mixer is None:
            waveform = clean
        else:
            normalized = clean.to(torch.float32) / INT16_NORMALIZER
            waveform = quantize_int16(mixer.mix(normalized))

        with torch.no_grad():
            features = extractor(waveform)
            for model in models:
                logits = model.model(features)
                logits_by_version[model.version].append(logits.detach().float().cpu().numpy())

    metrics = {}
    for model in models:
        logits = np.concatenate(logits_by_version[model.version], axis=0)
        probabilities = calibration.logits_to_probabilities(logits, model.temperature)
        metrics[model.version] = summarize_condition(
            probabilities, targets, model.threshold
        )
    return metrics


class RobustnessModel:
    """One checkpoint together with the calibration payload it is served with."""

    def __init__(self, version, model, checkpoint, temperature, threshold, calibration_source):
        self.version = version
        self.model = model
        self.checkpoint = checkpoint
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.calibration_source = calibration_source

    @property
    def label(self):
        return f"{MODEL_NAME} {self.version}"


def load_robustness_model(version, device):
    """Rebuild ``CommandSense`` for ``version`` in ``eval()`` mode, or return ``None``."""
    checkpoint = resolve_checkpoint(version)
    if checkpoint is None:
        return None

    print(f"[*] Loading {version} weights: {checkpoint}")
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    temperature, threshold, source = load_calibration_values(version)
    return RobustnessModel(version, model, checkpoint, temperature, threshold, source)


def _percent(value):
    return f"{value * 100:.2f}%"


def _snr_cell(snr_db):
    return "-" if snr_db is None else f"{snr_db:g}"


def _field(label, value):
    """One ``key : value`` metadata line, aligned like ``report_v4.0.txt``."""
    return f"{label:<17}: {value}"


def condition_label(snr_db):
    return "Clean" if snr_db is None else "Noisy"


def markdown_table(headers, rows):
    """Pad ``rows`` into a GitHub-style table that stays readable as plain text."""
    body = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells):
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render([str(h) for h in headers]), separator] + [render(r) for r in body])


MAIN_HEADERS = ("SNR Level (dB)", "Condition", "Top-1 Accuracy", "Rejection Rate")
DETAIL_HEADERS = (
    "SNR Level (dB)",
    "Condition",
    "Accepted Accuracy",
    "Coverage",
    "Correct Yield",
    "Mean Confidence",
)
COMPARISON_HEADERS = (
    "SNR Level (dB)",
    "Condition",
    f"{MODEL_VERSION} Top-1",
    f"{COMPARISON_VERSION} Top-1",
    "Delta (pp)",
    f"{MODEL_VERSION} Reject",
    f"{COMPARISON_VERSION} Reject",
)


def accuracy_rows(results, version):
    return [
        (
            _snr_cell(record["snr_db"]),
            condition_label(record["snr_db"]),
            _percent(record["metrics"][version]["accuracy"]),
            _percent(record["metrics"][version]["rejection_rate"]),
        )
        for record in results
    ]


def detail_rows(results, version):
    return [
        (
            _snr_cell(record["snr_db"]),
            condition_label(record["snr_db"]),
            _percent(record["metrics"][version]["accepted_accuracy"]),
            _percent(record["metrics"][version]["coverage"]),
            _percent(record["metrics"][version]["yield"]),
            _percent(record["metrics"][version]["mean_confidence"]),
        )
        for record in results
    ]


def comparison_rows(results):
    rows = []
    for record in results:
        new = record["metrics"][MODEL_VERSION]
        old = record["metrics"][COMPARISON_VERSION]
        rows.append(
            (
                _snr_cell(record["snr_db"]),
                condition_label(record["snr_db"]),
                _percent(new["accuracy"]),
                _percent(old["accuracy"]),
                f"{(new['accuracy'] - old['accuracy']) * 100:+.2f}",
                _percent(new["rejection_rate"]),
                _percent(old["rejection_rate"]),
            )
        )
    return rows


def build_report(*, model, clips, bank, results, models, args, device, elapsed_seconds):
    """Render the text report; console and file receive identical content."""
    version = model.version
    noisy = [record for record in results if record["snr_db"] is not None]
    clean = next((record for record in results if record["snr_db"] is None), None)

    lines = ["=" * 68, "          COMMANDSENSE - NOISE ROBUSTNESS REPORT", "=" * 68, ""]
    lines.append(_field("Model version", f"{model.version} ({MODEL_NAME})"))
    lines.append(_field("Checkpoint", model.checkpoint))
    lines.append(_field("Calibration", model.calibration_source or "none (defaults)"))
    lines.append(
        _field("Evaluation split", f"'{EVALUATION_SUBSET}' (official clean, speaker-disjoint)")
    )
    lines.append(
        _field(
            "Command clips",
            f"{clips['scored']} of {clips['commands']} scored, {clips['classes']} command classes",
        )
    )
    if clips["scored"] != clips["commands"]:
        lines.append(_field("Subset mode", "yes - subset figures, not a release figure"))
    lines.append(
        _field(
            "Split size",
            f"{clips['total']} clips, incl. "
            f"{clips['total'] - clips['commands']} rejection-class clips",
        )
    )
    lines.append(
        _field("Served op point", f"T={model.temperature:.6f}, tau={model.threshold:.4f}")
    )
    lines.append(_field("Device", str(device)))
    lines.append(_field("Mixing seed", str(args.seed)))
    lines.append(_field("Runtime", f"{elapsed_seconds:.1f} s"))
    lines.append(
        _field("Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    )
    lines.append("")

    lines.append("=" * 68)
    lines.append(" HELD-OUT NOISE BANK - disjoint from the training noise sources")
    lines.append("=" * 68)
    lines.append(
        _field(
            "Ambient noise",
            f"{bank['ambient_chunks']} non-overlapping chunks of 1.0 s "
            f"(full yield of the {len(bank['ambient_files'])} held-out recordings)",
        )
    )
    lines.append(_field("  source", ", ".join(bank["ambient_files"])))
    lines.append(_field("  excluded", ", ".join(bank["ambient_excluded"]) + " (training)"))
    speech_desc = ", ".join(f"{split}: {count}" for split, count in bank["speech_splits"].items())
    lines.append(
        _field("Speech noise", f"{bank['speech_chunks']} chunks of 1.0 s ({speech_desc})")
    )
    lines.append(_field("  excluded", ", ".join(TRAINING_SPEECH_SPLITS) + " (training)"))
    if bank.get("speech_absent"):
        lines.append(
            _field(
                "  absent",
                ", ".join(bank["speech_absent"]) + " (provision with --download_speech_noise)",
            )
        )
    lines.append(
        _field(
            "Mix composition",
            f"{1.0 - NOISE_SPEECH_SOURCE_PROB:.0%} ambient / "
            f"{NOISE_SPEECH_SOURCE_PROB:.0%} speech, drawn per clip",
        )
    )
    lines.append(_field("Bank size", f"{bank['ambient_chunks'] + bank['speech_chunks']} chunks"))
    lines.append("")

    lines.append("=" * 68)
    lines.append(" ROBUSTNESS")
    lines.append("=" * 68)
    lines.append("")
    lines.append(markdown_table(MAIN_HEADERS, accuracy_rows(results, version)))
    lines.append("")
    if clean is not None and noisy:
        clean_metrics = clean["metrics"][version]
        worst = min(noisy, key=lambda record: record["metrics"][version]["accuracy"])
        worst_metrics = worst["metrics"][version]
        lines.append(
            _field(
                "Clean anchor",
                f"{_percent(clean_metrics['accuracy'])} "
                f"({clean_metrics['n_correct']}/{clean_metrics['n_total']})",
            )
        )
        lines.append(
            _field(
                "Mean noisy Top-1",
                _percent(float(np.mean([r["metrics"][version]["accuracy"] for r in noisy]))),
            )
        )
        lines.append(
            _field(
                "Worst condition",
                f"{_snr_cell(worst['snr_db'])} dB -> {_percent(worst_metrics['accuracy'])}",
            )
        )
        lines.append(
            _field(
                "Clean -> worst",
                f"{(worst_metrics['accuracy'] - clean_metrics['accuracy']) * 100:+.2f} pp",
            )
        )
        lines.append("")

    lines.append("Operating point detail (temperature-scaled, tau applied)")
    lines.append("")
    lines.append(markdown_table(DETAIL_HEADERS, detail_rows(results, version)))
    lines.append("")


    if len(models) > 1:
        baseline = next((entry for entry in models if entry.version == COMPARISON_VERSION), None)
        lines.append("=" * 68)
        lines.append(f" COMPARISON - {MODEL_VERSION} vs {COMPARISON_VERSION}")
        lines.append("=" * 68)
        lines.append("")
        lines.append(markdown_table(COMPARISON_HEADERS, comparison_rows(results)))
        lines.append("")
        if baseline is not None:
            lines.append(
                _field(
                    f"{baseline.version} operating pt",
                    f"T={baseline.temperature:.6f}, tau={baseline.threshold:.4f}",
                )
            )
        gains = [
            record["metrics"][MODEL_VERSION]["accuracy"]
            - record["metrics"][COMPARISON_VERSION]["accuracy"]
            for record in noisy
        ]
        if gains:
            lines.append(
                _field(
                    "Mean noisy delta",
                    f"{float(np.mean(gains)) * 100:+.2f} pp ({MODEL_VERSION} minus "
                    f"{COMPARISON_VERSION}, Top-1)",
                )
            )
        lines.append("")

    lines.append("NOTES")
    lines.append("-" * 68)
    lines.append("* Top-1 Accuracy is the raw argmax over every command clip and is")
    lines.append("  threshold-independent, i.e. the same figure as 'Top-1 Accuracy")
    lines.append("  (35 Commands)' in report_v4.0.txt.")
    lines.append("* Rejection Rate is the share of command clips whose temperature-scaled")
    lines.append("  top-1 confidence falls below tau and is therefore served as '_unknown_'.")
    lines.append("* Accepted Accuracy covers only answered clips; Correct Yield is answered")
    lines.append("  and correct over all clips, the fraction a user sees handled correctly.")
    lines.append("* Every row re-uses the same clean clips and every model receives the same")
    lines.append("  mixture, so a model-to-model delta is a property of the weights.")
    lines.append("* The bank never reads SpeechCommandsDataset.noise_bank: ambient noise comes")
    lines.append("  from the held-out _background_noise_ recordings, speech noise from")
    lines.append("  held-out LibriSpeech splits, so no evaluated mixture was ever trained on.")
    return "\n".join(lines)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Score CommandSense on the clean 'testing' split mixed with noise that "
            "training never saw, at fixed target SNRs."
        )
    )
    parser.add_argument(
        "--compare_v3",
        action="store_true",
        help="also score the v3.0 checkpoint and print a side-by-side table",
    )
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=list(DEFAULT_SNR_LEVELS_DB),
        metavar="DB",
        help="target SNR levels in dB (default: 5 10 15 20)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=f"score only this many command clips (0 = all {EXPECTED_COMMAND_CLIPS})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="clips per forward pass"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="seed for mixing and sampling")
    parser.add_argument(
        "--device", default=None, help="torch device override, e.g. 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--download_speech_noise",
        action="store_true",
        help="provision a missing LibriSpeech dev-clean/test-clean split",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"report path (default: reports/model_{MODEL_VERSION}/{REPORT_FILENAME})",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.max_samples < 0:
        parser.error("--max_samples must be >= 0")
    if not args.snr:
        parser.error("--snr needs at least one level")
    return args


def main(argv=None):
    args = parse_args(argv)
    if any(not np.isfinite(level) for level in args.snr):
        raise ValueError(f"--snr expects finite numbers, got {args.snr}")

    set_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Device: {device}")
    started = time.perf_counter()

    model = load_robustness_model(MODEL_VERSION, device)
    if model is None:
        print(
            f"[!] No {MODEL_VERSION} checkpoint found under "
            f"'{checkpoint_dir(MODEL_VERSION)}'; nothing to evaluate."
        )
        return 1
    models = [model]
    if args.compare_v3:
        baseline = load_robustness_model(COMPARISON_VERSION, device)
        if baseline is None:
            print(
                f"[!] No {COMPARISON_VERSION} checkpoint found under "
                f"'{checkpoint_dir(COMPARISON_VERSION)}'; comparison skipped."
            )
        else:
            models.append(baseline)

    audio, targets, rows, class_names, total_clips = load_command_split(
        max_samples=args.max_samples, seed=args.seed
    )

    bank = build_eval_noise_bank(seed=args.seed, allow_download=args.download_speech_noise)
    print(
        f"[+] Held-out bank: {bank['ambient_chunks']} ambient + "
        f"{bank['speech_chunks']} speech chunks"
    )

    extractor = AudioToThreeChannelMel().to(device).eval()

    results = []
    total_conditions = len(args.snr) + 1
    for index, snr_db in enumerate(args.snr):
        # Each condition draws its own mixture, reproducible from the seed alone.
        torch.manual_seed(args.seed + index)
        mixer = HeldOutSnrMixer(
            ambient_noise=bank["ambient"],
            speech_noise=bank["speech"],
            snr_db=snr_db,
            speech_source_prob=NOISE_SPEECH_SOURCE_PROB,
        ).to(device)
        print(f"\n[{index + 1}/{total_conditions}] Scoring at {snr_db:g} dB SNR ...")
        metrics = evaluate_condition(
            audio, targets, rows, models, extractor, device, args.batch_size, mixer=mixer
        )
        results.append({"snr_db": float(snr_db), "metrics": metrics})
        for entry in models:
            print(
                f"    {entry.version}: Top-1 {_percent(metrics[entry.version]['accuracy'])}, "
                f"rejection {_percent(metrics[entry.version]['rejection_rate'])}"
            )

    print(f"\n[{total_conditions}/{total_conditions}] Scoring clean baseline ...")
    metrics = evaluate_condition(
        audio, targets, rows, models, extractor, device, args.batch_size, mixer=None
    )
    results.append({"snr_db": None, "metrics": metrics})
    for entry in models:
        print(
            f"    {entry.version}: Top-1 {_percent(metrics[entry.version]['accuracy'])}, "
            f"rejection {_percent(metrics[entry.version]['rejection_rate'])}"
        )

    elapsed = time.perf_counter() - started
    clips = {
        "scored": int(rows.size),
        "commands": EXPECTED_COMMAND_CLIPS,
        "classes": len(class_names) - len(REJECTION_CLASSES),
        "total": int(total_clips),
    }
    report = build_report(
        model=model,
        clips=clips,
        bank=bank,
        results=results,
        models=models,
        args=args,
        device=device,
        elapsed_seconds=elapsed,
    )

    print("\n" + report)

    output = args.output or os.path.join(str(reports_dir(MODEL_VERSION)), REPORT_FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(f"\n[+] Report saved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


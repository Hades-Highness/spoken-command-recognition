"""Checkpoint loading and prediction with an optional ONNX Runtime backend.

Artifacts are resolved only from the configured ``MODEL_VERSION``: weights from
``checkpoints/model_<version>/<MODEL_NAME>_<version>.pth`` (canonical) or the
legacy flat layout, and ONNX from the shipped INT8 graph first, then the
unsuffixed export, then the float32 reference. A release without any usable
artifact raises :class:`ModelUnavailableError` with the expected paths and the
releases URL instead of substituting a different version.

Two input contracts are supported, because both graphs exist in the releases:
``export_onnx.py`` bakes the waveform front-end into the graph, so the shipped
artifact takes ``raw_audio`` ``[batch, TARGET_SAMPLES]`` in [-1, 1] and needs no
feature extraction at all, while the v3.x/v4.0 graphs take
``[batch, 3, N_MELS, frames]`` features. The graph's declared input rank decides
which one is fed (2 -> waveform, 4 -> features), so an older artifact keeps
working unchanged.

Both backends return raw logits, and ``softmax(logits / T)`` is applied in
Python, so they score identically and changing ``T`` or ``tau`` never requires
a re-export. Predictions below ``tau`` are served as ``REJECT_LABEL`` with
``is_low_confidence`` set. When no usable calibration file is found, the
inferencer falls back to ``T = 1.0`` and ``tau = 0.0`` (no low-confidence
rejection) rather than failing.

Recordings of any length are supported without manual trimming: the waveform is
cut into 1.0 s windows with a 0.5 s hop, every window is scored in one batch,
and the window that reaches the highest command-class probability decides the
recording. ``_silence_`` and ``_unknown_`` are never returned by that search, so
a long clip is answered by the moment a command was actually spoken instead of
by its first second.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import (  # noqa: E402
    CHECKPOINT_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    ONNX_DIR,
    REJECTION_CLASSES,
    REJECT_LABEL,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    checkpoint_dir,
)
from src import calibration  # noqa: E402
from src.models import AudioFeatureExtractor, CommandSense  # noqa: E402
from src.utils import (  # noqa: E402
    extract_features,
    load_audio_file,
    prepare_waveform,
)

EXPECTED_CHANNELS = 3
EXPECTED_TIME_FRAMES = 63
EXPECTED_SAMPLES = TARGET_SAMPLES

# Sliding-window contract for variable-length recordings: 1.0 s windows scanned
# every 0.5 s, i.e. 50 % overlap, so a command spoken across a boundary is still
# fully contained by one of the windows.
WINDOW_SAMPLES = TARGET_SAMPLES
WINDOW_HOP_SAMPLES = TARGET_SAMPLES // 2

# Where the self-contained release bundle (weights, ONNX graphs, calibration) is
# published; quoted by the errors raised when an artifact of MODEL_VERSION is absent.
RELEASES_URL = "https://github.com/Hades-Highness/spoken-command-recognition/releases"


def _positive_float(value):
    """Return ``value`` as a finite ``float`` > 0, or ``None`` when it is unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _probability_float(value):
    """Return ``value`` as a finite ``float`` inside [0, 1], or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and 0.0 <= number <= 1.0 else None


def _as_mono_float32(audio) -> np.ndarray:
    """Return ``audio`` as a contiguous float32 mono ``[N]`` array.

    Accepts ``[N]`` and ``[1, N]`` (any layout whose singleton axes squeeze
    away) from a NumPy array, a ``torch.Tensor`` or a sequence of samples.
    Multi-channel input is refused instead of silently averaged, and integer
    PCM is refused because casting it would skip the int16 -> float scaling the
    model was trained with.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    samples = np.asarray(audio)

    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError(
            f"Audio samples must be floating point, got dtype {samples.dtype}. "
            "Convert integer PCM with 'pcm.astype(np.float32) / 32768.0'."
        )

    if samples.ndim > 1:
        samples = np.squeeze(samples)
    if samples.ndim == 0:
        samples = samples.reshape(1)
    if samples.ndim != 1:
        raise ValueError(
            f"Expected mono audio shaped [N] or [1, N], got {tuple(samples.shape)}. "
            "Downmix the channels before calling the inferencer."
        )

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("The audio array is empty.")
    if not np.isfinite(samples).all():
        raise ValueError("The audio array contains non-finite values (NaN or inf).")
    return samples


def sliding_windows(
    audio,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = WINDOW_HOP_SAMPLES,
):
    """Cut ``audio`` into ``[n_windows, window_samples]`` float32 windows.

    A recording shorter than one window is zero-padded and returned as a single
    window, so the result always holds at least one window. Longer recordings
    are stepped by ``hop_samples``; when the last regular window does not reach
    the final sample, one window aligned to the end of the recording is
    appended so the tail is never dropped.

    Returns ``(windows, starts)``, where ``starts[i]`` is the sample offset of
    window ``i`` inside the (unpadded) recording.
    """
    samples = _as_mono_float32(audio)

    window_samples = int(window_samples)
    hop_samples = int(hop_samples)
    if window_samples <= 0:
        raise ValueError(f"window_samples must be > 0, got {window_samples}.")
    if hop_samples <= 0:
        raise ValueError(f"hop_samples must be > 0, got {hop_samples}.")

    total = int(samples.size)
    if total <= window_samples:
        window = np.zeros(window_samples, dtype=np.float32)
        window[:total] = samples
        return window[np.newaxis, :], [0]

    starts = list(range(0, total - window_samples + 1, hop_samples))
    if starts[-1] + window_samples < total:
        starts.append(total - window_samples)

    windows = np.stack(
        [samples[start : start + window_samples] for start in starts]
    ).astype(np.float32, copy=False)
    return windows, starts


class ModelUnavailableError(FileNotFoundError):
    """Raised when no artifact of the configured ``MODEL_VERSION`` can be found.

    Subclasses :class:`FileNotFoundError` so callers can catch either the specific
    error or the builtin, and so a missing release bundle fails as loudly as any
    other missing file in the pipeline.
    """


class CommandInferencer:
    """Wraps whichever backend (ONNX Runtime or PyTorch) is usable."""

    def __init__(
        self,
        checkpoint_path=None,
        labels_path=None,
        device=None,
        version=MODEL_VERSION,
        temperature=None,
        confidence_threshold=None,
        reject_label=REJECT_LABEL,
    ):
        self.version = version
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pth_path = self._resolve_checkpoint(checkpoint_path)
        self.onnx_path = self._resolve_onnx()
        self.labels = self._load_labels(labels_path)
        self.reject_label = reject_label

        self.feature_extractor = AudioFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

        # Calibration state: explicit arguments win over the calibration file,
        # which in turn wins over the defaults from configs/config.py.
        self.calibration = {}
        self.calibration_source = None
        self.temperature = float(DEFAULT_TEMPERATURE if temperature is None else temperature)
        self.confidence_threshold = float(
            DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )

        self._onnx_session = None
        self._onnx_input_name = None
        self._onnx_input_rank = None
        self._torch_model = None
        self._command_indices = None

        self.available_backends = ["pytorch"]
        if self.onnx_path and self._onnx_runtime_installed():
            self.available_backends.insert(0, "onnx")

        if self.pth_path is None and not (self.onnx_path and self._onnx_runtime_installed()):
            expected_dir = checkpoint_dir(self.version)
            raise ModelUnavailableError(
                f"Cannot find the {MODEL_NAME} {self.version} model files for the "
                f"configured MODEL_VERSION. Looked for "
                f"'{expected_dir}/{MODEL_NAME}_{self.version}.pth' and "
                f"'{expected_dir}/{MODEL_NAME}_{self.version}.onnx' "
                f"(or the '{MODEL_NAME}_{self.version}_int8.onnx' / "
                f"'{MODEL_NAME}_{self.version}_fp32.onnx' graphs). "
                f"Rebuild them with 'python export_onnx.py', or download the "
                f"release bundle from the GitHub Releases page "
                f"({RELEASES_URL}) and place the {self.version} artifacts in "
                f"'{expected_dir}'."
            )

        self.load_calibration(
            temperature=None if temperature is None else temperature,
            confidence_threshold=(
                None if confidence_threshold is None else confidence_threshold
            ),
        )

    def load_calibration(self, temperature=None, confidence_threshold=None) -> dict:
        """Load ``T``/``tau`` from the version's calibration JSON.

        The canonical payload is
        ``configs/calibration/<MODEL_NAME>_calibration_<version>.json``; the
        version's checkpoint folder is also probed for a legacy copy. This method
        never raises: an absent or unusable file leaves ``T = 1.0`` and
        ``tau = 0.0`` in place, so the application keeps serving rather than
        failing.
        """
        path = calibration.resolve_calibration_file(self.version)
        if path is not None:
            payload = calibration.load_calibration(path)
            if payload and not self._usable_calibration(payload):
                print(
                    f"[!] Ignoring unusable calibration file {path}: T must be a "
                    "positive number and tau a probability in [0, 1]. Falling back "
                    f"on T={DEFAULT_TEMPERATURE:.4f}, "
                    f"tau={DEFAULT_CONFIDENCE_THRESHOLD:.2f}."
                )
            elif payload:
                self.calibration = payload
                self.calibration_source = str(path)
                print(
                    f"[+] Loaded calibration (T={payload.get('temperature')}, "
                    f"tau={payload.get('confidence_threshold')}) <- {path}"
                )

        if not self.calibration:
            print(
                "[!] No calibration file found; falling back on the defaults "
                f"(T={DEFAULT_TEMPERATURE:.4f}, tau={DEFAULT_CONFIDENCE_THRESHOLD:.2f}): "
                "raw logits are served and no clip is rejected for low confidence. "
                "Run evaluate.py to fit T and tau."
            )

        self.set_calibration(
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )
        return self.calibration

    @staticmethod
    def _usable_calibration(payload: dict) -> bool:
        """True when every value present in ``payload`` can be applied safely.

        While missing keys are fine - they simply keep their configured default -
        a value that *is* present has to be a positive ``T`` and a ``tau`` inside
        [0, 1]; anything else is refused so the defaults are served rather than a
        file that would make ``softmax(logits / T)`` fail at prediction time.
        """
        temperature = _positive_float(payload.get("temperature", DEFAULT_TEMPERATURE))
        threshold = _probability_float(
            payload.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
        return temperature is not None and threshold is not None

    def set_calibration(self, temperature=None, confidence_threshold=None) -> None:
        """Override ``T`` and/or ``tau`` at runtime (e.g. from the UI slider).

        Passing ``None`` keeps the current value, so the threshold can be swept
        live without touching the temperature and vice versa.
        """
        if temperature is not None:
            temperature = float(temperature)
            if temperature <= 0:
                raise ValueError(f"temperature must be > 0, got {temperature}.")
            self.temperature = temperature
        elif "temperature" in self.calibration:
            self.temperature = float(self.calibration["temperature"])

        if confidence_threshold is not None:
            confidence_threshold = float(confidence_threshold)
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(
                    f"confidence threshold must be in [0, 1], got {confidence_threshold}."
                )
            self.confidence_threshold = confidence_threshold
        elif "confidence_threshold" in self.calibration:
            self.confidence_threshold = float(self.calibration["confidence_threshold"])

    def _resolve_checkpoint(self, checkpoint_path):
        """Path of the ``.pth`` weights of ``self.version``, or ``None``.

        An explicit ``checkpoint_path`` always wins (callers and tests that point at
        one file); otherwise the version's canonical layout is probed, then the
        legacy flat one. Only ``self.version`` is ever considered - no other release
        is substituted for a missing one.
        """
        candidates = [
            checkpoint_path,
            checkpoint_dir(self.version) / f"{MODEL_NAME}_{self.version}.pth",
            CHECKPOINT_DIR / f"{MODEL_NAME}_{self.version}.pth",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return str(candidate)
        return None

    def _resolve_onnx(self):
        """Path of the ONNX graph of ``self.version``, or ``None``.

        The shipped deployment artifact is the quantized graph, so
        ``<MODEL_NAME>_<version>_int8.onnx`` (an older release's quantized copy)
        is preferred, then the unsuffixed ``<MODEL_NAME>_<version>.onnx`` that
        ``export_onnx.py`` writes today (front-end baked in, INT8), then the
        ``<MODEL_NAME>_<version>_fp32.onnx`` reference written next to it.
        Each name is probed in the version's checkpoint folder first, then the
        legacy flat checkpoint folder and ``onnx/``. As in
        :meth:`_resolve_checkpoint`, no other release is ever substituted.
        """
        names = [
            f"{MODEL_NAME}_{self.version}_int8.onnx",
            f"{MODEL_NAME}_{self.version}.onnx",
            f"{MODEL_NAME}_{self.version}_fp32.onnx",
        ]
        for directory in (checkpoint_dir(self.version), CHECKPOINT_DIR, ONNX_DIR):
            for name in names:
                candidate = directory / name
                if os.path.exists(candidate):
                    return str(candidate)
        return None

    def _load_labels(self, labels_path):
        candidates = [
            labels_path,
            LABELS_PATH,
            checkpoint_dir(self.version) / "labels.json",
            CHECKPOINT_DIR / "labels.json",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as handle:
                    labels = json.load(handle)
                print(f"[+] Loaded class mapping ({len(labels)} classes) <- {candidate}")
                return labels

        print(
            "[!] No labels.json found. Predictions will be labelled "
            f"Class_0..Class_{NUM_CLASSES - 1}. Run train.py or commit configs/labels.json."
        )
        return [f"Class_{i}" for i in range(NUM_CLASSES)]

    @staticmethod
    def _onnx_runtime_installed():
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_pytorch(self):
        if self._torch_model is not None:
            return
        if self.pth_path is None:
            raise FileNotFoundError(
                f"PyTorch checkpoint for {MODEL_NAME} {self.version} was not found. "
                f"Download '{MODEL_NAME}_{self.version}.pth' into "
                f"'{checkpoint_dir(self.version)}' from the release bundle at "
                f"{RELEASES_URL}, or train that version first."
            )

        model = CommandSense(num_classes=len(self.labels), in_channels=EXPECTED_CHANNELS)
        state_dict = torch.load(self.pth_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()

        self._torch_model = model
        print(f"[+] Loaded PyTorch model (3 channels) <- {self.pth_path}")

    def _load_onnx(self):
        if self._onnx_session is not None:
            return
        if self.onnx_path is None:
            raise FileNotFoundError(
                f"ONNX model for {MODEL_NAME} {self.version} was not found. "
                f"Download '{MODEL_NAME}_{self.version}.onnx' (the INT8 deployment "
                f"graph) or '{MODEL_NAME}_{self.version}_fp32.onnx' (the float32 "
                f"reference) into '{checkpoint_dir(self.version)}' from the release "
                f"bundle at {RELEASES_URL}, or export that version with "
                "'python export_onnx.py'."
            )

        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(self.onnx_path, providers=providers)
        input_meta = session.get_inputs()[0]
        declared = input_meta.shape
        rank = len(declared)

        if rank == 2:
            # Baked front-end: the graph takes the waveform itself.
            declared_samples = declared[-1]
            if isinstance(declared_samples, int) and declared_samples != EXPECTED_SAMPLES:
                raise ValueError(
                    f"ONNX model '{self.onnx_path}' expects {declared_samples} audio "
                    f"samples but each clip is {EXPECTED_SAMPLES}. "
                    "Re-export the model with 'python export_onnx.py'."
                )
            contract = "raw audio, front-end baked in"
        elif rank == 4:
            # Feature graph (v3.x / v4.0): the front-end runs in this process.
            declared_channels = declared[1]
            if declared_channels not in (EXPECTED_CHANNELS, "3"):
                raise ValueError(
                    f"ONNX model '{self.onnx_path}' declares {declared_channels} input "
                    f"channels but this pipeline builds {EXPECTED_CHANNELS}. "
                    "Re-export the model with 'python export_onnx.py'."
                )
            declared_frames = declared[3]
            if isinstance(declared_frames, int) and declared_frames != EXPECTED_TIME_FRAMES:
                raise ValueError(
                    f"ONNX model '{self.onnx_path}' expects {declared_frames} time frames "
                    f"but the feature pipeline produces {EXPECTED_TIME_FRAMES}. "
                    "This export is stale; re-export it."
                )
            contract = "features, extracted in-process"
        else:
            raise ValueError(
                f"ONNX model '{self.onnx_path}' has an unexpected input rank: {declared}. "
                f"Expected rank 2 '[batch, {EXPECTED_SAMPLES}]' (raw audio) or rank 4 "
                f"'[batch, {EXPECTED_CHANNELS}, N_MELS, {EXPECTED_TIME_FRAMES}]' "
                "(features)."
            )

        self._onnx_session = session
        self._onnx_input_name = input_meta.name
        self._onnx_input_rank = rank
        print(
            f"[+] Loaded ONNX Runtime session <- {self.onnx_path} "
            f"(input {declared}, {contract})"
        )

    def _features(self, waveform):
        """3-channel features of one ``[1, TARGET_SAMPLES]`` waveform, checked.

        Only the rank-4 contract needs this; for a baked graph the waveform goes
        straight to the session and torchaudio is not touched.
        """
        features = extract_features(
            waveform, device=self.device, extractor=self.feature_extractor
        )
        if features.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_CHANNELS}-channel features, got shape "
                f"{tuple(features.shape)}."
            )
        return features

    def _batch_features(self, waveforms):
        """3-channel features of a ``[B, TARGET_SAMPLES]`` waveform batch.

        ``AudioFeatureExtractor`` keeps the leading batch axis, so the whole
        sliding-window batch is transformed in a single front-end pass instead
        of one pass per window.
        """
        features = extract_features(
            waveforms, device=self.device, extractor=self.feature_extractor
        )
        if features.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_CHANNELS}-channel features, got shape "
                f"{tuple(features.shape)}."
            )
        return features

    def get_onnx_session(self):
        """Return the cached ``(session, input_name)`` of the ONNX graph.

        Loads the graph on first use so real-time callers reuse the session that
        ``predict(..., backend="onnx")`` uses instead of opening a second one.
        """
        self._load_onnx()
        return self._onnx_session, self._onnx_input_name

    @property
    def is_ready(self) -> bool:
        return bool(self.available_backends)

    @property
    def command_indices(self) -> np.ndarray:
        """Indices of the classes a recording may be served as.

        ``_silence_`` and ``_unknown_`` are rejection classes and are excluded,
        so the sliding-window search can only answer with a valid command.
        """
        if self._command_indices is None:
            rejected = {str(label).lower() for label in REJECTION_CLASSES}
            self._command_indices = np.array(
                [
                    index
                    for index, label in enumerate(self.labels)
                    if str(label).lower() not in rejected
                ],
                dtype=np.int64,
            )
        return self._command_indices

    def _select_backend(self, backend: str) -> bool:
        """True when ``backend`` resolves to ONNX Runtime, False for PyTorch.

        ``"auto"`` prefers ONNX, which is what single-recording inference used
        before the sliding window existed.
        """
        choice = str(backend).lower()
        if choice in ("pytorch", "pth", ".pth"):
            return False
        if choice == "onnx":
            return True
        return "onnx" in self.available_backends

    def _window_logits(self, windows: np.ndarray, use_onnx: bool) -> np.ndarray:
        """Raw logits ``[n_windows, n_classes]`` of a batch of 1.0 s windows.

        The batch is scored with a single graph call: a baked-front-end graph
        takes the ``[n, TARGET_SAMPLES]`` windows directly, while a feature
        graph gets its 3 channels built in one front-end pass.
        """
        batch = np.ascontiguousarray(windows, dtype=np.float32)
        if use_onnx:
            self._load_onnx()
            if self._onnx_input_rank == 2:
                payload = batch
            else:
                payload = self._batch_features(torch.from_numpy(batch)).cpu().numpy()
            logits = self._onnx_session.run(None, {self._onnx_input_name: payload})[0]
        else:
            self._load_pytorch()
            features = self._batch_features(torch.from_numpy(batch))
            with torch.no_grad():
                logits = self._torch_model(features).cpu().numpy()

        logits = np.asarray(logits, dtype=np.float64).reshape(batch.shape[0], -1)
        if logits.shape[1] != len(self.labels):
            raise ValueError(
                f"The backend returned {logits.shape[1]} logits per window but "
                f"{len(self.labels)} labels were loaded."
            )
        return logits

    def predict_from_raw_audio(
        self,
        audio_array,
        backend: str = "auto",
        confidence_threshold=None,
        temperature=None,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict:
        """Classify a raw mono recording of any length with a sliding window.

        ``audio_array`` is a float array-like shaped ``[N]`` or ``[1, N]``
        (NumPy array, ``torch.Tensor`` or a sequence of samples) scaled to
        [-1, 1] at ``sample_rate`` (16 kHz by default). Nothing has to be
        trimmed by hand: a clip shorter than 1.0 s is zero-padded to
        ``TARGET_SAMPLES``, a longer one is cut into 1.0 s windows with a 0.5 s
        hop (50 % overlap, plus an end-aligned tail window when needed).

        All windows are scored in one batched call and the "best match" rule
        picks the answer for the whole recording:

        * the search runs over the command classes only, so ``_silence_`` and
          ``_unknown_`` can never win a window;
        * the window holding the highest command probability decides;
        * that probability is compared with ``tau``; below it the served label
          is the reject label and ``accepted`` is False.

        The returned dictionary carries the same keys as :meth:`predict` plus
        the window diagnostics (``window_count``, ``window_index``,
        ``window_start_seconds``, ``window_predictions``), so a caller can see
        where in the recording the command was found.
        """
        if confidence_threshold is not None or temperature is not None:
            self.set_calibration(
                temperature=temperature, confidence_threshold=confidence_threshold
            )

        samples = _as_mono_float32(audio_array)
        if sample_rate is not None and int(sample_rate) != SAMPLE_RATE:
            samples = (
                prepare_waveform(
                    torch.from_numpy(samples).unsqueeze(0),
                    int(sample_rate),
                    target_samples=None,
                )
                .squeeze(0)
                .numpy()
            )

        windows, starts = sliding_windows(samples)

        command_indices = self.command_indices
        if command_indices.size == 0:
            raise ValueError(
                "The loaded labels contain rejection classes only; the "
                "sliding-window search needs at least one command class."
            )

        use_onnx = self._select_backend(backend)
        logits = self._window_logits(windows, use_onnx)
        probabilities = calibration.logits_to_probabilities(logits, self.temperature)

        # Best match: highest command probability across every window.
        command_probabilities = probabilities[:, command_indices]
        flat_index = int(np.argmax(command_probabilities))
        window_index, command_offset = np.unravel_index(
            flat_index, command_probabilities.shape
        )
        window_index = int(window_index)
        command_index = int(command_indices[command_offset])
        best_label = self.labels[command_index]
        best_confidence = float(probabilities[window_index, command_index])

        threshold = float(self.confidence_threshold)
        accepted = best_confidence >= threshold
        served_label = best_label if accepted else self.reject_label

        window_predictions = []
        for index, start in enumerate(starts):
            offset = int(np.argmax(command_probabilities[index]))
            class_index = int(command_indices[offset])
            score = float(command_probabilities[index, offset])
            window_predictions.append(
                {
                    "index": index,
                    "start_samples": int(start),
                    "start_seconds": round(start / SAMPLE_RATE, 3),
                    "label": self.labels[class_index],
                    "confidence": score,
                    "accepted": bool(score >= threshold),
                }
            )

        return {
            "label": served_label,
            "raw_label": best_label,
            "confidence": best_confidence,
            "is_low_confidence": not accepted,
            "accepted": accepted,
            "probabilities": {
                label: float(probability)
                for label, probability in zip(self.labels, probabilities[window_index])
            },
            "backend": "onnx" if use_onnx else "pytorch",
            "temperature": self.temperature,
            "threshold": threshold,
            "window_count": len(starts),
            "window_index": window_index,
            "window_start_samples": int(starts[window_index]),
            "window_start_seconds": round(starts[window_index] / SAMPLE_RATE, 3),
            "window_predictions": window_predictions,
        }

    def predict(
        self,
        audio_path,
        backend: str = "auto",
        confidence_threshold=None,
        temperature=None,
    ) -> dict:
        """Classify one recording, given as a file path or as raw samples.

        ``audio_path`` is either a filesystem path or a float array shaped
        ``[N]`` / ``[1, N]`` of any length. Both routes end in
        :meth:`predict_from_raw_audio`: a file is decoded once and resampled to
        16 kHz mono *without* being cut to 1.0 s, an array is used as it is, and
        the sliding window then scores the whole recording.

        The logits of the selected backend are rescaled by ``T`` before the
        softmax, and the highest command probability of the recording is
        compared with ``tau``.

        Returns ``{"label", "raw_label", "confidence", "is_low_confidence",
        "accepted", "probabilities", "backend", "temperature", "threshold"}``
        plus the window diagnostics, where ``label`` is the served label (the
        reject label when no window reaches ``tau``) and ``probabilities`` maps
        every class name of the winning window to its calibrated probability.
        """
        if not isinstance(audio_path, (str, os.PathLike)):
            return self.predict_from_raw_audio(
                audio_path,
                backend=backend,
                confidence_threshold=confidence_threshold,
                temperature=temperature,
            )
        if not audio_path:
            raise ValueError("No audio file provided.")

        waveform = prepare_waveform(*load_audio_file(audio_path), target_samples=None)
        return self.predict_from_raw_audio(
            waveform,
            backend=backend,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
        )

    def decide(self, logits, top_k: int = 5) -> dict:
        """Apply ``T`` and ``tau`` to one row of raw logits.

        Kept separate from :meth:`predict` so the evaluator and the tests can
        score pre-computed logits without touching audio files.
        """
        probabilities = calibration.logits_to_probabilities(logits, self.temperature)
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])[0]
        decision = calibration.apply_confidence_threshold(
            probabilities,
            threshold=self.confidence_threshold,
            labels=self.labels,
            reject_label=self.reject_label,
        )
        order = np.argsort(probabilities)[::-1][: max(1, int(top_k))]
        decision["probabilities"] = probabilities
        decision["top_k"] = [
            {"label": self.labels[index], "confidence": float(probabilities[index])}
            for index in order
        ]
        return decision


if __name__ == "__main__":
    engine = CommandInferencer()
    print(f"[+] CommandInferencer ready on {engine.device}")
    print(f"[+] Backends available : {', '.join(engine.available_backends)}")
    print(f"[+] Weights            : {engine.pth_path or engine.onnx_path}")
    print(
        f"[+] Calibration        : T={engine.temperature:.4f} "
        f"tau={engine.confidence_threshold:.2f} "
        f"(source: {engine.calibration_source or 'defaults'})"
    )
    print(f"[+] Classes            : {len(engine.labels)}")

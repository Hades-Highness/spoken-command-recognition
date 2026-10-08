"""Checkpoint loading and prediction, with an optional ONNX Runtime backend.

Resolution order for the checkpoint:

1. ``checkpoints/model_<version>/<MODEL_NAME>_<version>.pth``  (canonical)
2. ``checkpoints/model_<BASE_MODEL_VERSION>/...``              (weights reused by
   a calibration-only release such as v2.2, which ships no checkpoint of its own)
3. ``checkpoints/<MODEL_NAME>_<version>.pth``                 (legacy layout)

Backend selection:

* ``"onnx"``   - ONNX Runtime, requires the matching ``.onnx`` file.
* ``"pytorch"`` - PyTorch checkpoint.
* ``"auto"``   - ONNX when the file exists and loads, PyTorch otherwise.

Confidence calibration (v2.2):

Both backends return raw logits, and the temperature ``T`` is applied to them in
Python (``softmax(logits / T)``), so the two backends score identically and the
exported graph never has to be rebuilt when ``T`` or the threshold changes. Every
prediction also carries a confidence-threshold decision: when the calibrated
top-1 probability stays below ``tau``, the served label becomes the reject label
(``REJECT_LABEL``) and ``is_low_confidence`` is set.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import (  # noqa: E402
    BASE_MODEL_VERSION,
    CHECKPOINT_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    ONNX_DIR,
    REJECT_LABEL,
    checkpoint_dir,
)
from src import calibration  # noqa: E402
from src.models import AudioFeatureExtractor, CommandSense  # noqa: E402
from src.utils import load_and_preprocess_audio  # noqa: E402

EXPECTED_CHANNELS = 3
EXPECTED_TIME_FRAMES = 63


class ModelUnavailableError(RuntimeError):
    """Raised when neither backend can be initialised."""


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

        # Calibration state. Explicit arguments win over the calibration file,
        # which in turn wins over the defaults from configs/config.py.
        self.calibration = {}
        self.calibration_source = None
        self.temperature = float(DEFAULT_TEMPERATURE if temperature is None else temperature)
        self.confidence_threshold = float(
            DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )

        self._onnx_session = None
        self._onnx_input_name = None
        self._torch_model = None

        self.available_backends = ["pytorch"]
        if self.onnx_path and self._onnx_runtime_installed():
            self.available_backends.insert(0, "onnx")

        if self.pth_path is None and not (self.onnx_path and self._onnx_runtime_installed()):
            raise ModelUnavailableError(
                "No usable model found. Expected either "
                f"'{checkpoint_dir(self.version)}/{MODEL_NAME}_{self.version}.pth' or "
                f"a matching .onnx file. Download the release asset or run train.py."
            )

        self.load_calibration(
            temperature=None if temperature is None else temperature,
            confidence_threshold=(
                None if confidence_threshold is None else confidence_threshold
            ),
        )

    # ----------------------------------------------------------- calibration
    def load_calibration(self, temperature=None, confidence_threshold=None) -> dict:
        """Load ``T``/``tau`` from the version's calibration JSON.

        A calibration-only release (v2.2) reuses the weights and the calibration
        file of ``BASE_MODEL_VERSION`` when it has none of its own. Whatever is
        missing falls back to the configured defaults, so an absent or partial
        file never breaks inference.
        """
        for candidate_version in (self.version, BASE_MODEL_VERSION):
            path = calibration.resolve_calibration_file(candidate_version)
            if path is None:
                continue
            payload = calibration.load_calibration(path)
            if not payload:
                continue
            self.calibration = payload
            self.calibration_source = str(path)
            print(
                f"[+] Loaded calibration (T={payload.get('temperature')}, "
                f"tau={payload.get('confidence_threshold')}) <- {path}"
            )
            break

        if not self.calibration:
            print(
                "[!] No calibration file found; using defaults "
                f"(T={self.temperature:.4f}, tau={self.confidence_threshold:.2f}). "
                "Run evaluate.py to fit them."
            )

        self.set_calibration(
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )
        return self.calibration

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

    # ------------------------------------------------------------------ paths
    def _resolve_checkpoint(self, checkpoint_path):
        candidates = [
            checkpoint_path,
            checkpoint_dir(self.version) / f"{MODEL_NAME}_{self.version}.pth",
            checkpoint_dir(BASE_MODEL_VERSION) / f"{MODEL_NAME}_{BASE_MODEL_VERSION}.pth",
            CHECKPOINT_DIR / f"{MODEL_NAME}_{self.version}.pth",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return str(candidate)
        return None

    def _resolve_onnx(self):
        candidates = [
            checkpoint_dir(self.version) / f"{MODEL_NAME}_{self.version}.onnx",
            checkpoint_dir(BASE_MODEL_VERSION) / f"{MODEL_NAME}_{BASE_MODEL_VERSION}.onnx",
            CHECKPOINT_DIR / f"{MODEL_NAME}_{self.version}.onnx",
            ONNX_DIR / f"{MODEL_NAME}_{self.version}.onnx",
            ONNX_DIR / f"{MODEL_NAME}_{BASE_MODEL_VERSION}.onnx",
        ]
        for candidate in candidates:
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

    # --------------------------------------------------------------- backends
    def _load_pytorch(self):
        if self._torch_model is not None:
            return
        if self.pth_path is None:
            raise FileNotFoundError(
                f"PyTorch checkpoint not found for '{self.version}' under '{CHECKPOINT_DIR}'."
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
                f"ONNX model not found for '{self.version}'. Export it with "
                "'python export_onnx.py' or download the release asset."
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

        if len(declared) != 4:
            raise ValueError(
                f"ONNX model '{self.onnx_path}' has an unexpected input rank: {declared}."
            )
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

        self._onnx_session = session
        self._onnx_input_name = input_meta.name
        print(f"[+] Loaded ONNX Runtime session <- {self.onnx_path} (input {declared})")

    @property
    def is_ready(self) -> bool:
        return bool(self.available_backends)

    # -------------------------------------------------------------- inference
    def predict(
        self,
        audio_path: str,
        backend: str = "auto",
        confidence_threshold=None,
        temperature=None,
    ) -> dict:
        """Classify one audio file.

        The logits of the selected backend are rescaled by ``T`` before the
        softmax, then the top-1 probability is compared with ``tau``.

        Returns ``{"label", "raw_label", "confidence", "is_low_confidence",
        "accepted", "probabilities", "backend", "temperature", "threshold"}``
        where ``label`` is the served label (the reject label when the decision
        is too uncertain) and ``probabilities`` maps every class name to its
        calibrated probability.
        """
        if not audio_path:
            raise ValueError("No audio file provided.")
        if confidence_threshold is not None or temperature is not None:
            self.set_calibration(
                temperature=temperature, confidence_threshold=confidence_threshold
            )

        features = load_and_preprocess_audio(
            audio_path, device=self.device, extractor=self.feature_extractor
        )
        if features.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_CHANNELS}-channel features, got shape {tuple(features.shape)}."
            )

        choice = str(backend).lower()
        if choice in ("pytorch", "pth", ".pth"):
            use_onnx = False
        elif choice == "onnx":
            use_onnx = True
        else:  # "auto"
            use_onnx = "onnx" in self.available_backends

        if use_onnx:
            self._load_onnx()
            logits = self._onnx_session.run(
                None, {self._onnx_input_name: features.cpu().numpy()}
            )[0]
        else:
            self._load_pytorch()
            with torch.no_grad():
                logits = self._torch_model(features).cpu().numpy()

        decision = self.decide(logits, top_k=1)

        probabilities = decision["probabilities"]
        return {
            "label": decision["label"],
            "raw_label": decision["predicted_label"],
            "confidence": decision["confidence"],
            "is_low_confidence": decision["is_low_confidence"],
            "accepted": decision["accepted"],
            "probabilities": {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            },
            "backend": "onnx" if use_onnx else "pytorch",
            "temperature": self.temperature,
            "threshold": self.confidence_threshold,
        }

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


CommandSenseInferencer = CommandInferencer


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


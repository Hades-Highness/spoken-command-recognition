"""Checkpoint loading and prediction, with an optional ONNX Runtime backend.

Resolution order for the checkpoint:

1. ``checkpoints/model_<version>/<MODEL_NAME>_<version>.pth``  (canonical)
2. ``checkpoints/<MODEL_NAME>_<version>.pth``                 (legacy layout)

Backend selection:

* ``"onnx"``   - ONNX Runtime, requires the matching ``.onnx`` file.
* ``"pytorch"`` - PyTorch checkpoint.
* ``"auto"``   - ONNX when the file exists and loads, PyTorch otherwise.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import CHECKPOINT_DIR, LABELS_PATH, MODEL_NAME, MODEL_VERSION, NUM_CLASSES, ONNX_DIR, checkpoint_dir # noqa: E402
from src.models import AudioFeatureExtractor, CommandSense  # noqa: E402
from src.utils import load_and_preprocess_audio  # noqa: E402

EXPECTED_CHANNELS = 3
EXPECTED_TIME_FRAMES = 63


class ModelUnavailableError(RuntimeError):
    """Raised when neither backend can be initialised."""


class CommandInferencer:
    """Wraps whichever backend (ONNX Runtime or PyTorch) is usable."""

    def __init__(self, checkpoint_path=None, labels_path=None, device=None, version=MODEL_VERSION):
        self.version = version
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pth_path = self._resolve_checkpoint(checkpoint_path)
        self.onnx_path = self._resolve_onnx()
        self.labels = self._load_labels(labels_path)

        self.feature_extractor = AudioFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

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

    # ------------------------------------------------------------------ paths
    def _resolve_checkpoint(self, checkpoint_path):
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
        candidates = [
            checkpoint_dir(self.version) / f"{MODEL_NAME}_{self.version}.onnx",
            CHECKPOINT_DIR / f"{MODEL_NAME}_{self.version}.onnx",
            ONNX_DIR / f"{MODEL_NAME}_{self.version}.onnx",
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
    def predict(self, audio_path: str, backend: str = "auto") -> dict:
        """Classify one audio file.

        Returns ``{"label", "confidence", "probabilities"}`` where
        ``probabilities`` maps every class name to its probability.
        """
        if not audio_path:
            raise ValueError("No audio file provided.")

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
            logits = torch.from_numpy(
                self._onnx_session.run(
                    None, {self._onnx_input_name: features.cpu().numpy()}
                )[0]
            )
        else:
            self._load_pytorch()
            with torch.no_grad():
                logits = self._torch_model(features)

        probabilities = F.softmax(logits, dim=1).squeeze(0)
        confidence, index = torch.max(probabilities, dim=0)

        return {
            "label": self.labels[index.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            },
            "backend": "onnx" if use_onnx else "pytorch",
        }


CommandSenseInferencer = CommandInferencer


if __name__ == "__main__":
    engine = CommandInferencer()
    print(f"[+] CommandInferencer ready on {engine.device}")
    print(f"[+] Backends: {engine.available_backends} | classes: {len(engine.labels)}")

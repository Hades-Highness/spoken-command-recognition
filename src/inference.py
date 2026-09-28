import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

from src.models import CommandSense
from src.utils import load_and_preprocess_audio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import MODEL_NAME, MODEL_VERSION, NUM_CLASSES, CHECKPOINT_DIR

CLASS_NAMES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

class CommandSenseInferencer:
    """Wrapper class for running predictions using either ONNX Runtime or PyTorch for v2.0.0 (3 channels)."""
    def __init__(self, model_dir=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build candidate folder paths
        version_folder = f"model_{MODEL_VERSION.lower()}"
        candidate_dirs = [
            model_dir,
            os.path.join(CHECKPOINT_DIR, version_folder),
            os.path.join("..", CHECKPOINT_DIR, version_folder),
            CHECKPOINT_DIR,
            os.path.join("..", CHECKPOINT_DIR)
        ]

        self.model_dir = None
        for path in candidate_dirs:
            if path and os.path.exists(path):
                self.model_dir = path
                break

        if not self.model_dir:
            self.model_dir = os.path.join(CHECKPOINT_DIR, version_folder)

        # File paths setup
        self.onnx_path = os.path.join(self.model_dir, f"{MODEL_NAME}_{MODEL_VERSION}.onnx")
        self.pth_path = os.path.join(self.model_dir, f"{MODEL_NAME}_{MODEL_VERSION}.pth")
        self.labels_path = os.path.join(self.model_dir, "labels.json")

        # Dynamic class names resolution (v2.0.0)
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.class_names = json.load(f)
            print(f"[+] Loaded class mapping from {self.labels_path}")
        else:
            self.class_names = CLASS_NAMES

        # Lazy loading sessions
        self.ort_session = None
        self.pytorch_model = None

    def _load_onnx(self):
        """Loads ONNX Runtime session with strict v2.0.0 input shape validation."""
        if self.ort_session is None:
            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(f"ONNX model file not found at {self.onnx_path}")
            
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(self.onnx_path, providers=providers)
            
            # Input shape validation guard for v2.0.0 (in_channels == 3)
            input_meta = self.ort_session.get_inputs()[0]
            self.input_name = input_meta.name
            declared_shape = input_meta.shape
            
            # Strict check for 3-channel input in ONNX model
            if len(declared_shape) >= 2 and declared_shape[1] != 3:
                raise ValueError(f"[v2.0.0 Error] Expected ONNX model with 3 channels (declared_shape[1] == 3), got shape {declared_shape}")

            print(f"[+] Loaded ONNX runtime session -> {self.onnx_path} (Declared shape: {declared_shape})")

    def _load_pytorch(self):
        """Loads PyTorch model checkpoint with 3 input channels for v2.0.0."""
        if self.pytorch_model is None:
            if not os.path.exists(self.pth_path):
                raise FileNotFoundError(f"PyTorch model checkpoint not found at {self.pth_path}")
            
            # v2.0.0 requires in_channels=3
            self.pytorch_model = CommandSense(num_classes=len(self.class_names), in_channels=3).to(self.device)
            self.pytorch_model.load_state_dict(torch.load(self.pth_path, map_location=self.device))
            self.pytorch_model.eval()
            print(f"[+] Loaded PyTorch model (3 channels) -> {self.pth_path}")

    @torch.no_grad()
    def predict(self, audio_path, backend="onnx"):
        """Preprocesses audio file and runs inference via requested backend (default: 'onnx')."""
        if not audio_path:
            return {}

        features = load_and_preprocess_audio(audio_path, device=self.device)
        
        # Strict shape validation for 3 channels (v2.0.0)
        assert features.shape[1] == 3, f"[v2.0.0 Error] Expected 3-channel feature input (1, 3, N_MELS, T), got shape {features.shape}"

        backend = str(backend).lower()

        if "onnx" in backend:
            self._load_onnx()
            ort_inputs = {self.input_name: features.cpu().numpy()}
            ort_outputs = self.ort_session.run(None, ort_inputs)[0]
            probs = F.softmax(torch.from_numpy(ort_outputs), dim=1).squeeze(0)
            
        elif "pth" in backend or "pytorch" in backend:
            self._load_pytorch()
            outputs = self.pytorch_model(features)
            probs = F.softmax(outputs, dim=1).squeeze(0)
            
        else:
            raise ValueError(f"Unsupported backend: {backend}. Choose 'onnx' or 'pth'.")

        return {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
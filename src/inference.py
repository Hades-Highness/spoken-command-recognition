import os
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

from configs.config import MODEL_NAME, MODEL_VERSION, NUM_CLASSES, CHECKPOINT_DIR
from src.models import CommandSense
from src.utils import load_and_preprocess_audio

CLASS_NAMES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

class CommandSenseInferencer:
    """Wrapper class for running predictions using either ONNX Runtime or PyTorch."""
    def __init__(self, model_dir=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build candidate folder paths (e.g. checkpoints/model_v1.0 or ../checkpoints/model_v1.0)
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

        # Lazy loading sessions
        self.ort_session = None
        self.pytorch_model = None

    def _load_onnx(self):
        """Loads ONNX Runtime session if not already loaded."""
        if self.ort_session is None:
            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(f"ONNX model file not found at {self.onnx_path}")
            
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"[+] Loaded ONNX runtime session -> {self.onnx_path}")

    def _load_pytorch(self):
        """Loads PyTorch model checkpoint if not already loaded."""
        if self.pytorch_model is None:
            if not os.path.exists(self.pth_path):
                raise FileNotFoundError(f"PyTorch model checkpoint not found at {self.pth_path}")
            
            self.pytorch_model = CommandSense(num_classes=NUM_CLASSES, in_channels=1).to(self.device)
            self.pytorch_model.load_state_dict(torch.load(self.pth_path, map_location=self.device))
            self.pytorch_model.eval()
            print(f"[+] Loaded PyTorch model -> {self.pth_path}")

    @torch.no_grad()
    def predict(self, audio_path, backend="onnx"):
        """Preprocesses audio file and runs inference via requested backend (default: 'onnx')."""
        if not audio_path:
            return {}

        features = load_and_preprocess_audio(audio_path, device=self.device)
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

        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
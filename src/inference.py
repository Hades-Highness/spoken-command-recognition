import os
import sys
import json
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import MODEL_NAME, MODEL_VERSION, CHECKPOINT_DIR, SAMPLE_RATE, NUM_CLASSES
from src.models import CommandSense, AudioFeatureExtractor


class CommandInferencer:
    def __init__(self, checkpoint_path: str = None, labels_path: str = None, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"model_{MODEL_VERSION.lower()}",
                f"{MODEL_NAME}_{MODEL_VERSION}.pth"
            )

        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.pth")

        # Resolve labels path
        if labels_path is None:
            labels_path = os.path.join("configs", "labels.json")

        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            self.labels = [f"Class_{i}" for i in range(NUM_CLASSES)]

        # Initialize feature extractor and 3-channel ResNet model
        self.feature_extractor = AudioFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

        self.model = CommandSense(num_classes=len(self.labels), in_channels=3).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def _load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Safely loads and standardizes audio to 16kHz mono (1.0 sec length)."""
        data, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(data)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T

        # Resample to 16kHz if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Standardize duration to exactly 1 second (16000 samples)
        target_len = SAMPLE_RATE
        current_len = waveform.shape[1]

        if current_len < target_len:
            waveform = F.pad(waveform, (0, target_len - current_len))
        elif current_len > target_len:
            waveform = waveform[:, :target_len]

        return waveform

    def predict(self, audio_path: str) -> dict:
        """Extracts 3-channel features and returns predicted command with confidence."""
        waveform = self._load_and_preprocess_audio(audio_path).to(self.device)

        with torch.no_grad():
            # Generate 3-channel features: [1, 3, 64, 63]
            features = self.feature_extractor(waveform)
            logits = self.model(features)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

            top_prob, top_idx = torch.max(probabilities, dim=0)
            predicted_label = self.labels[top_idx.item()]

        return {
            "label": predicted_label,
            "confidence": float(top_prob.cpu().numpy()),
            "probabilities": {
                label: float(prob.cpu().numpy())
                for label, prob in zip(self.labels, probabilities)
            }
        }


if __name__ == "__main__":
    # Quick sanity check example
    print("[*] CommandInferencer module ready.")
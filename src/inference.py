import torch
import torch.nn.functional as F
from configs.config import NUM_CLASSES
from src.models import CommandSense
from src.utils import load_and_preprocess_audio

CLASS_NAMES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

class CommandSenseInferencer:
    """Wrapper class for loading model checkpoints and running predictions."""
    def __init__(self, checkpoint_path="checkpoints/CommandSense_v1.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CommandSense(num_classes=NUM_CLASSES, in_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path):
        """Preprocesses audio file and returns class probability dictionary."""
        if not audio_path:
            return {}

        features = load_and_preprocess_audio(audio_path, device=self.device)
        outputs = self.model(features)
        probs = F.softmax(outputs, dim=1).squeeze(0)

        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
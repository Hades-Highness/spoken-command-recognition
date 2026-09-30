import torch
from pathlib import Path

# Chemins d'accès
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ONNX_DIR = BASE_DIR / "onnx"

# Audio Settings
SAMPLE_RATE = 16000
DURATION = 1.0 
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Configuration Entraînement
BATCH_SIZE = 256
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modèle & Versioning
MODEL_NAME = "CommandSense"
MODEL_VERSION = "v2.0"
NUM_CLASSES = 35
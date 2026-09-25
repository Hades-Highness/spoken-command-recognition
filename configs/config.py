import torch
from pathlib import Path

# Chemins d'accès
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ONNX_DIR = BASE_DIR / "onnx"

# Paramètres Audio (Google Speech Commands v2)
SAMPLE_RATE = 16000
DURATION = 1.0  # en secondes
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16000 samples exacts

# Spectrogramme Log-Mel (Donne un shape fixe de [1, 64, 63])
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Configuration Entraînement
BATCH_SIZE = 256
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modèle & Versioning
MODEL_NAME = "CommandSense"
MODEL_VERSION = "v1.0"
NUM_CLASSES = 35
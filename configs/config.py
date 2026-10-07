import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ONNX_DIR = BASE_DIR / "onnx"          # optional: standalone .onnx files
REPORTS_DIR = BASE_DIR / "reports"
CONFIG_DIR = BASE_DIR / "configs"
LABELS_PATH = CONFIG_DIR / "labels.json"

# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)   # 16000 samples = 1.0 s

# Log-Mel front-end. n_fft=1024 / hop_length=256 over 1.0 s gives 63 frames,
# so the 3-channel tensor is [batch, 3, 64, 63].
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
# The whole dataset is cached in RAM as spectrograms, so worker processes bring
# no I/O benefit and would duplicate the cache on platforms that spawn instead
# of fork (Windows, macOS). Keep this at 0 unless you know why you need it.
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model & versioning
# ---------------------------------------------------------------------------
MODEL_NAME = "CommandSense"
MODEL_VERSION = "v2.1"
NUM_CLASSES = 37


def checkpoint_dir(version: str = MODEL_VERSION) -> Path:
    return CHECKPOINT_DIR / f"model_{version.lower()}"


def reports_dir(version: str = MODEL_VERSION) -> Path:
    return REPORTS_DIR / f"model_{version.lower()}"

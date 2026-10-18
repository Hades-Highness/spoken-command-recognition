import torch
from pathlib import Path

MODEL_NAME = "CommandSense"
MODEL_VERSION = "v4.0"
NUM_CLASSES = 37

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ONNX_DIR = BASE_DIR / "onnx"          
REPORTS_DIR = BASE_DIR / "reports"
CONFIG_DIR = BASE_DIR / "configs"
LABELS_PATH = CONFIG_DIR / "labels.json"
CALIBRATION_DIR = CONFIG_DIR / "calibration"

SAMPLE_RATE = 16000
DURATION = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 256
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SILENCE_LABEL = "_silence_"
SILENCE_CLASS_INDEX = 0

PITCH_SHIFT_MODE = "spectral"             # "spectral" (fast) | "vocoder" | "off"
PITCH_SHIFT_PROB = 0.3
PITCH_SHIFT_SEMITONES = (-1, 1)           # 0 is excluded
PITCH_SHIFT_REFERENCE_HZ = 1000.0         # semitone -> STFT-bin anchor (spectral)
PITCH_SHIFT_N_FFT = 512
PITCH_SHIFT_BINS_PER_OCTAVE = 12
TIME_SHIFT_PROB = 0.5
TIME_SHIFT_RATIO = 0.10                   # max shift, as a share of the 1.0 s clip

# SNR-based noise mixing (v4.0). Noise is mixed into the training waveforms on
# the accelerator, never into a cached int16 clip. '_silence_' is excluded, it
# has to stay pure background noise for the OOD boundary, and LibriSpeech is
# mixed at a much higher SNR than ambient noise so a command always dominates
# the background speech and its label cannot be corrupted.
NOISE_MIX_PROB = 0.5                      # P(noise mixing) per training clip
SNR_AMBIENT_MIN_DB = 5.0                  # '_background_noise_' SNR range
SNR_AMBIENT_MAX_DB = 15.0
SNR_SPEECH_MIN_DB = 15.0                  # LibriSpeech / '_unknown_' SNR range
SNR_SPEECH_MAX_DB = 25.0
NOISE_SPEECH_SOURCE_PROB = 0.5            # share of mixes drawn from LibriSpeech
NOISE_BANK_AMBIENT_CLIPS = 256            # 1.0 s int16 noise clips kept in RAM
NOISE_BANK_SPEECH_CLIPS = 512

FREQ_MASK_PARAM = 8
TIME_MASK_PARAM = 10
N_FREQ_MASKS = 2
N_TIME_MASKS = 2

CALIBRATION_BINS = 15
TEMPERATURE_MAX_ITER = 50
TEMPERATURE_STEPS = 12
TEMPERATURE_LR = 1.0
TEMPERATURE_TOL = 1e-9
TEMPERATURE_GRID = (0.05, 20.0)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.0
CONFIDENCE_THRESHOLD_RANGE = (0.0, 1.0)
CONFIDENCE_THRESHOLD_STEP = 0.01
REJECT_LABEL = "_unknown_"
THRESHOLD_CRITERION = "min_cost"
THRESHOLD_MIN_COVERAGE = 0.90          
REJECTION_CLASSES = ("_silence_", "_unknown_")


def checkpoint_dir(version: str = MODEL_VERSION) -> Path:
    return CHECKPOINT_DIR / f"model_{version.lower()}"


def reports_dir(version: str = MODEL_VERSION) -> Path:
    return REPORTS_DIR / f"model_{version.lower()}"


def ensure_version_dirs(version: str = MODEL_VERSION):
    """Create (and return) the checkpoint and report folders of ``version``."""
    checkpoints = checkpoint_dir(version)
    reports = reports_dir(version)
    checkpoints.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return checkpoints, reports


def calibration_path(version: str = MODEL_VERSION, directory=None) -> Path:
    """Path of the JSON file holding the fitted temperature and threshold"""
    base = Path(directory) if directory is not None else CALIBRATION_DIR
    return base / f"{MODEL_NAME}_calibration_{version}.json"

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
# Calibration payloads (the fitted T and tau) are versioned configuration, not
# weights: they live here so they can be reviewed, diffed and shipped with the
# code instead of sitting in the git-ignored checkpoints/ tree.
CALIBRATION_DIR = CONFIG_DIR / "calibration"

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
EPOCHS = 30
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Conditional augmentation (v3.0): train-only, GPU-side, never on '_silence_'
# ---------------------------------------------------------------------------
# Every augmentation below runs on the accelerator, right after the pinned int16
# batch lands on the GPU, and only while the front-end is in train() mode. The
# '_silence_' clips are always left untouched: they teach the reject (OOD)
# boundary from raw background noise, so perturbing them would only blur it.
SILENCE_LABEL = "_silence_"
# '_silence_' is index 0 because the dataset sorts its label list and '_' sorts
# before every letter (see configs/labels.json). train.py overrides this with the
# dataset's own label_to_idx index, so the two can never drift apart.
SILENCE_CLASS_INDEX = 0

# Waveform level (applied on the normalized [-1, 1] audio).
# Pitch shift is the one augmentation whose cost is worth guarding: the
# torchaudio phase vocoder (AF.pitch_shift) is ~180x more expensive than the
# cheap STFT-domain shift below, and its cost is almost constant per call
# (~3.4 s for 8 clips vs ~16.8 s for 256 on CPU), so lowering the probability
# alone cannot keep an epoch inside budget. "spectral" is therefore the default.
PITCH_SHIFT_MODE = "spectral"             # "spectral" (fast) | "vocoder" | "off"
PITCH_SHIFT_PROB = 0.3                    # per-clip probability of a shift
PITCH_SHIFT_SEMITONES = (-1, 1)           # allowed shifts (0 excluded)
PITCH_SHIFT_REFERENCE_HZ = 1000.0         # semitone -> STFT-bin anchor (spectral)
PITCH_SHIFT_N_FFT = 512                   # analysis window (and bin spacing)
PITCH_SHIFT_BINS_PER_OCTAVE = 12          # 12 -> n_steps counts semitones
TIME_SHIFT_PROB = 0.5                     # per-batch probability of a roll
TIME_SHIFT_RATIO = 0.10                   # max shift, as a share of the 1.0 s clip

# Spectral level (SpecAugment, Park et al., 2019). Widths are drawn per clip and
# per call, so no two epochs mask the same bins. The values stay deliberately mild
# for 1.0 s commands: 64 mel bins / 63 frames leave little room for large holes.
FREQ_MASK_PARAM = 8
TIME_MASK_PARAM = 10
N_FREQ_MASKS = 2
N_TIME_MASKS = 2

# ---------------------------------------------------------------------------
# Model & versioning
# ---------------------------------------------------------------------------
MODEL_NAME = "CommandSense"
MODEL_VERSION = "v4.0"
NUM_CLASSES = 37

# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (v4.0)
# ---------------------------------------------------------------------------
# Architecture switch for the SE-ResNet introduced in v4.0. USE_SE = True trains
# the channel-recalibrated graph; USE_SE = False reproduces the exact v3.0
# network, so a quick A/B run is a one-line config flip. SE_REDUCTION is the
# bottleneck ratio of every SE block (hidden width = C // SE_REDUCTION); it is
# kept at 8 to stay well under 5% of the v3.0 parameter count.
USE_SE = True
SE_REDUCTION = 8

# ---------------------------------------------------------------------------
# Confidence calibration (T + tau)
# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al., 2017) divides the logits by a single scalar
# T > 0 fitted on the held-out validation split. The confidence threshold tau
# then rejects the predictions the calibrated model is not sure about.
CALIBRATION_BINS = 15                  # M bins for the ECE and reliability diagram
TEMPERATURE_MAX_ITER = 50              # inner LBFGS evaluations per step
TEMPERATURE_STEPS = 12                 # outer LBFGS steps (one step per call)
TEMPERATURE_LR = 1.0                   # initial LBFGS step size
TEMPERATURE_TOL = 1e-9                 # stop when the NLL stops improving
TEMPERATURE_GRID = (0.05, 20.0)        # (min, max) clamp for the fitted T
# Safety fallbacks used when no calibration file can be read (absent, unreadable
# or holding values the runtime cannot apply). T = 1.0 serves the raw logits
# unchanged and tau = 0.0 can never reject a clip, so a missing
# CommandSense_calibration_<MODEL_VERSION>.json degrades to the uncalibrated
# decision rule instead of silently dropping predictions or crashing the server.
DEFAULT_TEMPERATURE = 1.0              # T used when no calibration file exists
DEFAULT_CONFIDENCE_THRESHOLD = 0.0     # tau used when no calibration file exists
CONFIDENCE_THRESHOLD_RANGE = (0.0, 1.0)
CONFIDENCE_THRESHOLD_STEP = 0.01       # sweep resolution for tau
# Label returned when max softmax < tau. '_unknown_' is the project's existing
# rejection class, so a low-confidence rejection is expressed in the same
# vocabulary as the learned rejection.
REJECT_LABEL = "_unknown_"
# Selection criterion for the optimal tau (see src/calibration.select_threshold):
# 'min_cost' balances the false rejection rate against the false acceptance rate.
THRESHOLD_CRITERION = "min_cost"
THRESHOLD_MIN_COVERAGE = 0.90          # tau* must still answer this share of clips
REJECTION_CLASSES = ("_silence_", "_unknown_")


def checkpoint_dir(version: str = MODEL_VERSION) -> Path:
    return CHECKPOINT_DIR / f"model_{version.lower()}"


def reports_dir(version: str = MODEL_VERSION) -> Path:
    return REPORTS_DIR / f"model_{version.lower()}"


def ensure_version_dirs(version: str = MODEL_VERSION):
    """Create (and return) the checkpoint and report folders of ``version``.

    The folders do not exist until the first artifact of that version is
    written, so callers can ask for them unconditionally.
    """
    checkpoints = checkpoint_dir(version)
    reports = reports_dir(version)
    checkpoints.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return checkpoints, reports


def calibration_path(version: str = MODEL_VERSION, directory=None) -> Path:
    """Path of the JSON file holding the fitted temperature and threshold.

    Defaults to ``configs/calibration/``: the payload is a few hundred bytes of
    versioned configuration, so it is committed with the code and reviewed in a
    diff instead of sitting in the git-ignored ``checkpoints/`` tree. Pass
    ``directory`` to write it elsewhere, e.g. a scratch folder for a run whose
    result must never be served.
    """
    base = Path(directory) if directory is not None else CALIBRATION_DIR
    return base / f"{MODEL_NAME}_calibration_{version}.json"

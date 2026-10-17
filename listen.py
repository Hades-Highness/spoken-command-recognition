"""Real-time microphone keyword spotting for the configured release.

Run with:  python listen.py

The script captures 16 kHz mono float32 audio with ``sounddevice``, keeps it in a
ring buffer and classifies one 1.0 s window every 0.5 s (50 % overlap). The
artifact paths are never written down here: ``src.inference.CommandInferencer``
resolves the ONNX graph, the labels and the calibration payload of the
``MODEL_VERSION`` selected in ``configs/config.py``, so this file does not need
to know whether that release ships an INT8 or a float32 graph.

Preprocessing matches training exactly: the captured float32 signal is clamped
and quantized onto the int16 grid the dataset cache uses (scale 32767), then fed
as float32 to the shared ``AudioToThreeChannelMel`` transform, whose normalizer
is 32768. The transform runs in ``eval()`` mode, so no augmentation is applied,
and its output is checked against ``[1, 3, 64, 63]`` before inference.

Calibration follows the deployment contract: the graph returns raw logits and
``softmax(logits / T)`` is applied in Python. A window is printed as a command
only when its top-1 probability reaches ``tau``, its label is not one of the
reject classes, and the 1.5 s cooldown that follows a detection has elapsed.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from configs.config import (  # noqa: E402
    HOP_LENGTH,
    MODEL_NAME,
    MODEL_VERSION,
    N_MELS,
    REJECTION_CLASSES,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    calibration_path,
)
from src import calibration  # noqa: E402
from src.inference import (  # noqa: E402
    RELEASES_URL,
    CommandInferencer,
    ModelUnavailableError,
)
from src.transforms import AudioToThreeChannelMel  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:  # reported by setup_runtime() with an actionable message
    ort = None

try:
    import sounddevice as sd
except ImportError:  # reported by open_microphone() with an actionable message
    sd = None


# Sliding window: 1.0 s frames, 0.5 s stride (50 % overlap).
WINDOW_SAMPLES = TARGET_SAMPLES
HOP_SAMPLES = TARGET_SAMPLES // 2
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE

# A detection is reported at most once per cooldown, so consecutive overlapping
# windows that hear the same word cannot spam the console.
COOLDOWN_SECONDS = 1.5

# 100 ms capture blocks keep the callback cheap; the ring holds four seconds, so
# a slow consumer can fall three seconds behind before a window is overwritten.
BLOCK_SAMPLES = SAMPLE_RATE // 10
RING_SECONDS = 4.0
RING_CAPACITY = int(round(SAMPLE_RATE * RING_SECONDS))
POLL_SECONDS = 0.02

# Console status line: ASCII spinner, redrawn in place, padded to a fixed width.
STATUS_INTERVAL = 0.12
STATUS_WIDTH = 78
SPINNER = "|/-\\"

# Training caches int16 PCM obtained with round(clamp(audio) * 32767); the shared
# transform divides by 32768. The microphone signal is put on the same grid.
INT16_SCALE = 32767.0
EXPECTED_CHANNELS = 3
EXPECTED_FRAMES = TARGET_SAMPLES // HOP_LENGTH + 1  # 63 frames for 1.0 s
EXPECTED_SHAPE = (1, EXPECTED_CHANNELS, N_MELS, EXPECTED_FRAMES)


class SetupError(RuntimeError):
    """Raised when a required artifact, backend or device is unavailable."""


class StatusLine:
    """Single-line ``Listening...`` status with an ASCII spinner.

    When stdout is redirected, the animated line is suppressed so a log file is
    not filled with carriage returns; detected commands are still printed.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._frame = 0
        self._last_render = 0.0
        self._visible = False

    @property
    def interactive(self) -> bool:
        return bool(self._stream.isatty())

    def update(self, detail: str = "", force: bool = False) -> None:
        if not self.interactive:
            return
        now = time.monotonic()
        if not force and now - self._last_render < STATUS_INTERVAL:
            return
        self._last_render = now

        frame = SPINNER[self._frame % len(SPINNER)]
        self._frame += 1
        text = f"Listening... {frame}"
        if detail:
            text = f"{text}  {detail}"
        self._stream.write("\r" + text[:STATUS_WIDTH].ljust(STATUS_WIDTH))
        self._stream.flush()
        self._visible = True

    def clear(self) -> None:
        if self._visible and self.interactive:
            self._stream.write("\r" + " " * STATUS_WIDTH + "\r")
            self._stream.flush()
        self._visible = False

    def announce(self, message: str) -> None:
        self.clear()
        self._stream.write(message + "\n")
        self._stream.flush()


class AudioRingBuffer:
    """Thread-safe float32 ring buffer addressed by absolute sample index."""

    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive.")
        self._capacity = int(capacity_samples)
        self._data = np.zeros(self._capacity, dtype=np.float32)
        self._write_index = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def write(self, samples: np.ndarray) -> None:
        """Append ``samples``; the oldest audio is overwritten when full."""
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        if block.size >= self._capacity:
            block = block[-self._capacity:]

        with self._lock:
            count = block.size
            first = min(count, self._capacity - self._write_index)
            self._data[self._write_index:self._write_index + first] = block[:first]
            remaining = count - first
            if remaining:
                self._data[:remaining] = block[first:]
            self._write_index = (self._write_index + count) % self._capacity
            self._total_written += count

    def read(self, end: int, count: int) -> np.ndarray:
        """Return samples ``[end - count, end)``, or ``None`` if overwritten."""
        if count <= 0 or count > self._capacity:
            raise ValueError("count must be in [1, capacity].")
        with self._lock:
            total = self._total_written
            start = int(end) - int(count)
            if end > total or start < 0:
                return None
            if start < total - self._capacity:
                return None

            begin = start % self._capacity
            if begin + count <= self._capacity:
                return self._data[begin:begin + count].copy()

            first = self._capacity - begin
            window = np.empty(count, dtype=np.float32)
            window[:first] = self._data[begin:]
            window[first:] = self._data[:count - first]
            return window


class FeatureFrontend:
    """Microphone float32 -> ``[1, 3, N_MELS, 63]`` float32 features.

    Reuses the exact ``AudioToThreeChannelMel`` deployed by training, validation
    and ONNX export. The module is kept in ``eval()`` mode, which disables both
    waveform augmentation and SpecAugment.
    """

    def __init__(self):
        self.extractor = AudioToThreeChannelMel().eval()

    def __call__(self, window: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(window, dtype=np.float32).reshape(-1)
        if samples.size != WINDOW_SAMPLES:
            raise ValueError(
                f"Expected {WINDOW_SAMPLES} samples (1.0 s at {SAMPLE_RATE} Hz), "
                f"got {samples.size}."
            )

        # Dataset parity: quantize to the int16 grid before the transform, which
        # normalizes by 32768 exactly as it does for the raw-audio cache.
        waveform = torch.from_numpy(samples).clamp(-1.0, 1.0) * INT16_SCALE
        waveform = torch.round(waveform).to(torch.int16).to(torch.float32)
        waveform = waveform.view(1, 1, WINDOW_SAMPLES)

        with torch.no_grad():
            features = self.extractor(waveform)

        features = features.detach().cpu().to(torch.float32).contiguous().numpy()
        if features.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"Feature shape {tuple(features.shape)} does not match the "
                f"expected {EXPECTED_SHAPE}."
            )
        return features


class OnnxKeywordSpotter:
    """CPU-only ONNX Runtime session with temperature scaling and rejection."""

    def __init__(self, onnx_path, labels, temperature: float, threshold: float):
        if ort is None:
            raise SetupError(
                "[ERROR] onnxruntime is not installed. Install it with "
                "'pip install onnxruntime'."
            )

        try:
            self.session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise SetupError(
                f"[ERROR] Could not load the ONNX graph '{onnx_path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self.labels = list(labels)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self._validate_io(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name

    def _validate_io(self, onnx_path: str) -> None:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' must expose exactly one input, "
                f"found {len(inputs)}."
            )

        declared = inputs[0].shape
        if len(declared) != 4:
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' has an unexpected input rank "
                f"{declared}; expected [batch, 3, {N_MELS}, {EXPECTED_FRAMES}]."
            )
        if declared[1] not in (EXPECTED_CHANNELS, str(EXPECTED_CHANNELS)):
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' declares {declared[1]} input "
                f"channels; this pipeline builds {EXPECTED_CHANNELS}."
            )
        if isinstance(declared[3], int) and declared[3] != EXPECTED_FRAMES:
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' expects {declared[3]} time "
                f"frames; the front-end produces {EXPECTED_FRAMES}."
            )

        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' must expose exactly one output, "
                f"found {len(outputs)}."
            )
        classes = outputs[0].shape[-1]
        if isinstance(classes, int) and classes != len(self.labels):
            raise SetupError(
                f"[ERROR] ONNX graph '{onnx_path}' emits {classes} logits but "
                f"{len(self.labels)} labels were loaded."
            )

    def predict(self, features: np.ndarray):
        """Return ``(label, calibrated_confidence)`` for one feature window."""
        logits = self.session.run(None, {self.input_name: features})[0]
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits.size != len(self.labels):
            raise RuntimeError(
                f"Model returned {logits.size} logits for {len(self.labels)} labels."
            )

        probabilities = calibration.logits_to_probabilities(
            logits, self.temperature
        ).reshape(-1)
        index = int(np.argmax(probabilities))
        return self.labels[index], float(probabilities[index])


def _validate_calibration(payload: dict, path):
    """Return ``(temperature, threshold)`` or raise an actionable error."""
    missing = [
        key for key in ("temperature", "confidence_threshold") if key not in payload
    ]
    if missing:
        raise SetupError(
            f"[ERROR] Calibration file '{path}' is missing "
            f"{', '.join(missing)}. Re-fit it with 'python evaluate.py'."
        )

    try:
        temperature = float(payload["temperature"])
        threshold = float(payload["confidence_threshold"])
    except (TypeError, ValueError) as exc:
        raise SetupError(
            f"[ERROR] Calibration file '{path}' holds non-numeric values: {exc}."
        ) from exc

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise SetupError(
            f"[ERROR] Calibration temperature must be a finite number > 0, "
            f"got {temperature} in '{path}'."
        )
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SetupError(
            f"[ERROR] Calibration threshold must be a probability in [0, 1], "
            f"got {threshold} in '{path}'."
        )
    return temperature, threshold


def setup_runtime():
    """Resolve the configured release and build the front-end and ONNX engine."""
    try:
        engine = CommandInferencer(version=MODEL_VERSION, device="cpu")
    except ModelUnavailableError as exc:
        raise SetupError(
            f"[ERROR] Model {MODEL_NAME} {MODEL_VERSION} was not found. "
            f"Download the release bundle from the GitHub Releases page "
            f"({RELEASES_URL}) or generate it locally with "
            "'python export_onnx.py' (then 'python quantize_int8.py')."
        ) from exc
    except Exception as exc:
        raise SetupError(
            f"[ERROR] Could not initialise {MODEL_NAME} {MODEL_VERSION}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if engine.onnx_path is None:
        raise SetupError(
            f"[ERROR] No ONNX graph was found for {MODEL_NAME} {MODEL_VERSION}. "
            f"Download the release bundle from the GitHub Releases page "
            f"({RELEASES_URL}) or generate it locally with "
            "'python export_onnx.py' (then 'python quantize_int8.py')."
        )

    calibration_file = calibration.resolve_calibration_file(MODEL_VERSION)
    if calibration_file is None:
        raise SetupError(
            f"[ERROR] The calibration file for {MODEL_NAME} {MODEL_VERSION} was "
            f"not found. Expected '{calibration_path(MODEL_VERSION)}' or a copy "
            "next to the checkpoints. Fit it with 'python evaluate.py' or "
            f"download the release bundle from the GitHub Releases page "
            f"({RELEASES_URL})."
        )

    payload = calibration.load_calibration(calibration_file)
    if not payload:
        raise SetupError(
            f"[ERROR] Calibration file '{calibration_file}' could not be read or "
            "is not valid JSON. Re-fit it with 'python evaluate.py' or download "
            f"the release bundle from the GitHub Releases page ({RELEASES_URL})."
        )
    temperature, threshold = _validate_calibration(payload, calibration_file)

    frontend = FeatureFrontend()
    spotter = OnnxKeywordSpotter(
        engine.onnx_path, engine.labels, temperature, threshold
    )
    info = {
        "model_path": engine.onnx_path,
        "calibration_path": str(calibration_file),
        "temperature": temperature,
        "threshold": threshold,
        "labels": len(engine.labels),
    }
    return frontend, spotter, info


def open_microphone(buffer: AudioRingBuffer, callback_status: list, device=None):
    """Open and start the 16 kHz mono float32 input stream."""
    if sd is None:
        raise SetupError(
            "[ERROR] sounddevice is not installed. Install the microphone "
            "dependency with 'pip install sounddevice'."
        )

    def callback(indata, frames, time_info, status):
        # The callback runs on the audio thread: never print or raise here.
        if status:
            callback_status[0] = str(status)
        buffer.write(indata[:, 0])

    stream = None
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            device=device,
            callback=callback,
        )
        stream.start()
    except Exception as exc:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        raise SetupError(
            f"[ERROR] Could not open a {SAMPLE_RATE} Hz mono input stream: "
            f"{type(exc).__name__}: {exc}. Check the microphone, or run "
            "'python listen.py --list-devices' and pass '--device <index>'."
        ) from exc
    return stream


def list_input_devices() -> int:
    """Print the devices that can capture audio."""
    if sd is None:
        print(
            "[ERROR] sounddevice is not installed. Install it with "
            "'pip install sounddevice'.",
            file=sys.stderr,
        )
        return 1

    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(
            f"[ERROR] Could not query audio devices: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Input devices:")
    found = False
    for index, info in enumerate(devices):
        channels = int(info.get("max_input_channels", 0))
        if channels <= 0:
            continue
        found = True
        rate = int(round(float(info.get("default_samplerate", 0) or 0)))
        print(f"  [{index}] {info.get('name', 'unknown')} ({channels} ch, {rate} Hz)")
    if not found:
        print("  (none)")
    return 0


def listen(device=None) -> int:
    """Run the capture, inference and status loop until Ctrl+C."""
    if sd is None:
        print(
            "[ERROR] sounddevice is not installed. Install the microphone "
            "dependency with 'pip install sounddevice'.",
            file=sys.stderr,
        )
        return 1

    status = StatusLine()

    try:
        frontend, spotter, info = setup_runtime()
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[+] {MODEL_NAME} {MODEL_VERSION}")
    print(f"[+] ONNX graph        : {info['model_path']}")
    print(f"[+] Calibration       : {info['calibration_path']}")
    print(f"[+] Temperature (T)   : {info['temperature']:.4f}")
    print(f"[+] Threshold (tau)   : {info['threshold']:.2f}")
    print(f"[+] Classes           : {info['labels']}")
    print(f"[+] Window / stride   : {WINDOW_SECONDS:.1f} s / {HOP_SECONDS:.1f} s")
    print(f"[+] Rejection classes : {', '.join(REJECTION_CLASSES)}")

    try:
        started = time.perf_counter()
        spotter.predict(frontend(np.zeros(WINDOW_SAMPLES, dtype=np.float32)))
        warmup_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        print(
            f"[ERROR] Warm-up inference failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"[+] Warm-up           : {warmup_ms:.1f} ms (front-end + ONNX CPU)")

    buffer = AudioRingBuffer(RING_CAPACITY)
    callback_status = [None]
    try:
        stream = open_microphone(buffer, callback_status, device=device)
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[*] Listening at {SAMPLE_RATE} Hz mono. Press Ctrl+C to stop.")
    status.update("")

    next_end = WINDOW_SAMPLES
    cooldown_until = 0.0
    last_detail = ""
    last_stream_warning = 0.0

    try:
        with stream:
            while True:
                stream_status = callback_status[0]
                if stream_status:
                    callback_status[0] = None
                    now = time.monotonic()
                    if now - last_stream_warning >= 5.0:
                        status.announce(f"[WARN] Audio stream status: {stream_status}")
                        last_stream_warning = now

                total = buffer.total_written
                now = time.monotonic()

                if total < next_end:
                    if now < cooldown_until:
                        status.update(f"cooldown {cooldown_until - now:.1f}s")
                    else:
                        status.update(last_detail)
                    time.sleep(POLL_SECONDS)
                    continue

                # If the consumer fell further behind than the ring can hold, the
                # pending windows are already overwritten: resynchronise instead
                # of feeding the model discontinuous audio.
                if total - next_end > RING_CAPACITY - WINDOW_SAMPLES:
                    lag_seconds = (total - next_end) / SAMPLE_RATE
                    next_end = total
                    status.announce(
                        f"[WARN] Fell behind by {lag_seconds:.1f} s; "
                        "skipped to the live audio."
                    )

                window = buffer.read(next_end, WINDOW_SAMPLES)
                if window is None:
                    next_end = total
                    time.sleep(POLL_SECONDS)
                    continue
                next_end += HOP_SAMPLES

                try:
                    label, confidence = spotter.predict(frontend(window))
                except Exception as exc:
                    status.announce(
                        f"[ERROR] Inference failed: {type(exc).__name__}: {exc}"
                    )
                    return 1

                now = time.monotonic()
                if confidence < spotter.threshold:
                    last_detail = (
                        f"rejected: {label} {confidence:.2f} < "
                        f"tau {spotter.threshold:.2f}"
                    )
                elif label in REJECTION_CLASSES:
                    last_detail = f"idle: {label} {confidence:.2f}"
                elif now < cooldown_until:
                    last_detail = f"cooldown {cooldown_until - now:.1f}s"
                else:
                    status.announce(f"[DETECTED] {label} (conf: {confidence:.2f})")
                    cooldown_until = now + COOLDOWN_SECONDS
                    last_detail = ""
                    continue

                status.update(last_detail, force=True)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        status.announce(
            f"[ERROR] Audio capture failed: {type(exc).__name__}: {exc}"
        )
        return 1
    finally:
        status.clear()

    print("[*] Stopped listening.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time keyword spotting with the configured CommandSense release."
    )
    parser.add_argument(
        "--device",
        help="Input device index or name; see --list-devices.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List the available audio input devices and exit.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        return list_input_devices()

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return listen(device=device)


if __name__ == "__main__":
    sys.exit(main())







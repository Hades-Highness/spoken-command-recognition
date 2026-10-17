"""Reusable real-time keyword spotting for the configured release.

:class:`KeywordSpotter` owns one microphone session and runs capture, feature
extraction, ONNX inference and cooldown in a background thread. The Gradio
interface in ``app.py`` consumes it through ``start``, ``stop`` and ``poll``,
and the class can also be embedded directly::

    spotter = KeywordSpotter(MODEL_VERSION)
    spotter.start()
    try:
        while True:
            for detection in spotter.poll():
                print(detection.label, detection.confidence)
            time.sleep(0.1)
    finally:
        spotter.stop()

:class:`SharedRuntime` resolves the labels, ONNX graph and calibration payload
through ``src.inference.CommandInferencer`` and caches one runtime per release,
so this module never names an artifact path. Preprocessing mirrors training:
the signal is clamped and quantized onto the int16 grid, then passed to the
shared ``AudioToThreeChannelMel`` transform in evaluation mode. The graph
returns raw logits; ``softmax(logits / T)`` is applied in Python. A window is
published only when its top-1 probability reaches ``tau``, its label is not a
rejection class, and the cooldown has elapsed.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
)
from src.inference import (  # noqa: E402
    RELEASES_URL,
    CommandInferencer,
    ModelUnavailableError,
)
from src.transforms import AudioToThreeChannelMel  # noqa: E402

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
# windows that hear the same word cannot spam the detection history.
COOLDOWN_SECONDS = 1.5

# 100 ms capture blocks keep the callback cheap; the ring holds four seconds, so
# a slow consumer can fall three seconds behind before a window is overwritten.
BLOCK_SAMPLES = SAMPLE_RATE // 10
RING_SECONDS = 4.0
RING_CAPACITY = int(round(SAMPLE_RATE * RING_SECONDS))
POLL_SECONDS = 0.02

# Training caches int16 PCM obtained with round(clamp(audio) * 32767); the shared
# transform divides by 32768. The microphone signal is put on the same grid.
INT16_SCALE = 32767.0
EXPECTED_CHANNELS = 3
EXPECTED_FRAMES = TARGET_SAMPLES // HOP_LENGTH + 1  # 63 frames for 1.0 s
EXPECTED_SHAPE = (1, EXPECTED_CHANNELS, N_MELS, EXPECTED_FRAMES)


class SetupError(RuntimeError):
    """Raised when a required artifact, backend or device is unavailable."""


@dataclass(frozen=True)
class Detection:
    """One accepted command emitted by :class:`KeywordSpotter`."""

    label: str
    confidence: float
    timestamp: float

    @property
    def time_text(self) -> str:
        """Local wall-clock time of the detection, formatted ``HH:MM:SS``."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation for a polling UI."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "time": self.time_text,
        }


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

    def read(self, end: int, count: int) -> Optional[np.ndarray]:
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
    """Training-identical waveform to ``[1, 3, 64, 63]`` mel conversion."""

    def __init__(self):
        self._transform = AudioToThreeChannelMel(n_mels=N_MELS)
        self._transform.eval()
        if hasattr(self._transform, "augment"):
            self._transform.augment = False

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size != WINDOW_SAMPLES:
            raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {samples.size}.")
        if not np.isfinite(samples).all():
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = np.clip(samples, -1.0, 1.0)
        samples = np.round(samples * INT16_SCALE).astype(np.float32)

        with torch.inference_mode():
            tensor = torch.from_numpy(samples.copy()).unsqueeze(0)
            features = self._transform(tensor)
            batch = features.detach().cpu().numpy().astype(np.float32)

        if batch.shape != EXPECTED_SHAPE:
            raise SetupError(
                f"Unexpected input shape {tuple(batch.shape)}; the model expects {EXPECTED_SHAPE}."
            )
        if not np.isfinite(batch).all():
            raise SetupError(
                "Feature extraction produced non-finite values; the input is unusable."
            )
        return np.ascontiguousarray(batch)


class OnnxKeywordSpotter:
    """Temperature-scaled classifier that reuses the shared ONNX session.

    The session comes from :meth:`CommandInferencer.get_onnx_session`, so the
    real-time loop and the single-recording tab execute the very same graph.
    ``session.run`` is serialized by the runtime lock because both tabs can
    classify concurrently.
    """

    def __init__(
        self,
        session,
        input_name: str,
        labels: List[str],
        temperature: float,
        threshold: float,
    ):
        self.session = session
        self.input_name = input_name
        self.labels = list(labels)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self._validate_io()

    def _validate_io(self) -> None:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise SetupError(
                f"The ONNX graph declares {len(inputs)} inputs; exactly one is expected."
            )
        declared = inputs[0].shape
        if len(declared) != 4:
            raise SetupError(f"The ONNX graph declares input shape {declared}; rank 4 is expected.")
        channels = declared[1]
        if isinstance(channels, int) and channels != EXPECTED_CHANNELS:
            raise SetupError(
                f"The ONNX graph declares {channels} input channels; "
                f"{EXPECTED_CHANNELS} are expected."
            )
        frames = declared[3]
        if isinstance(frames, int) and frames != EXPECTED_FRAMES:
            raise SetupError(
                f"The ONNX graph expects {frames} time frames; {EXPECTED_FRAMES} are expected."
            )
        if inputs[0].name != self.input_name:
            raise SetupError(
                f"The ONNX graph input is named '{inputs[0].name}', not '{self.input_name}'."
            )

        outputs = self.session.get_outputs()
        if not outputs:
            raise SetupError("The ONNX graph declares no outputs.")
        classes = outputs[0].shape[-1] if outputs[0].shape else None
        if isinstance(classes, int) and classes != len(self.labels):
            raise SetupError(
                f"The ONNX graph returns {classes} classes but "
                f"{len(self.labels)} labels were loaded."
            )

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Return the top-1 label and its temperature-scaled probability."""
        logits = self.session.run(None, {self.input_name: features})[0]
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if logits.size != len(self.labels):
            raise SetupError(
                f"The ONNX graph returned {logits.size} logits for {len(self.labels)} labels."
            )
        scaled = logits / self.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return self.labels[index], float(probabilities[index])


class SharedRuntime:
    """Artifacts of one release, loaded once and shared by every consumer.

    ``engine`` resolves ``MODEL_VERSION`` and reads the labels and calibration
    payload. The ONNX session is materialised once here; the real-time spotter
    runs on that exact session, and ``inference_lock`` serializes ``session.run``
    across the background loop and the single-recording tab.
    """

    def __init__(self, model_version: str = MODEL_VERSION, device=None):
        self.model_version = str(model_version)
        try:
            self.engine = CommandInferencer(version=self.model_version, device=device)
        except ModelUnavailableError as exc:
            raise SetupError(
                f"[ERROR] Model {MODEL_NAME} {self.model_version} was not found. "
                f"Download the release bundle from the GitHub Releases page ({RELEASES_URL}) "
                "or generate it locally with 'python export_onnx.py' "
                "(then 'python quantize_int8.py')."
            ) from exc
        except Exception as exc:
            raise SetupError(
                f"[ERROR] Could not initialise {MODEL_NAME} {self.model_version}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self.labels = list(self.engine.labels)
        self.temperature = float(self.engine.temperature)
        self.threshold = float(self.engine.confidence_threshold)
        self.calibration_source = self.engine.calibration_source
        self.onnx_path = self.engine.onnx_path
        self.pth_path = self.engine.pth_path
        self.warmup_ms = 0.0
        self.frontend: Optional[FeatureFrontend] = None
        self.spotter: Optional[OnnxKeywordSpotter] = None
        self.inference_lock = threading.Lock()

        if self.onnx_path is not None and "onnx" in self.engine.available_backends:
            self._initialise_onnx()

    def _initialise_onnx(self) -> None:
        try:
            session, input_name = self.engine.get_onnx_session()
            self.frontend = FeatureFrontend()
            self.spotter = OnnxKeywordSpotter(
                session=session,
                input_name=input_name,
                labels=self.labels,
                temperature=self.temperature,
                threshold=self.threshold,
            )
        except SetupError:
            raise
        except Exception as exc:
            raise SetupError(
                f"[ERROR] Could not load the ONNX graph of {MODEL_NAME} "
                f"{self.model_version}: {type(exc).__name__}: {exc}"
            ) from exc

        warmup = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        with self.inference_lock:
            started = time.perf_counter()
            self.spotter.predict(self.frontend(warmup))
            self.warmup_ms = (time.perf_counter() - started) * 1000.0
        print(f"[+] Real-time front-end warmed up in {self.warmup_ms:.1f} ms")

    @property
    def realtime_available(self) -> bool:
        """True when the graph, the front-end and a calibration payload exist."""
        return (
            self.spotter is not None
            and self.frontend is not None
            and self.calibration_source is not None
        )

    def require_realtime(self) -> None:
        """Raise :class:`SetupError` when the real-time tab cannot run."""
        if self.onnx_path is None:
            raise SetupError(
                f"[ERROR] No ONNX graph was found for {MODEL_NAME} {self.model_version}. "
                f"Download the release bundle from the GitHub Releases page ({RELEASES_URL}) "
                "or generate it locally with 'python export_onnx.py' "
                "(then 'python quantize_int8.py')."
            )
        if "onnx" not in self.engine.available_backends:
            raise SetupError(
                "[ERROR] onnxruntime is not installed. Install it with "
                "'pip install onnxruntime' and restart the application."
            )
        if self.calibration_source is None:
            raise SetupError(
                f"[ERROR] The calibration file for {MODEL_NAME} {self.model_version} "
                "was not found. Real-time spotting needs the fitted temperature and "
                "threshold. Fit them with 'python evaluate.py' or download the release "
                f"bundle ({RELEASES_URL})."
            )


_RUNTIMES: Dict[Tuple[str, str], SharedRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_shared_runtime(model_version: str = MODEL_VERSION, device=None) -> SharedRuntime:
    """Return the process-wide runtime of ``model_version``, loading it once.

    Concurrent callers wait for the first load and then receive the same
    instance, so the ONNX session, the mel front-end and the calibration values
    exist exactly once regardless of how many tabs or threads use them.
    """
    key = (str(model_version), str(device))
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = SharedRuntime(model_version=model_version, device=device)
            _RUNTIMES[key] = runtime
    return runtime


def query_input_devices() -> List[dict]:
    """Return the capturable input devices, or ``[]`` when none can be listed.

    ``sounddevice`` reports every host device, including output-only ones, so
    only devices with at least one input channel are kept. A missing PortAudio
    library or missing ``sounddevice`` yields an empty list instead of raising,
    which lets the UI stay usable for single-recording analysis.
    """
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    result = []
    for index, info in enumerate(devices):
        channels = int(info.get("max_input_channels", 0) or 0)
        if channels <= 0:
            continue
        result.append(
            {
                "index": int(index),
                "name": str(info.get("name", f"Device {index}")),
                "channels": channels,
                "default_samplerate": float(info.get("default_samplerate", 0.0) or 0.0),
            }
        )
    return result


def open_microphone(
    buffer: AudioRingBuffer,
    status: List[Optional[str]],
    device: Optional[int] = None,
):
    """Open a 16 kHz mono input stream that feeds ``buffer``.

    The callback runs on PortAudio's thread and only converts the block to
    mono float32 before appending it to the ring, so it stays short. Failures
    are stored in ``status[0]`` for the consumer to report instead of being
    raised from the audio thread.
    """
    if sd is None:
        raise SetupError(
            "[ERROR] sounddevice is not installed. Install it with "
            "'pip install sounddevice' and restart the application."
        )

    def _callback(indata, frames, time_info, status_flag):
        if status_flag:
            status[0] = str(status_flag)
        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 2:
            block = block.mean(axis=1)
        buffer.write(block)

    try:
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            channels=1,
            dtype="float32",
            callback=_callback,
            device=device,
        )
    except Exception as exc:
        raise SetupError(
            f"[ERROR] Could not open the microphone: {type(exc).__name__}: {exc}. "
            "Check the operating-system microphone permissions or pick another "
            "input device."
        ) from exc


class KeywordSpotter:
    """Drive one microphone session against the shared release.

    Parameters
    ----------
    model_version:
        Release to serve; defaults to ``MODEL_VERSION`` from
        ``configs/config.py``. The matching ONNX graph, labels and calibration
        payload are resolved by :class:`SharedRuntime`.
    device:
        ``sounddevice`` input-device index, or ``None`` for the system default.
        Indices are the ones returned by :func:`query_input_devices`.

    The constructor validates that the release can serve the real-time tab and
    raises :class:`SetupError` otherwise; the microphone is opened by
    :meth:`start`, so no device is held while the UI is idle.
    """

    def __init__(self, model_version: str = MODEL_VERSION, device: Optional[int] = None):
        self.model_version = str(model_version)
        self.device = device
        self._runtime = get_shared_runtime(self.model_version)
        self._runtime.require_realtime()

        self._buffer: Optional[AudioRingBuffer] = None
        self._stream = None
        self._stream_status: List[Optional[str]] = [None]
        self._stop_event = threading.Event()
        self._detections: "queue.Queue[Detection]" = queue.Queue()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._last_window: Optional[dict] = None
        self._last_rejected: Optional[dict] = None
        self._cooldown_until = 0.0
        self._stream_warning: Optional[str] = None

    @property
    def runtime(self) -> SharedRuntime:
        """The shared runtime this spotter classifies with."""
        return self._runtime

    @property
    def labels(self) -> List[str]:
        return list(self._runtime.labels)

    @property
    def temperature(self) -> float:
        return self._runtime.temperature

    @property
    def threshold(self) -> float:
        return self._runtime.threshold

    @property
    def warmup_ms(self) -> float:
        return self._runtime.warmup_ms

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            return thread is not None and thread.is_alive()

    def start(self) -> bool:
        """Open the microphone and begin classifying in a daemon thread.

        Returns ``False`` when the loop is already running. Raises
        :class:`SetupError` when the microphone cannot be opened; in that case
        the spotter stays idle and can be started again after the device or the
        operating-system permissions are fixed.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            buffer = AudioRingBuffer(RING_CAPACITY)
            stream_status: List[Optional[str]] = [None]
            stream = open_microphone(buffer, stream_status, device=self.device)
            try:
                stream.start()
            except Exception as exc:
                self._release_stream(stream)
                raise SetupError(
                    f"[ERROR] Could not start the microphone: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            self._buffer = buffer
            self._stream = stream
            self._stream_status = stream_status
            self._detections = queue.Queue()
            self._stop_event.clear()
            self._error = None
            self._last_window = None
            self._last_rejected = None
            self._cooldown_until = 0.0
            self._stream_warning = None
            self._thread = threading.Thread(
                target=self._run,
                name=f"KeywordSpotter-{self.model_version}",
                daemon=True,
            )
            thread = self._thread

        thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the loop, join it and release the microphone.

        Safe to call repeatedly and from any thread. Returns ``True`` when the
        worker has finished; ``False`` only if it did not join within
        ``timeout`` seconds, in which case the microphone is still released so a
        later ``start`` can open it again.
        """
        with self._lock:
            thread = self._thread
            self._stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        finished = thread is None or not thread.is_alive()

        with self._lock:
            if finished:
                self._thread = None
                self._cooldown_until = 0.0
            stream, self._stream = self._stream, None
            self._buffer = None
            self._stream_status = [None]

        self._release_stream(stream)
        return finished

    def close(self) -> None:
        """Alias of :meth:`stop` for callers that release resources explicitly."""
        self.stop()

    def __enter__(self) -> "KeywordSpotter":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.stop()
        return False

    def poll(self) -> List[Detection]:
        """Return the accepted detections since the previous call.

        Non-blocking: the background loop pushes accepted commands into a
        thread-safe queue and the caller drains it, so the UI never waits on
        inference and no detection is lost between polls.
        """
        detections: List[Detection] = []
        pending = self._detections
        while True:
            try:
                detections.append(pending.get_nowait())
            except queue.Empty:
                break
        return detections

    def status(self) -> dict:
        """Thread-safe snapshot of the loop for a polling UI.

        ``last_window`` describes the most recent classified window (accepted or
        not) and ``last_rejected`` the most recent low-confidence guess, so the
        interface can show current confidence without storing duplicate state.
        """
        with self._lock:
            thread = self._thread
            running = thread is not None and thread.is_alive()
            state = "listening" if running else ("error" if self._error else "idle")
            return {
                "state": state,
                "running": running,
                "error": self._error,
                "last_window": dict(self._last_window) if self._last_window else None,
                "last_rejected": dict(self._last_rejected) if self._last_rejected else None,
                "cooldown_remaining": max(0.0, self._cooldown_until - time.monotonic()),
                "temperature": self._runtime.temperature,
                "threshold": self._runtime.threshold,
                "warmup_ms": self._runtime.warmup_ms,
                "model_name": MODEL_NAME,
                "model_version": self.model_version,
                "stream_warning": self._stream_warning,
            }

    @staticmethod
    def _release_stream(stream) -> None:
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _publish(
        self,
        label: str,
        confidence: float,
        accepted: bool,
        reason: str,
        timestamp: float,
    ) -> None:
        """Store the classified window for ``status()`` (never blocks inference)."""
        record = {
            "label": label,
            "confidence": float(confidence),
            "accepted": bool(accepted),
            "reason": reason,
            "timestamp": timestamp,
        }
        with self._lock:
            self._last_window = record
            if not accepted:
                self._last_rejected = record

    def _run(self) -> None:
        """Classify one 1.0 s window every 0.5 s until ``stop`` is called."""
        runtime = self._runtime
        spotter = runtime.spotter
        frontend = runtime.frontend
        buffer = self._buffer
        stream_status = self._stream_status
        next_end = max(WINDOW_SAMPLES, buffer.total_written)
        cooldown_until = 0.0
        last_stream_warning = 0.0

        try:
            while not self._stop_event.is_set():
                warning = stream_status[0]
                if warning:
                    stream_status[0] = None
                    now = time.monotonic()
                    if now - last_stream_warning >= 5.0:
                        last_stream_warning = now
                        with self._lock:
                            self._stream_warning = str(warning)

                total = buffer.total_written
                if total < next_end:
                    self._stop_event.wait(POLL_SECONDS)
                    continue
                if total - next_end > RING_CAPACITY - WINDOW_SAMPLES:
                    next_end = total

                window = buffer.read(next_end, WINDOW_SAMPLES)
                if window is None:
                    next_end = total
                    self._stop_event.wait(POLL_SECONDS)
                    continue
                next_end += HOP_SAMPLES
                if self._stop_event.is_set():
                    break

                features = frontend(window)
                with runtime.inference_lock:
                    label, confidence = spotter.predict(features)

                now = time.monotonic()
                if confidence < spotter.threshold:
                    reason, accepted = "below_threshold", False
                elif label in REJECTION_CLASSES:
                    reason, accepted = "reject_class", False
                elif now < cooldown_until:
                    reason, accepted = "cooldown", False
                else:
                    reason, accepted = "accepted", True
                    cooldown_until = now + COOLDOWN_SECONDS

                timestamp = time.time()
                with self._lock:
                    self._cooldown_until = cooldown_until
                self._publish(label, confidence, accepted, reason, timestamp)
                if accepted:
                    self._detections.put(
                        Detection(
                            label=label,
                            confidence=float(confidence),
                            timestamp=timestamp,
                        )
                    )
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._cooldown_until = 0.0


def _run_cli(argv: Optional[List[str]] = None) -> int:
    """Optional command-line front-end, kept for scripted microphone checks."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Real-time keyword spotting with {MODEL_NAME} {MODEL_VERSION}."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input-device index (default: system default).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List the input devices that can be captured and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        devices = query_input_devices()
        if not devices:
            print("[!] No capturable input devices were reported by sounddevice.")
            return 1
        for device in devices:
            print(
                f"  [{device['index']}] {device['name']} - "
                f"{device['channels']} ch, "
                f"{device['default_samplerate']:.0f} Hz"
            )
        return 0

    try:
        spotter = KeywordSpotter(MODEL_VERSION, device=args.device)
        spotter.start()
    except SetupError as exc:
        print(exc)
        return 1

    print(
        f"[*] Listening with {MODEL_NAME} {MODEL_VERSION} "
        f"(tau={spotter.threshold:.2f}). Press Ctrl+C to stop."
    )
    try:
        while spotter.is_running:
            for detection in spotter.poll():
                print(
                    f"{detection.time_text}  {detection.label}  "
                    f"{detection.confidence * 100:.2f} %"
                )
            snapshot = spotter.status()
            if snapshot["error"]:
                print(f"[ERROR] {snapshot['error']}")
                break
            time.sleep(POLL_SECONDS * 5)
    except KeyboardInterrupt:
        print("\n[*] Stopping.")
    finally:
        spotter.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())

"""Unified Gradio interface for real-time and one-shot command recognition.

Both tabs are backed by one :class:`listen.SharedRuntime`, built from the release
selected in ``configs/config.py``. The runtime resolves the labels, the ONNX
graph and the calibration payload once at startup; the real-time tab drives a
:class:`listen.KeywordSpotter` per browser session and the single-recording tab
calls ``CommandInferencer.predict`` on the same engine. No artifact path is
hardcoded here.

Run with::

    python app.py

The real-time tab owns a background thread and the microphone, so its lifecycle
is tied to the browser session: ``Blocks.unload`` (tab close or refresh) stops
the spotter, and the polling event detects a loop that stopped on its own. The
``gr.Timer`` that drains detections is only active while a session is listening.
"""

from __future__ import annotations

import atexit
import html
import threading
from typing import Dict, List, Optional

import gradio as gr

from configs.config import (  # noqa: E402
    CONFIDENCE_THRESHOLD_RANGE,
    CONFIDENCE_THRESHOLD_STEP,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    REJECT_LABEL,
    REJECTION_CLASSES,
    SAMPLE_RATE,
)
from listen import (  # noqa: E402
    COOLDOWN_SECONDS,
    HOP_SECONDS,
    WINDOW_SECONDS,
    KeywordSpotter,
    SetupError,
    get_shared_runtime,
    query_input_devices,
)

TIMER_INTERVAL = 0.15
MAX_HISTORY = 100
DEFAULT_DEVICE_LABEL = "Default input device"

try:
    RUNTIME = get_shared_runtime(MODEL_VERSION)
except Exception as exc:  # keep the interface alive so the cause is visible
    RUNTIME = None
    INIT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    INIT_ERROR = None

INITIAL_THRESHOLD = (
    float(RUNTIME.threshold) if RUNTIME is not None else float(DEFAULT_CONFIDENCE_THRESHOLD)
)

BACKEND_KEYS = {"ONNX": "onnx", "PyTorch (.pth)": "pytorch"}
BACKENDS: List[str] = []
if RUNTIME is not None:
    if "onnx" in RUNTIME.engine.available_backends:
        BACKENDS.append("ONNX")
    if RUNTIME.pth_path is not None:
        BACKENDS.append("PyTorch (.pth)")
INITIAL_BACKEND = BACKENDS[0] if BACKENDS else "ONNX"

DEVICES = query_input_devices()
DEVICE_CHOICES = [DEFAULT_DEVICE_LABEL]
DEVICE_INDEX: Dict[str, int] = {}
for _device in DEVICES:
    _label = (
        f"{_device['index']}: {_device['name']} "
        f"({_device['channels']} ch, {_device['default_samplerate']:.0f} Hz)"
    )
    DEVICE_CHOICES.append(_label)
    DEVICE_INDEX[_label] = _device["index"]

_SESSIONS: Dict[str, KeywordSpotter] = {}
_SESSIONS_LOCK = threading.Lock()


def _session_key(request: gr.Request) -> str:
    """Stable per-browser identifier used to key the microphone sessions."""
    return getattr(request, "session_hash", None) or "anonymous"


def _get_spotter(request: gr.Request) -> Optional[KeywordSpotter]:
    """Return the spotter registered to this browser session, if any."""
    with _SESSIONS_LOCK:
        return _SESSIONS.get(_session_key(request))


def _register_spotter(request: gr.Request, spotter: KeywordSpotter) -> None:
    """Store ``spotter`` for this session, stopping a previous one first."""
    key = _session_key(request)
    with _SESSIONS_LOCK:
        previous = _SESSIONS.get(key)
        _SESSIONS[key] = spotter
    if previous is not None and previous is not spotter:
        previous.stop()


def _unregister_spotter(request: gr.Request) -> Optional[KeywordSpotter]:
    """Remove and stop the session's spotter; safe when none is registered."""
    key = _session_key(request)
    with _SESSIONS_LOCK:
        spotter = _SESSIONS.pop(key, None)
    if spotter is not None:
        spotter.stop()
    return spotter


def _stop_all() -> None:
    """Stop every registered spotter; used at process exit."""
    with _SESSIONS_LOCK:
        spotters = list(_SESSIONS.values())
        _SESSIONS.clear()
    for spotter in spotters:
        spotter.stop()


atexit.register(_stop_all)

_STATUS_STYLES = {
    "idle": ("Idle", "#9ca3af"),
    "listening": ("Listening", "#16a34a"),
    "error": ("Error", "#dc2626"),
}

_REASON_TEXT = {
    "accepted": "reported",
    "below_threshold": "below tau",
    "reject_class": "rejection class",
    "cooldown": "cooldown",
}


def _status_html(state: str, note: str = "") -> str:
    """Render the coloured status indicator shown above the controls."""
    label, color = _STATUS_STYLES.get(state, _STATUS_STYLES["idle"])
    note_html = (
        f'<span style="color:#4b5563; margin-left:10px;">{html.escape(note)}</span>'
        if note
        else ""
    )
    return (
        '<div style="display:flex; align-items:center; gap:10px; font-size:1.05rem;">'
        f'<span style="color:{color}; font-size:1.5rem; line-height:1;">&#9679;</span>'
        f"<strong>{label}</strong>{note_html}"
        "</div>"
    )


def _detail_markdown(spotter: Optional[KeywordSpotter]) -> str:
    """Explain the current window: label, calibrated confidence and calibration."""
    if spotter is None:
        return (
            "**Idle.** The loop classifies a "
            f"{WINDOW_SECONDS:.1f} s window every {HOP_SECONDS:.1f} s. A command is "
            "reported when its calibrated confidence reaches tau, its label is not a "
            f"rejection class, and the {COOLDOWN_SECONDS:.1f} s cooldown has elapsed."
        )

    snapshot = spotter.status()
    lines: List[str] = []
    if snapshot["error"]:
        lines.append(f"**Runtime error:** `{snapshot['error']}`")

    last = snapshot["last_window"]
    if last is None:
        lines.append("Waiting for the first audio window...")
    else:
        reason = _REASON_TEXT.get(last["reason"], last["reason"])
        lines.append(
            f"**Latest window:** `{last['label']}` - calibrated confidence "
            f"**{last['confidence'] * 100:.1f} %** ({reason})"
        )

    rejected = snapshot["last_rejected"]
    if rejected is not None and (last is None or rejected["timestamp"] != last["timestamp"]):
        reason = _REASON_TEXT.get(rejected["reason"], rejected["reason"])
        lines.append(
            f"**Last rejected window:** `{rejected['label']}` at "
            f"{rejected['confidence'] * 100:.1f} % ({reason})"
        )

    calibration = (
        f"`T = {snapshot['temperature']:.4f}` | `tau = {snapshot['threshold']:.2f}`"
    )
    if snapshot["cooldown_remaining"] > 0.05:
        calibration += f" | cooldown `{snapshot['cooldown_remaining']:.1f} s`"
    lines.append(calibration)

    if snapshot["stream_warning"]:
        lines.append(f"**Audio stream warning:** `{snapshot['stream_warning']}`")
    return "  \n".join(lines)


def _idle_button() -> gr.Button:
    """Return the button component in its idle configuration."""
    return gr.Button(
        value="Start Listening",
        variant="primary",
        size="lg",
        interactive=RUNTIME is not None,
    )


def _error_outputs(history: List[dict], message: str) -> tuple:
    """Build the component updates shown when listening cannot start."""
    return (
        _status_html("error", "Listening did not start."),
        message,
        history,
        _idle_button(),
        gr.Timer(value=TIMER_INTERVAL, active=False),
    )


def toggle_listening(history, device_label, request: gr.Request):
    """Start the microphone when idle, stop it when listening."""
    history = list(history or [])
    current = _get_spotter(request)
    if current is not None and current.is_running:
        _unregister_spotter(request)
        return (
            _status_html("idle", "Microphone released."),
            _detail_markdown(None),
            history,
            _idle_button(),
            gr.Timer(value=TIMER_INTERVAL, active=False),
        )

    if RUNTIME is None:
        _unregister_spotter(request)
        return _error_outputs(
            history,
            "**The inference engine is unavailable.**\n\n"
            f"```\n{INIT_ERROR}\n```\n"
            f"See the README for the {MODEL_NAME} {MODEL_VERSION} release layout.",
        )

    device = DEVICE_INDEX.get(device_label)
    spotter = None
    try:
        spotter = KeywordSpotter(MODEL_VERSION, device=device)
        spotter.start()
    except SetupError as exc:
        if spotter is not None:
            spotter.stop()
        _unregister_spotter(request)
        return _error_outputs(history, f"**Could not start listening.**\n\n```\n{exc}\n```")
    except Exception as exc:
        if spotter is not None:
            spotter.stop()
        _unregister_spotter(request)
        return _error_outputs(
            history,
            f"**Could not start listening.**\n\n```\n{type(exc).__name__}: {exc}\n```",
        )

    _register_spotter(request, spotter)
    snapshot = spotter.status()
    return (
        _status_html(
            "listening",
            f"tau = {snapshot['threshold']:.2f} | "
            f"{snapshot['model_name']} {snapshot['model_version']}",
        ),
        _detail_markdown(spotter),
        history,
        gr.Button(value="Stop Listening", variant="stop", size="lg"),
        gr.Timer(value=TIMER_INTERVAL, active=True),
    )


def poll_realtime(history, request: gr.Request):
    """Drain new detections, refresh the status line and stop dead loops."""
    previous = list(history or [])
    history = list(previous)
    spotter = _get_spotter(request)
    if spotter is None:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.Timer(value=TIMER_INTERVAL, active=False),
        )

    for detection in spotter.poll():
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"**{detection.label}** - "
                    f"{detection.confidence * 100:.1f} % at {detection.time_text}"
                ),
            }
        )
    history_changed = len(history) > len(previous)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        history_changed = True

    snapshot = spotter.status()
    if snapshot["error"]:
        detail = _detail_markdown(spotter)
        _unregister_spotter(request)
        return (
            _status_html("error", "The listening loop stopped."),
            detail,
            history,
            _idle_button(),
            gr.Timer(value=TIMER_INTERVAL, active=False),
        )

    if not snapshot["running"]:
        _unregister_spotter(request)
        return (
            _status_html("idle", "Listening stopped."),
            _detail_markdown(None) + "  \nThe loop is no longer running.",
            history,
            _idle_button(),
            gr.Timer(value=TIMER_INTERVAL, active=False),
        )

    return (
        _status_html(
            "listening",
            f"tau = {snapshot['threshold']:.2f} | "
            f"{snapshot['model_name']} {snapshot['model_version']}",
        ),
        _detail_markdown(spotter),
        history if history_changed else gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def clear_history() -> List[dict]:
    """Return an empty detection history for the Clear History button."""
    return []


def close_session(request: gr.Request) -> None:
    """Release the microphone when the tab closes or reloads."""
    _unregister_spotter(request)


def predict_command(audio, backend, threshold):
    """Classify one uploaded/recorded clip with the shared engine.

    ``tau`` is applied to the returned probabilities here instead of being
    written back into the shared engine, so sweeping the slider cannot change
    the calibration values the real-time tab (or another session) is using.
    """
    if RUNTIME is None:
        return (
            {},
            "",
            "**The inference engine is unavailable.**",
            f"```\n{INIT_ERROR}\n```",
        )
    if not audio:
        return (
            {},
            "**No audio yet.**\n\nRecord or upload a clip, then press **Classify Command**.",
            "",
            "",
        )

    backend_key = BACKEND_KEYS.get(backend, "auto")
    try:
        with RUNTIME.inference_lock:
            result = RUNTIME.engine.predict(audio, backend=backend_key)
    except Exception as exc:
        return (
            {},
            "",
            "**Prediction failed.**",
            f"```\n{type(exc).__name__}: {exc}\n```",
        )

    probabilities = result["probabilities"]
    top_label = result["raw_label"]
    confidence = float(probabilities[top_label])
    threshold = float(threshold)
    passes = confidence >= threshold
    is_rejection_class = top_label in REJECTION_CLASSES
    served_label = top_label if passes else REJECT_LABEL

    window_count = int(result.get("window_count", 1))
    window_start = float(result.get("window_start_seconds", 0.0))
    window_note = (
        ""
        if window_count <= 1
        else (
            " The command was found in the 1.0 s window starting at "
            f"**{window_start:.2f} s** ({window_count} windows scanned with a "
            "0.5 s hop)."
        )
    )

    if passes and not is_rejection_class:
        summary = (
            f"### {top_label}\n\n"
            f"Calibrated confidence **{confidence * 100:.1f} %** reaches "
            f"tau **{threshold:.2f}**.{window_note}"
        )
        status = f"**Accepted as a command:** `{top_label}`."
    elif passes and is_rejection_class:
        summary = (
            f"### Non-command class\n\n"
            f"Top-1 is `{top_label}` at **{confidence * 100:.1f} %**. The probability "
            "distribution is dominated by a rejection class, so the real-time spotter "
            "would not report a command."
        )
        status = f"**Top-1 is a rejection class:** `{top_label}`."
    else:
        summary = (
            f"### Rejected\n\n"
            f"Top-1 was `{top_label}` at **{confidence * 100:.1f} %**, below "
            f"tau **{threshold:.2f}**; the served label is `{REJECT_LABEL}`."
        )
        status = f"**Below tau:** top-1 `{top_label}` did not reach {threshold:.2f}."

    details = "\n".join(
        [
            "| Field | Value |",
            "| :-- | :-- |",
            f"| Served label | `{served_label}` |",
            f"| Top-1 label | `{top_label}` |",
            f"| Calibrated confidence | {confidence * 100:.2f} % |",
            f"| tau | {threshold:.2f} |",
            f"| Passes tau | {'yes' if passes else 'no'} |",
            f"| Rejection class | {'yes' if is_rejection_class else 'no'} |",
            f"| Temperature T | {result['temperature']:.4f} |",
            f"| Backend | `{result['backend']}` |",
            f"| Windows scanned | {window_count} |",
            f"| Winning window start | {window_start:.2f} s |",
        ]
    )
    return probabilities, summary, status, details


def _header_markdown() -> str:
    """Describe the shared release and flag anything the tabs cannot do."""
    if RUNTIME is None:
        return (
            f"## {MODEL_NAME} {MODEL_VERSION}\n\n"
            "**The inference engine could not be loaded.** Both tabs are disabled.\n\n"
            f"```\n{INIT_ERROR}\n```"
        )

    source = (
        str(RUNTIME.calibration_source).replace("\\", "/").rsplit("/", 1)[-1]
        if RUNTIME.calibration_source
        else "none (T = 1.0, tau = 0.0)"
    )
    backend = "ONNX Runtime" if "onnx" in RUNTIME.engine.available_backends else "PyTorch"
    lines = [
        f"## {MODEL_NAME} {MODEL_VERSION}",
        (
            "One runtime, two views: stream the microphone or analyse a single clip. "
            "The real-time tab uses the fitted operating point; the clip tab lets you "
            "sweep `tau` without changing the shared calibration."
        ),
        (
            f"**{len(RUNTIME.labels)} classes** | warm-up `{RUNTIME.warmup_ms:.0f} ms` | "
            f"real-time backend `{backend}` | calibration `{source}`"
        ),
    ]
    if not RUNTIME.realtime_available:
        lines.append(
            "> **The real-time tab is unavailable for this release.** It needs the "
            "ONNX graph and a fitted calibration payload; the single-recording tab "
            "still works with the backends listed below."
        )
    return "\n\n".join(lines)


def build_interface() -> gr.Blocks:
    """Build the two-tab Blocks application around the shared runtime."""
    with gr.Blocks(title=f"{MODEL_NAME} - Spoken Command Recognition") as demo:
        gr.Markdown(_header_markdown())

        with gr.Tabs():
            with gr.Tab("Real-time Keyword Spotting"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=320):
                        status_output = gr.HTML(_status_html("idle"))
                        with gr.Row():
                            toggle_button = gr.Button(
                                value="Start Listening",
                                variant="primary",
                                size="lg",
                                interactive=RUNTIME is not None,
                            )
                            clear_button = gr.Button(
                                "Clear History", variant="secondary", size="lg"
                            )
                        device_dropdown = gr.Dropdown(
                            choices=DEVICE_CHOICES,
                            value=DEFAULT_DEVICE_LABEL,
                            label="Input device",
                            info="Pick a microphone, or keep the system default.",
                            interactive=RUNTIME is not None and len(DEVICE_CHOICES) > 1,
                        )
                        timer = gr.Timer(value=TIMER_INTERVAL, active=False)
                        detail_output = gr.Markdown(_detail_markdown(None))
                    with gr.Column(scale=2, min_width=320):
                        history_output = gr.Chatbot(
                            value=[],
                            label="Detection history",
                            height=430,
                            layout="panel",
                            group_consecutive_messages=False,
                            feedback_options=None,
                            placeholder="Accepted commands appear here with their confidence.",
                        )

                toggle_button.click(
                    toggle_listening,
                    inputs=[history_output, device_dropdown],
                    outputs=[
                        status_output,
                        detail_output,
                        history_output,
                        toggle_button,
                        timer,
                    ],
                )
                timer.tick(
                    poll_realtime,
                    inputs=[history_output],
                    outputs=[
                        status_output,
                        detail_output,
                        history_output,
                        toggle_button,
                        timer,
                    ],
                )
                clear_button.click(clear_history, outputs=[history_output])

            with gr.Tab("Single Recording Analysis"):
                gr.Markdown(
                    "Record with the microphone or upload a clip, then classify it with "
                    "the same release. Clips are force-fitted to "
                    f"**{WINDOW_SECONDS:.1f} s** at **{SAMPLE_RATE} Hz** by the shared "
                    "front-end."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Command clip",
                        )
                        backend_radio = gr.Radio(
                            choices=BACKENDS or ["ONNX"],
                            value=INITIAL_BACKEND,
                            label="Inference backend",
                            interactive=len(BACKENDS) > 1,
                        )
                        threshold_slider = gr.Slider(
                            minimum=CONFIDENCE_THRESHOLD_RANGE[0],
                            maximum=CONFIDENCE_THRESHOLD_RANGE[1],
                            value=INITIAL_THRESHOLD,
                            step=CONFIDENCE_THRESHOLD_STEP,
                            label="Confidence threshold (tau)",
                            info=(
                                "Applied to this clip only; the real-time tab keeps the "
                                "fitted tau."
                            ),
                        )
                        classify_button = gr.Button(
                            "Classify Command",
                            variant="primary",
                            size="lg",
                            interactive=RUNTIME is not None,
                        )
                    with gr.Column(scale=1, min_width=320):
                        verdict_output = gr.Markdown(
                            "### Waiting for a clip\n\n"
                            "Record or upload a clip, then press **Classify Command**."
                        )
                        status_output_single = gr.Markdown()
                        label_output = gr.Label(
                            num_top_classes=6,
                            label="Calibrated class probabilities",
                        )
                        details_output = gr.Markdown()

                classify_button.click(
                    predict_command,
                    inputs=[audio_input, backend_radio, threshold_slider],
                    outputs=[
                        label_output,
                        verdict_output,
                        status_output_single,
                        details_output,
                    ],
                )

    demo.unload(close_session)
    return demo


demo = build_interface()


if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Soft())

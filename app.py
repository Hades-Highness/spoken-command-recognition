"""Gradio demo for CommandSense.

Run with:  python app.py

v2.2 adds a live view of the confidence calibration: the fitted temperature ``T``
is applied to the raw logits before the softmax, and a ``tau`` slider re-applies
the confidence threshold without reloading the model. Clips whose calibrated
top-1 probability stays below ``tau`` are rejected with the project's
``_unknown_`` label and flagged in the UI.
"""

import gradio as gr

from configs.config import (
    CONFIDENCE_THRESHOLD_STEP,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    REJECT_LABEL,
)
from src.inference import CommandInferencer

inferencer = None
init_error = None

try:
    inferencer = CommandInferencer()
except Exception as exc:  # missing checkpoint, missing labels, ...
    init_error = f"{type(exc).__name__}: {exc}"

# The inference engine is the single source of truth for which backends exist,
# so the UI can never offer a backend the code cannot serve.
BACKENDS = ["ONNX", "PyTorch (.pth)"] if (inferencer and "onnx" in inferencer.available_backends) \
    else ["PyTorch (.pth)"]

# The fitted values come from checkpoints/model_v2.2/*_calibration_v2.2.json; a
# missing or partial file leaves the configured defaults in place.
INITIAL_TAU = (
    float(inferencer.confidence_threshold) if inferencer else float(DEFAULT_CONFIDENCE_THRESHOLD)
)
CALIBRATION_INFO = (
    f"**T** = `{inferencer.temperature:.4f}` &nbsp;|&nbsp; **tau** = "
    f"`{inferencer.confidence_threshold:.2f}` &nbsp;|&nbsp; reject label "
    f"`{REJECT_LABEL}`"
    if inferencer
    else "Calibration values unavailable."
)


def predict_command(audio, backend, threshold):
    if inferencer is None:
        return {}, f"⚠️ Inference engine unavailable.\n\n`{init_error}`", ""
    if not audio:
        return {}, "Upload a clip or record one, then press **Classify Command**.", ""

    try:
        result = inferencer.predict(
            audio,
            backend="onnx" if backend == "ONNX" else "pytorch",
            confidence_threshold=threshold,
        )
    except Exception as exc:
        return {}, f"⚠️ Prediction failed.\n\n`{type(exc).__name__}: {exc}`", ""

    if result["is_low_confidence"]:
        summary = (
            f"### 🚫 {result['label']} — low confidence\n"
            f"Best guess **{result['raw_label']}** scored only "
            f"{result['confidence'] * 100:.1f} %, below tau = "
            f"{result['threshold']:.2f}, so the clip was rejected as "
            f"`{REJECT_LABEL}` instead of being answered."
        )
        badge = (
            f"⚠️ **Rejected** — the calibrated confidence did not reach the "
            f"threshold. Lower **tau** to answer more clips, raise it to answer "
            f"fewer mistakes."
        )
        gr.Warning(
            f"Low confidence: {result['raw_label']} {result['confidence'] * 100:.1f} % "
            f"< tau {result['threshold']:.2f} -> {REJECT_LABEL}"
        )
    else:
        summary = (
            f"### {result['label']} — {result['confidence'] * 100:.1f} %\n"
            f"Calibrated confidence reached tau = {result['threshold']:.2f}, so the "
            f"command was accepted."
        )
        badge = f"✅ **Accepted** — confidence ≥ tau = {result['threshold']:.2f}."

    summary += (
        f"\n\nBackend: `{result['backend']}` &nbsp;|&nbsp; T = "
        f"`{result['temperature']:.4f}` &nbsp;|&nbsp; raw argmax: "
        f"`{result['raw_label']}`"
    )
    return result["probabilities"], summary, badge


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🎙️ {MODEL_NAME} {MODEL_VERSION} — Spoken Command Classifier

        Record a command or upload a 1-second audio file, then classify it over
        **35 command words**. Confidence is temperature-calibrated and clips below
        the threshold are rejected with `{REJECT_LABEL}`.
        """
    )

    if inferencer is None:
        gr.Markdown(f"> ⚠️ **Inference engine not initialised.**\n>\n> `{init_error}`")
    else:
        gr.Markdown(
            f"> **Calibration** — {CALIBRATION_INFO}\n>\n> Source: "
            f"`{inferencer.calibration_source or 'configs/config.py defaults'}`"
        )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio input",
            )
            backend_radio = gr.Radio(
                choices=BACKENDS,
                value=BACKENDS[0],
                label="Inference backend",
            )
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=INITIAL_TAU,
                step=CONFIDENCE_THRESHOLD_STEP,
                label="Confidence threshold (tau) — drag to re-decide without reloading",
            )
            submit_btn = gr.Button("Classify Command", variant="primary")
        with gr.Column():
            summary_output = gr.Markdown()
            status_output = gr.Markdown()
            label_output = gr.Label(num_top_classes=5, label="Top 5 predictions")

    submit_btn.click(
        fn=predict_command,
        inputs=[audio_input, backend_radio, threshold_slider],
        outputs=[label_output, summary_output, status_output],
    )
    audio_input.change(
        fn=predict_command,
        inputs=[audio_input, backend_radio, threshold_slider],
        outputs=[label_output, summary_output, status_output],
    )
    # Re-decide on the same clip when the slider is released: T is already
    # fitted, so only the threshold comparison changes.
    threshold_slider.release(
        fn=predict_command,
        inputs=[audio_input, backend_radio, threshold_slider],
        outputs=[label_output, summary_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()

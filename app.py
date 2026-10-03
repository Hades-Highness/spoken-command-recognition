"""Gradio demo for CommandSense.

Run with:  python app.py
"""

import gradio as gr

from configs.config import MODEL_NAME, MODEL_VERSION
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


def predict_command(audio, backend):
    if inferencer is None:
        return {}, f"⚠️ Inference engine unavailable.\n\n`{init_error}`"
    if not audio:
        return {}, "Upload a clip or record one, then press **Classify Command**."

    try:
        result = inferencer.predict(audio, backend="onnx" if backend == "ONNX" else "pytorch")
    except Exception as exc:
        return {}, f"⚠️ Prediction failed.\n\n`{type(exc).__name__}: {exc}`"

    summary = (
        f"### {result['label']} — {result['confidence'] * 100:.1f} %\n"
        f"Backend: `{result['backend']}`"
    )
    return result["probabilities"], summary


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🎙️ {MODEL_NAME} {MODEL_VERSION} — Spoken Command Classifier

        Record a command or upload a 1-second audio file, then classify it over
        **35 command words**.
        """
    )

    if inferencer is None:
        gr.Markdown(f"> ⚠️ **Inference engine not initialised.**\n>\n> `{init_error}`")

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
            submit_btn = gr.Button("Classify Command", variant="primary")
        with gr.Column():
            summary_output = gr.Markdown()
            label_output = gr.Label(num_top_classes=5, label="Top 5 predictions")

    submit_btn.click(
        fn=predict_command,
        inputs=[audio_input, backend_radio],
        outputs=[label_output, summary_output],
    )
    audio_input.change(
        fn=predict_command,
        inputs=[audio_input, backend_radio],
        outputs=[label_output, summary_output],
    )

if __name__ == "__main__":
    demo.launch()

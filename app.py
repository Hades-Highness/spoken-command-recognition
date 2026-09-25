import os
import gradio as gr
from configs.config import MODEL_NAME, MODEL_VERSION
from src.inference import CommandSenseInferencer

inferencer = None

try:
    inferencer = CommandSenseInferencer()
    print(f"[+] Inferencer initialized successfully for {MODEL_NAME} {MODEL_VERSION}.")
except Exception as e:
    print(f"[!] Warning: Could not initialize inferencer: {e}")

def predict_command(audio, backend):
    if inferencer is None or audio is None:
        return {}
    return inferencer.predict(audio, backend=backend)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🎙️ {MODEL_NAME} {MODEL_VERSION} - Real-Time Audio Command Classifier
        Say a keyword or upload an audio file (1 second) to run real-time inference.
        """
    )
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
            backend_radio = gr.Radio(
                choices=["ONNX", "PyTorch (.pth)"],
                value="ONNX",
                label="Model Format Backend"
            )
            submit_btn = gr.Button("Classify Command", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")

    submit_btn.click(fn=predict_command, inputs=[audio_input, backend_radio], outputs=label_output)
    audio_input.change(fn=predict_command, inputs=[audio_input, backend_radio], outputs=label_output)

if __name__ == "__main__":
    demo.launch()
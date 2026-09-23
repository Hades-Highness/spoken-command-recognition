import os
import gradio as gr
from src.inference import CommandSenseInferencer

CHECKPOINT_PATH = "checkpoints/CommandSense_v1.pth"
inferencer = None

if os.path.exists(CHECKPOINT_PATH):
    inferencer = CommandSenseInferencer(CHECKPOINT_PATH)
    print(f"[+] Loaded inferencer with model: {CHECKPOINT_PATH}")
else:
    print(f"[!] Warning: Checkpoint {CHECKPOINT_PATH} not found.")

def predict_command(audio):
    if inferencer is None or audio is None:
        return {}
    return inferencer.predict(audio)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ CommandSense v1.0 - Real-Time Audio Command Classifier
        Say a keyword or upload an audio file (1 second) to run inference.
        """
    )
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
            submit_btn = gr.Button("Classify Command", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")

    submit_btn.click(fn=predict_command, inputs=audio_input, outputs=label_output)
    audio_input.change(fn=predict_command, inputs=audio_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
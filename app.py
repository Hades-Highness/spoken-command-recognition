import os
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import gradio as gr

from configs.config import SAMPLE_RATE, TARGET_SAMPLES, N_MELS, N_FFT, HOP_LENGTH, NUM_CLASSES
from src.models import CommandSense

# List of 35 Google Speech Commands classes
CLASS_NAMES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/CommandSense_v1.pth"

# Load Model
model = CommandSense(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"[+] Loaded weights from {CHECKPOINT_PATH}")
else:
    print(f"[!] Warning: {CHECKPOINT_PATH} not found. Please train the model first.")

model.eval()

# Audio Preprocessing Transform
mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

def preprocess_audio(audio_path):
    """Loads audio, enforces 16kHz mono, pads/crops to 1s, and converts to Log-Mel."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or crop to 16,000 samples (1 second)
    num_samples = waveform.shape[1]
    if num_samples < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - num_samples
        waveform = F.pad(waveform, (0, padding))
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]

    # Compute Log-Mel Spectrogram
    mel_spec = mel_spectrogram(waveform)
    log_mel_spec = torch.log(mel_spec + 1e-6)
    
    # Add batch dimension -> [1, 1, 64, 63]
    return log_mel_spec.unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(audio):
    if audio is None:
        return {}

    tensor = preprocess_audio(audio)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)

    # Format top predictions as dictionary for Gradio Label
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return confidences

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ CommandSense v1.0 - Real-Time Audio Command Classifier
        Say a keyword (e.g., *'stop'*, *'go'*, *'left'*, *'right'*, *'three'*) or upload an audio file to classify it.
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Record or Upload Audio (1 second)"
            )
            submit_btn = gr.Button("Classify Command", variant="primary")
            
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Top 5 Predictions")

    submit_btn.click(fn=predict, inputs=audio_input, outputs=label_output)
    audio_input.change(fn=predict, inputs=audio_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
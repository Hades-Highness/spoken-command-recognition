import os
import sys
import torch
import soundfile as sf
import torchaudio

def _soundfile_load(filepath, **kwargs):
    data, sample_rate = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.T
    return tensor, sample_rate

torchaudio.load = _soundfile_load

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import MODEL_NAME, MODEL_VERSION, BATCH_SIZE, CHECKPOINT_DIR, NUM_CLASSES
from src.dataset import SpeechCommandsDataset
from src.models import CommandSense, AudioFeatureExtractor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware device: {device}")

    # Build checkpoint path
    checkpoint_folder = os.path.join(CHECKPOINT_DIR, f"model_{MODEL_VERSION.lower()}")
    checkpoint_path = os.path.join(checkpoint_folder, f"{MODEL_NAME}_{MODEL_VERSION}.pth")
    
    if not os.path.exists(checkpoint_path):
        # Fallback check directly in CHECKPOINT_DIR
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"[!] Error: Model checkpoint file not found at {checkpoint_path}")
            return

    print(f"[*] Loading model checkpoint: {checkpoint_path}")

    # Load test dataset
    test_dataset = SpeechCommandsDataset(subset="testing", cache_in_ram=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize feature extractor and 3-channel ResNet model for v2.0.0
    feature_extractor = AudioFeatureExtractor().to(device)
    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    print("[*] Running evaluation on testing set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            # Extract 3-channel Log-Mel + Delta + Delta-Delta features if input is audio waveform
            if inputs.ndim == 2 or (inputs.ndim == 3 and inputs.shape[1] == 1):
                features = feature_extractor(inputs)
            else:
                features = inputs
                
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    class_names = test_dataset.labels

    # Print Classification Report
    print(f"\n--- CLASSIFICATION REPORT ({MODEL_NAME} {MODEL_VERSION}) ---")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # Save Classification Report to text file
    report_dir = os.path.join("reports", f"model_{MODEL_VERSION.lower()}")
    os.makedirs(report_dir, exist_ok=True)
    
    report_txt_path = os.path.join(report_dir, f"report_{MODEL_VERSION}.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"--- CLASSIFICATION REPORT ({MODEL_NAME} {MODEL_VERSION}) ---\n")
        f.write(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    print(f"[+] Text report saved -> {report_txt_path}")

    # Dark Mode Confusion Matrix Plot
    plt.style.use('dark_background')
    cm = confusion_matrix(all_targets, all_preds)
    
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    sns.heatmap(
        cm, 
        annot=False, 
        fmt='d', 
        cmap='mako', 
        xticklabels=class_names, 
        yticklabels=class_names, 
        ax=ax
    )
    
    plt.title(f'Confusion Matrix - {MODEL_NAME} {MODEL_VERSION}', color='#c9d1d9', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', color='#8b949e', fontsize=12)
    plt.ylabel('True Label', color='#8b949e', fontsize=12)
    
    report_filename = os.path.join(report_dir, f"confusion_matrix_{MODEL_VERSION}.png")
    plt.savefig(report_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    print(f"[+] Confusion matrix saved -> {report_filename} ({MODEL_NAME} {MODEL_VERSION})")

if __name__ == "__main__":
    main()
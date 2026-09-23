import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from configs.config import BATCH_SIZE
from src.dataset import SpeechCommandsDataset
from src.models import CommandSense

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/CommandSense_v1.pth"

    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: File {checkpoint_path} not found.")
        return

    test_dataset = SpeechCommandsDataset(subset="testing", cache_in_ram=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CommandSense(num_classes=35, in_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    class_names = test_dataset.labels

    # Print Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # Dark Mode Confusion Matrix Plot
    plt.style.use('dark_background')
    cm = confusion_matrix(all_targets, all_preds)
    
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    sns.heatmap(cm, annot=False, fmt='d', cmap='mako', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    plt.title('Confusion Matrix - CommandSense v1.0', color='#c9d1d9', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', color='#8b949e', fontsize=12)
    plt.ylabel('True Label', color='#8b949e', fontsize=12)
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("[+] Dark mode confusion matrix saved -> reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
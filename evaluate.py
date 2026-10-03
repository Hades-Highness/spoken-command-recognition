"""Evaluate a checkpoint on the official Speech Commands test split.

Writes into the versioned reports folder:

    reports/model_v2.0/report_v2.0.txt
    reports/model_v2.0/confusion_matrix_v2.0.png

Run with:  python evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    NUM_WORKERS,
    checkpoint_dir,
    reports_dir,
)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.models import CommandSense  # noqa: E402
import src.utils  # noqa: E402,F401  (installs the torchaudio.load fallback early)


def resolve_checkpoint(version=MODEL_VERSION):
    candidates = [
        checkpoint_dir(version) / f"{MODEL_NAME}_{version}.pth",
        CHECKPOINT_DIR / f"{MODEL_NAME}_{version}.pth",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(candidate)
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware device: {device}")

    checkpoint_path = resolve_checkpoint()
    if checkpoint_path is None:
        print(
            f"[!] No checkpoint found at '{checkpoint_dir(MODEL_VERSION)}/"
            f"{MODEL_NAME}_{MODEL_VERSION}.pth'. Download the release asset or run train.py."
        )
        return
    print(f"[*] Loading model checkpoint: {checkpoint_path}")

    test_dataset = SpeechCommandsDataset(subset="testing", cache_in_ram=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    class_names = test_dataset.labels

    # Guard-rail: the model was trained with the index order written by train.py.
    # If the mapping has drifted, every reported number would be meaningless.
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as handle:
            declared_labels = json.load(handle)
        if declared_labels != class_names:
            raise RuntimeError(
                f"Class mapping mismatch: '{LABELS_PATH}' differs from the test split's "
                "label order. Delete configs/labels.json and re-run train.py."
            )
        print(f"[+] Class mapping verified against {LABELS_PATH} ({len(class_names)} classes)")

    model = CommandSense(num_classes=NUM_CLASSES, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    print("[*] Running evaluation on the testing split...")
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            preds = model(features).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    accuracy = (sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)) * 100
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    print(f"\n--- CLASSIFICATION REPORT ({MODEL_NAME} {MODEL_VERSION}) ---")
    print(report)

    out_dir = reports_dir(MODEL_VERSION)
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, f"report_{MODEL_VERSION}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"--- CLASSIFICATION REPORT ({MODEL_NAME} {MODEL_VERSION}) ---\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Samples: {len(all_targets)}  Accuracy: {accuracy:.4f}%\n\n")
        handle.write(report)
    print(f"[+] Saved {report_path}")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        confusion_matrix(all_targets, all_preds),
        annot=False, fmt="d", cmap="mako",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    plt.title(
        f"Confusion Matrix - {MODEL_NAME} {MODEL_VERSION} (test split)",
        color="#c9d1d9", fontsize=16, pad=20,
    )
    plt.xlabel("Predicted Label", color="#8b949e", fontsize=12)
    plt.ylabel("True Label", color="#8b949e", fontsize=12)

    matrix_path = os.path.join(out_dir, f"confusion_matrix_{MODEL_VERSION}.png")
    plt.savefig(matrix_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {matrix_path}")
    print(f"[+] Test accuracy: {accuracy:.2f}% on {len(all_targets)} clips")


if __name__ == "__main__":
    main()

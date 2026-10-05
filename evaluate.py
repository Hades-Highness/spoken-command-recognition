import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from train import set_seed

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LABELS_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
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
    set_seed(SEED)
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

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # --- Metrics Computation ---
    overall_accuracy = (np.sum(all_preds == all_targets) / len(all_targets)) * 100

    # Separate core 35 commands and rejection classes
    rejection_classes = {"_silence_", "_unknown_"}
    cmd_indices = [
        idx for idx, label in enumerate(class_names) if label not in rejection_classes
    ]

    cmd_mask = np.isin(all_targets, cmd_indices)
    cmd_targets = all_targets[cmd_mask]
    cmd_preds = all_preds[cmd_mask]
    cmd_accuracy = (np.sum(cmd_preds == cmd_targets) / len(cmd_targets)) * 100 if len(cmd_targets) > 0 else 0.0

    # Recall for rejection classes
    rejection_recalls = {}
    for rej_label in rejection_classes:
        if rej_label in class_names:
            rej_idx = class_names.index(rej_label)
            rej_mask = (all_targets == rej_idx)
            if np.sum(rej_mask) > 0:
                rec = (np.sum(all_preds[rej_mask] == rej_idx) / np.sum(rej_mask)) * 100
                rejection_recalls[rej_label] = (rec, int(np.sum(all_preds[rej_mask] == rej_idx)), int(np.sum(rej_mask)))
            else:
                rejection_recalls[rej_label] = (0.0, 0, 0)

    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    # Print summary block to console
    print("\n" + "=" * 60)
    print(f"       EVALUATION REPORT ({MODEL_NAME} {MODEL_VERSION})       ")
    print("=" * 60)
    print(f"Overall Accuracy (37 classes): {overall_accuracy:.2f}% ({np.sum(all_preds == all_targets)}/{len(all_targets)})")
    print("-" * 60)
    print(f"35 Command Words Top-1 Acc:    {cmd_accuracy:.2f}% ({np.sum(cmd_preds == cmd_targets)}/{len(cmd_targets)})")
    for rej_label, (rec, corr, tot) in rejection_recalls.items():
        print(f"Rejection Recall '{rej_label}':    {rec:.2f}% ({corr}/{tot})")
    print("=" * 60 + "\n")

    print(f"--- DETAILED CLASSIFICATION REPORT ---")
    print(report)

    out_dir = reports_dir(MODEL_VERSION)
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, f"report_{MODEL_VERSION}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("=======================================================\n")
        handle.write(f"       EVALUATION REPORT ({MODEL_NAME} {MODEL_VERSION})       \n")
        handle.write("=======================================================\n\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Total Samples: {len(all_targets)}\n")
        handle.write(f"Overall Accuracy: {overall_accuracy:.4f}%\n\n")
        handle.write("--- SEPARATED METRICS ---\n")
        handle.write(f"Top-1 Accuracy (35 Commands): {cmd_accuracy:.4f}% ({np.sum(cmd_preds == cmd_targets)}/{len(cmd_targets)})\n")
        for rej_label, (rec, corr, tot) in rejection_recalls.items():
            handle.write(f"Recall '{rej_label}':               {rec:.4f}% ({corr}/{tot})\n")
        handle.write("\n--- DETAILED CLASSIFICATION REPORT ---\n")
        handle.write(report)
    print(f"[+] Saved {report_path}")

    # Plot Confusion Matrix
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


if __name__ == "__main__":
    main()
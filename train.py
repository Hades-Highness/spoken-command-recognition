"""Training pipeline for CommandSense.

Writes everything into the versioned folder for the current MODEL_VERSION:

    checkpoints/model_v2.0/CommandSense_v2.0.pth
    reports/model_v2.0/history_v2.0.json
    reports/model_v2.0/training_curves_v2.0.png
    configs/labels.json                        (index -> class name mapping)
"""

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.config import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LABELS_PATH,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_WORKERS,
    SEED,
    WEIGHT_DECAY,
    checkpoint_dir,
    reports_dir,
)
from src.dataset import SpeechCommandsDataset  # noqa: E402
from src.models import CommandSense  # noqa: E402


def set_seed(seed: int = SEED) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_labels(labels, path=LABELS_PATH) -> None:
    """Persist the index -> class name mapping next to the source code.

    The file is committed, so it is only rewritten when the mapping actually
    changes; this keeps training runs from dirtying the working tree.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            if json.load(handle) == labels:
                return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=4)
        handle.write("\n")
    print(f"[+] Updated class mapping -> {path}")


def save_reports(history, model_name, model_version, suffix=""):
    """Write the metric history and the training curves into the version folder."""
    out_dir = reports_dir(model_version)
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")
    fig.suptitle(
        f"{model_name} {model_version} - Training Metrics",
        color="#c9d1d9", fontsize=16, fontweight="bold",
    )

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color="#58a6ff", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], color="#f85149", label="Val Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Loss", color="#8b949e")
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], color="#58a6ff", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], color="#f85149", label="Val Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", color="#c9d1d9", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("Accuracy (%)", color="#8b949e")
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    curves_path = os.path.join(out_dir, f"training_curves_{model_version}{suffix}.png")
    plt.savefig(curves_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    history_path = os.path.join(out_dir, f"history_{model_version}{suffix}.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"model_name": model_name, "model_version": model_version, "history": history},
            handle,
            indent=4,
        )

    print(f"[+] Saved {curves_path}")
    print(f"[+] Saved {history_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for features, targets in pbar:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(features)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    set_seed(SEED)
    print(f"[*] Global seed set to: {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware device: {device}")

    print("\n[*] Loading Training Dataset into RAM...")
    train_dataset = SpeechCommandsDataset(subset="training", cache_in_ram=True)

    print("\n[*] Loading Validation Dataset into RAM...")
    val_dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=True)

    print("\n[+] Datasets cached! Initializing DataLoaders...")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    save_labels(train_dataset.labels)

    model = CommandSense(num_classes=len(train_dataset.labels), in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(checkpoint_dir(MODEL_VERSION), exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir(MODEL_VERSION), f"{MODEL_NAME}_{MODEL_VERSION}.pth"
    )

    print(f"\n--- Starting {MODEL_NAME} {MODEL_VERSION} Training ---")
    interrupted = False
    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"   [+] New best model saved ({val_acc:.2f}%) -> {checkpoint_path}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[!] Manual interruption detected. Stopping training gracefully...")

    finally:
        completed = len(history["train_loss"])
        if completed == 0:
            print("\n[!] No completed epochs. Nothing written.")
            return

        # An interrupted run must never overwrite an already published history,
        # so it is written under a distinct name.
        save_reports(
            history, MODEL_NAME, MODEL_VERSION,
            suffix="_interrupted" if interrupted else "",
        )
        print(f"\n[*] Best validation accuracy: {best_val_acc:.2f}% over {completed} epochs")
        if interrupted:
            print("[!] Run was interrupted: the *_interrupted files are not release artifacts.")


if __name__ == "__main__":
    main()

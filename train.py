import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from configs.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, CHECKPOINT_DIR
from src.dataset import SpeechCommandsDataset
from src.models import CommandSense

def save_dark_plots(history):
    os.makedirs("reports", exist_ok=True)
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    ax1.set_facecolor('#0d1117')
    ax2.set_facecolor('#0d1117')

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Plot
    ax1.plot(epochs, history["train_loss"], color='#58a6ff', label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], color='#f85149', label="Val Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", color='#c9d1d9', fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", color='#8b949e')
    ax1.set_ylabel("Loss", color='#8b949e')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend()

    # Accuracy Plot
    ax2.plot(epochs, history["train_acc"], color='#3fb950', label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], color='#d29922', label="Val Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", color='#c9d1d9', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", color='#8b949e')
    ax2.set_ylabel("Accuracy (%)", color='#8b949e')
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("reports/training_curves.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    with open("reports/history.json", "w") as f:
        json.dump(history, f, indent=4)
    print("[+] Dark mode training curves saved -> reports/training_curves.png")

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware device: {device}")

    train_dataset = SpeechCommandsDataset(subset="training", cache_in_ram=True)
    val_dataset = SpeechCommandsDataset(subset="validation", cache_in_ram=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CommandSense(num_classes=35, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n--- Starting CommandSense v1.0 Training ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, "CommandSense_v1.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [+] New best model saved ({val_acc:.2f}%) -> {save_path}")

    save_dark_plots(history)

if __name__ == "__main__":
    main()
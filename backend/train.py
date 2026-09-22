import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from backend.dataset import get_dataloaders
from backend.model import AudioSEDNet

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio SED ResNet Model with Local Checkpoints")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing datasets")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8 recommended for 2GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from latest checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start fresh training ignore existing checkpoints")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    torch.save(state, epoch_path)
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, best_path)
        print(f" Saved new best model to {best_path}")
    print(f" Checkpoint saved for Epoch {epoch} at {latest_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 1, float("inf"), []

    print(f" Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])
        print(f" Resumed successfully! Starting at Epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
        return start_epoch, best_val_loss, history
    except Exception as e:
        print(f" Notice: Checkpoint architecture mismatch ({e}). Starting fresh with AudioResNet18 backbone.")
        return 1, float("inf"), []


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, total_epochs=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}] Train", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        current_loss = running_loss / total
        current_acc = (correct / total) * 100
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    num_workers = 2 if device.type == "cuda" else 0
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=num_workers
    )
    num_classes = len(class_names)
    print(f"Dataset split complete: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    model = AudioSEDNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    latest_checkpoint_file = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")

    if args.resume and os.path.exists(latest_checkpoint_file):
        start_epoch, best_val_loss, history = load_checkpoint(
            model, optimizer, scheduler, latest_checkpoint_file
        )

    if start_epoch > args.epochs:
        print(f"Training already completed up to epoch {start_epoch - 1} (target: {args.epochs}). Exiting.")
        return

    patience_counter = 0

    print(f"\n--- Starting Training (Epochs {start_epoch} -> {args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=args.epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
        print(f"Epoch [{epoch}/{args.epochs}] ({elapsed}) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        history.append(history_entry)

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "class_names": class_names,
            "history": history
        }
        save_checkpoint(checkpoint_state, is_best, args.checkpoint_dir, epoch)

        # Save history log to CSV
        df_history = pd.DataFrame(history)
        df_history.to_csv("outputs/training_log.csv", index=False)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no val loss improvement for {args.patience} epochs).")
            break

    print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()

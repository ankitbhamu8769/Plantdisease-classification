"""
train.py
--------
Main training script. Run with:  python train.py

Handles: data loading, model/optimizer/scheduler setup, the train/val loop
(with progressive backbone unfreezing), checkpointing the best model, and
plotting accuracy/loss curves at the end.

Includes Kaggle-specific timeout handling: gracefully stops at 8.5 hours
and resumes from the latest epoch checkpoint if restarted.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import config
from dataset import get_dataloaders, compute_class_weights
from model import build_model
from losses import get_loss_fn

# ---------------------------------------------------------
# Kaggle Timeout Config (Stop safely at 8.5 hours)
# ---------------------------------------------------------
MAX_RUNTIME_SEC = 8.5 * 3600  


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. If `optimizer` is given, trains; otherwise evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, accuracy, precision, recall, f1


def plot_curves(history):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[0].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out_path = f"{config.OUTPUT_DIR}/training_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def build_optimizer(model, lr):
    """Rebuilds the optimizer over currently-trainable params. Called once
    initially, and again right after unfreezing so newly-unfrozen backbone
    params are actually included in the param group."""
    return AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=lr, weight_decay=config.WEIGHT_DECAY)


def main():
    script_start_time = time.time()
    print(f"Using device: {config.DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    model = build_model().to(config.DEVICE)
    print(f"Backbone stage count: {len(list(model.backbone.backbone.blocks.children()))} "
          f"(FREEZE_BACKBONE_STAGES={config.FREEZE_BACKBONE_STAGES})")

    # --- Loss: Focal Loss + Label Smoothing, optionally alpha-weighted ---
    if getattr(config, "USE_CLASS_ALPHA", False):
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(config.DEVICE)
        print(f"Class alpha weights: min={class_weights.min():.3f}, "
              f"max={class_weights.max():.3f}, ratio={class_weights.max()/class_weights.min():.2f}x")
    else:
        class_weights = None

    criterion = get_loss_fn(gamma=config.FOCAL_GAMMA,
                             smoothing=config.LABEL_SMOOTHING,
                             class_weights=class_weights)

    # ---------------------------------------------------------
    # Resume Checkpoint Logic
    # ---------------------------------------------------------
    start_epoch = 1
    best_val_acc = 0.0
    backbone_unfrozen = False
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if getattr(config, "RESUME_TRAINING", False) and os.path.exists(config.LATEST_MODEL_PATH):
        print(f"\n[INFO] Found checkpoint. Resuming from {config.LATEST_MODEL_PATH}...")
        checkpoint = torch.load(config.LATEST_MODEL_PATH, map_location=config.DEVICE)
        
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint.get("history", history)
        backbone_unfrozen = checkpoint.get("backbone_unfrozen", False)
        
        # Must unfreeze backbone BEFORE loading optimizer/model states if it was already unfrozen
        if backbone_unfrozen:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, config.FINE_TUNE_LEARNING_RATE)
        else:
            optimizer = build_optimizer(model, config.LEARNING_RATE)
            
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        
        # Load states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[INFO] Successfully resumed at Epoch {start_epoch}. Best Val Acc: {best_val_acc:.4f}\n")
    else:
        optimizer = build_optimizer(model, config.LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()

        if (not backbone_unfrozen) and epoch >= config.UNFREEZE_EPOCH:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, config.FINE_TUNE_LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
            backbone_unfrozen = True
            print(f"  -> Backbone fully unfrozen at epoch {epoch}, "
                  f"LR set to {config.FINE_TUNE_LEARNING_RATE}")

        train_loss, train_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_prec, val_rec, val_f1 = run_epoch(model, val_loader, criterion)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_prec={val_prec:.4f} val_rec={val_rec:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
            }, config.BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
            
        # Save Latest Epoch State for Resuming
        torch.save({
            "epoch": epoch + 1,  # Points to the next epoch to run
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "backbone_unfrozen": backbone_unfrozen
        }, config.LATEST_MODEL_PATH)

        # ---------------------------------------------------------
        # Time Check: Graceful Exit
        # ---------------------------------------------------------
        total_elapsed = time.time() - script_start_time
        if total_elapsed > MAX_RUNTIME_SEC:
            print(f"\n[WARNING] Approaching Kaggle maximum runtime ({total_elapsed/3600:.2f} hours).")
            print(f"Stopping gracefully to save progress. Re-run notebook to resume from Epoch {epoch + 1}.")
            break

    plot_curves(history)

    # Test Eval (Only run if we have a best model saved)
    if os.path.exists(config.BEST_MODEL_PATH):
        checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, test_prec, test_rec, test_f1 = run_epoch(model, test_loader, criterion)
        print(f"\nFINAL TEST RESULTS: acc={test_acc:.4f} precision={test_prec:.4f} "
              f"recall={test_rec:.4f} f1={test_f1:.4f}")
    else:
        print("\nSkipping test eval: No best model saved yet.")


if __name__ == "__main__":
    main()
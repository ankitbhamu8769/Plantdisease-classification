"""
test_run.py
-----------
A dry-run script verifying data loading, OpenCV cropping,
forward/backward pass, and checkpoint saving.
"""

import os
import torch
import config
from dataset import get_dataloaders, compute_class_weights
from model import build_model
from losses import get_loss_fn


def run_test():
    print("--- STARTING DRY RUN ---")
    print(f"Device: {config.DEVICE}")
    print(f"CV2 Cropping Enabled: {getattr(config, 'USE_CV2_CROP', False)}")
    print(f"SMOTE Balancing Enabled: {getattr(config, 'USE_SMOTE', False)}")

    print("\n1. Testing Dataloaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Classes found: {num_classes}")

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")

    print("\n2. Testing Model Instantiation...")
    model = build_model().to(config.DEVICE)
    print("Model built successfully.")

    if getattr(config, "USE_CLASS_ALPHA", False):
        class_weights = compute_class_weights(train_loader.dataset, num_classes).to(config.DEVICE)
    else:
        class_weights = None

    criterion = get_loss_fn(
        gamma=config.FOCAL_GAMMA,
        smoothing=config.LABEL_SMOOTHING,
        class_weights=class_weights,
    )

    print("\n3. Testing Forward and Backward Pass...")
    logits = model(imgs.to(config.DEVICE))
    loss = criterion(logits, labels.to(config.DEVICE))
    print(f"Loss value: {loss.item():.4f}")

    loss.backward()
    print("Backward pass successful. Gradients computed.")

    print("\n4. Testing File I/O...")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({"test": "success"}, config.LATEST_MODEL_PATH)
    print(f"Successfully wrote to {config.LATEST_MODEL_PATH}")

    print("\n--- ALL TESTS PASSED! YOU ARE READY TO TRAIN ---")


if __name__ == "__main__":
    run_test()
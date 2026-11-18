"""
sanity_check.py
---------------
Pre-training sanity check validating the OpenCV localizer, model shapes,
and loss gradients.
"""

import sys
import glob
import traceback
import torch

import config


def check(step_name, fn):
    print(f"\n[{step_name}] ...", flush=True)
    try:
        result = fn()
        print(f"[{step_name}] PASS")
        return result
    except Exception:
        print(f"[{step_name}] FAILED")
        traceback.print_exc()
        sys.exit(1)


def check_device():
    print(f"  Device selected: {config.DEVICE}")
    return config.DEVICE


def check_cv2_crop():
    if not getattr(config, "USE_CV2_CROP", False):
        print("  Skipped (USE_CV2_CROP is False)")
        return None

    from cv2_crop import LeafLocalizer
    localizer = LeafLocalizer()

    sample_paths = glob.glob(f"{config.TRAIN_DIR}/*/*.jpg")[:5]
    if not sample_paths:
        raise ValueError(f"No sample .jpg files found under {config.TRAIN_DIR} to test cropping.")

    for p in sample_paths:
        img = localizer.crop_leaf(p)
        print(f"  {p} -> cropped size {img.size}")

    return localizer


def check_dataset():
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"  Classes found: {len(class_names)}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    imgs, labels = next(iter(train_loader))
    print(f"  One batch shape: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
    return train_loader, val_loader, test_loader, class_names


def check_model_forward_backward():
    from model import build_model
    from losses import get_loss_fn

    model = build_model()
    criterion = get_loss_fn(gamma=config.FOCAL_GAMMA, smoothing=config.LABEL_SMOOTHING)

    dummy_imgs = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    dummy_labels = torch.randint(0, config.NUM_CLASSES, (2,)).to(config.DEVICE)

    logits = model(dummy_imgs)
    loss = criterion(logits, dummy_labels)
    loss.backward()
    print(f"  Loss value: {loss.item():.4f} (backward pass succeeded)")
    return model


def main():
    print("=" * 70)
    print("PRE-TRAINING SANITY CHECK (OPENCV + BALANCED PIPELINE)")
    print("=" * 70)

    check("1. Device", check_device)
    check("2. OpenCV leaf cropping", check_cv2_crop)
    train_loader, val_loader, test_loader, class_names = check("3. Dataset loading", check_dataset)
    check("4. Model forward/backward", check_model_forward_backward)

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — ready to train.")
    print("=" * 70)


if __name__ == "__main__":
    main()
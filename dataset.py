"""
dataset.py
----------
Dataset pipeline with OpenCV leaf cropping and memory-safe imbalance handling.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

import config

CACHE_DIR = os.path.join(config.PROJECT_ROOT, "cv2_crop_cache")

if getattr(config, "USE_CV2_CROP", True):
    from cv2_crop import LeafLocalizer
    _localizer = LeafLocalizer()
else:
    _localizer = None


def _cache_path_for(path: str) -> str:
    rel = os.path.relpath(path, config.DATA_DIR)
    return os.path.join(CACHE_DIR, rel)


def _cv2_crop_loader(path: str):
    """Crops via OpenCV saliency with persistent disk caching."""
    if _localizer is None:
        return Image.open(path).convert("RGB")

    cache_path = _cache_path_for(path)
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    cropped = _localizer.crop_leaf(path).convert("RGB")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cropped.save(cache_path)
    return cropped


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class MemorySafeSMOTEDataset(Dataset):
    """Memory-safe SMOTE implementation that operates on thumbnail-downsampled

    tensors (32x32) to prevent Kaggle 30GB RAM exhaustion.
    """
    def __init__(self, base_dataset):
        from imblearn.over_sampling import SMOTE
        self.classes = base_dataset.classes
        print("[SMOTE] Generating compact representations to prevent OOM...")

        downsample = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        X_list, y_list = [], []
        # Sample directly without invoking full augmentations
        for path, label in base_dataset.samples:
            img = _cv2_crop_loader(path)
            X_list.append(downsample(img).view(-1).numpy())
            y_list.append(label)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        smote = SMOTE(sampling_strategy="auto", k_neighbors=3, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Upscale synthesized tensors back to target training dimensions
        tensors = torch.tensor(X_res, dtype=torch.float32).view(-1, 3, 32, 32)
        upsampler = transforms.Resize((config.IMG_SIZE, config.IMG_SIZE))
        self.samples = upsampler(tensors)
        self.targets = torch.tensor(y_res, dtype=torch.long)
        print(f"[SMOTE] Balanced training set from {len(y)} to {len(y_res)} samples.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def compute_class_weights(train_ds_or_loader, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    if hasattr(train_ds_or_loader, "samples") and isinstance(train_ds_or_loader.samples, list):
        for _, label in train_ds_or_loader.samples:
            counts[label] += 1
    elif hasattr(train_ds_or_loader, "targets"):
        for lbl in train_ds_or_loader.targets.numpy():
            counts[lbl] += 1
    else:
        for _, labels in train_ds_or_loader:
            for lbl in labels.numpy():
                counts[lbl] += 1

    counts = np.clip(counts, a_min=1, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders():
    loader_fn = _cv2_crop_loader if _localizer is not None else datasets.folder.default_loader

    train_ds = datasets.ImageFolder(
        config.TRAIN_DIR, transform=get_transforms(train=True), loader=loader_fn,
    )
    val_ds = datasets.ImageFolder(
        config.VAL_DIR, transform=get_transforms(train=False), loader=loader_fn,
    )
    test_ds = datasets.ImageFolder(
        config.TEST_DIR, transform=get_transforms(train=False), loader=loader_fn,
    )

    class_names = train_ds.classes
    num_classes = len(class_names)

    if getattr(config, "USE_SMOTE", False):
        final_train_ds = MemorySafeSMOTEDataset(train_ds)
        train_loader = DataLoader(
            final_train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.USE_PIN_MEMORY
        )
    elif getattr(config, "USE_WEIGHTED_SAMPLER", False):
        class_weights = compute_class_weights(train_ds, num_classes)
        sample_weights = [class_weights[label].item() for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
            num_workers=config.NUM_WORKERS, pin_memory=config.USE_PIN_MEMORY
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.USE_PIN_MEMORY
        )

    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.USE_PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.USE_PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, class_names
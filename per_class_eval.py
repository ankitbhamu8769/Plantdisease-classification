"""
per_class_eval.py
------------------
Per-class precision/recall/F1 breakdown — reveals whether minority classes
(e.g. your ~150-image classes) are underperforming relative to majority
classes (e.g. ~1400-image classes), which train.py's macro-averaged metrics
alone don't show.

Run with:  python per_class_eval.py
"""

import torch
from sklearn.metrics import classification_report

import config
from model import build_model
from dataset import get_dataloaders


def main():
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    class_names = checkpoint["class_names"]

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, _ = get_dataloaders()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(
        all_labels, all_preds, target_names=class_names,
        zero_division=0, digits=3,
    ))


if __name__ == "__main__":
    main()
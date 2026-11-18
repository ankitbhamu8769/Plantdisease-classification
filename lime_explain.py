"""
lime_explain.py
----------------
LIME — used here as a SECONDARY explanation method for a handful of example
images (config.NUM_LIME_SAMPLES), to visually cross-check Grad-CAM++'s regions.
LIME is much slower than Grad-CAM++ (it needs many perturbed forward passes per
image), so it is not meant to run over the whole test set.

Applies the same YOLO leaf-crop used at training time (if enabled in
config) before running inference, for consistency with gradcam++_explain.py
and the training pipeline.

Run with:  python lime_explain.py --image path/to/leaf.jpg
"""

import argparse
import numpy as np
import torch
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

import config
from model import build_model
from dataset import get_transforms

if getattr(config, "USE_CV2_CROP", False):
    from cv2_crop import LeafLocalizer
    _localizer = LeafLocalizer()
else:
    _localizer = None


def load_and_crop(image_path: str) -> Image.Image:
    """Applies the same YOLO leaf-crop used at training time, if enabled."""
    if _localizer is not None:
        return _localizer.crop_leaf(image_path).convert("RGB")
    return Image.open(image_path).convert("RGB")


def build_predict_fn(model, transform):
    """LIME needs a function: batch of HWC uint8 numpy images -> class probabilities."""

    def predict_fn(images_np):
        batch = []
        for img_np in images_np:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            batch.append(transform(pil_img))
        batch_tensor = torch.stack(batch).to(config.DEVICE)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


def explain_image(image_path: str, checkpoint_path: str = config.BEST_MODEL_PATH):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    class_names = checkpoint["class_names"]

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_transforms(train=False)

    cropped_img = load_and_crop(image_path).resize((config.IMG_SIZE, config.IMG_SIZE))
    img_np = np.array(cropped_img)

    predict_fn = build_predict_fn(model, transform)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=5, hide_rest=False
    )

    predicted_label = class_names[top_label]
    print(f"LIME explanation for predicted class: {predicted_label}")

    plt.figure(figsize=(5, 5))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME: {predicted_label}")
    plt.axis("off")

    out_path = f"{config.OUTPUT_DIR}/lime_{predicted_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved LIME explanation to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    args = parser.parse_args()

    explain_image(args.image, args.checkpoint)
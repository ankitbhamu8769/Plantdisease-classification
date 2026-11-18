"""
gradcam_explain.py
-------------------
Grad-CAM++ explanations — the primary XAI method for this project.

Fuses TWO target layers: a mid-backbone layer (higher spatial resolution,
finer localization) and a late-backbone layer (more semantically
disease-specific, but coarser resolution). Averaging the two gives sharper,
more lesion-aligned heatmaps than either layer alone — a single late-layer
hook tends to produce blobby, edge-biased heatmaps because by that depth the
feature map has shrunk to roughly 7x7 for a 224x224 input.

Applies the same YOLO leaf-crop used at training time (if enabled in
config) before running inference, so the model sees the same framing here
as it did during training, and so the heatmap aligns spatially with the
cropped image it was computed on.

Run with:  python gradcam_explain.py --image path/to/leaf.jpg
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

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


class GradCAMPlusPlus:
    """Single-layer Grad-CAM++. Used as a building block by
    MultiLayerGradCAMPlusPlus below."""

    def __init__(self, model, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute_cam(self, logits, class_idx):
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        alpha_num = grad_2
        alpha_denom = 2.0 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom

        weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class MultiLayerGradCAMPlusPlus:
    """Fuses Grad-CAM++ from two layers (mid + late backbone stages) into
    one sharper heatmap: average the normalized CAMs, then denoise and
    re-normalize."""

    def __init__(self, model, mid_layer: torch.nn.Module, late_layer: torch.nn.Module):
        self.model = model
        self.mid_cam = GradCAMPlusPlus(model, mid_layer)
        self.late_cam = GradCAMPlusPlus(model, late_layer)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        cam_mid = self.mid_cam.compute_cam(logits, class_idx)
        # Re-run forward for a clean autograd graph before the second backward pass
        logits2 = self.model(input_tensor)
        cam_late = self.late_cam.compute_cam(logits2, class_idx)

        fused = (cam_mid + cam_late) / 2.0
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

        # Denoise: blur + threshold weak activations so the overlay isn't
        # dominated by low-confidence background noise.
        fused = cv2.GaussianBlur(fused, (5, 5), sigmaX=0)
        fused[fused < 0.2] = 0
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

        confidence = logits.softmax(dim=1)[0, class_idx].item()

        self.mid_cam.remove_hooks()
        self.late_cam.remove_hooks()

        return fused, class_idx, confidence


def overlay_heatmap(cam, original_img_path, out_path):
    """Blend the CAM heatmap on top of the (cropped) image and save it."""
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)
    cv2.imwrite(out_path, overlay)
    print(f"Saved Grad-CAM++ overlay to {out_path}")


def explain_image(image_path: str, checkpoint_path: str = config.BEST_MODEL_PATH):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    class_names = checkpoint["class_names"]

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stages = list(model.backbone.backbone.blocks.children())
    mid_layer = stages[config.GRADCAM_MID_LAYER_STAGE]
    late_layer = stages[config.GRADCAM_LATE_LAYER_STAGE]

    cam_tool = MultiLayerGradCAMPlusPlus(model, mid_layer, late_layer)

    cropped_img = load_and_crop(image_path)
    transform = get_transforms(train=False)
    input_tensor = transform(cropped_img).unsqueeze(0).to(config.DEVICE)

    cam, class_idx, confidence = cam_tool.generate(input_tensor)

    predicted_label = class_names[class_idx]
    print(f"Prediction: {predicted_label}  (confidence={confidence:.2%})")

    # Save the cropped image so the overlay lines up with the CAM's
    # coordinate space.
    cropped_path = f"{config.OUTPUT_DIR}/_tmp_cropped_input.png"
    cropped_img.save(cropped_path)

    out_path = f"{config.OUTPUT_DIR}/gradcam_plus_plus_{predicted_label}.png"
    overlay_heatmap(cam, cropped_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    args = parser.parse_args()

    explain_image(args.image, args.checkpoint)
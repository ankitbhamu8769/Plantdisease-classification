"""
losses.py
---------
Focal Loss + Label Smoothing, with optional per-class alpha weighting.

The gamma term alone only down-weights EASY examples (regardless of which
class they belong to) — it does not correct for class-frequency imbalance.
The alpha_t term (from the original Focal Loss / RetinaNet paper) is what
actually compensates for rare classes having fewer training images: it
multiplies each sample's loss by a per-class weight derived from inverse
class frequency (see dataset.py's compute_class_weights).

Label smoothing on top prevents overconfident predictions on ambiguous
lesion/spot boundaries, independent of the class-balance question.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Args:
        gamma: focusing parameter — higher values down-weight easy examples more.
        smoothing: label smoothing epsilon (0.1 = spec default).
        class_weights: optional (num_classes,) tensor of per-class alpha
            weights (inverse class frequency). When provided, each sample's
            loss is multiplied by its true class's weight, so rare classes
            contribute proportionally more to the total loss/gradient.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Smoothed target distribution: (1 - eps) on the true class,
        # eps spread uniformly over the remaining classes.
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Gamma term: down-weights easy, well-classified examples.
        focal_weight = (1.0 - probs).pow(self.gamma)

        per_sample_loss = -(true_dist * focal_weight * log_probs).sum(dim=-1)

        # Alpha term: up-weights rare-class samples (class-frequency correction).
        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device)[targets]
            per_sample_loss = per_sample_loss * alpha_t

        return per_sample_loss.mean()


def get_loss_fn(gamma: float = 2.0, smoothing: float = 0.1,
                 class_weights: torch.Tensor = None) -> nn.Module:
    """Single entry point used by train.py and sanity_check.py.

    Example (train.py, with class balancing enabled):
        class_weights = compute_class_weights(train_ds, num_classes).to(config.DEVICE)
        criterion = get_loss_fn(gamma=config.FOCAL_GAMMA,
                                 smoothing=config.LABEL_SMOOTHING,
                                 class_weights=class_weights)

    Example (without class balancing — class_weights defaults to None):
        criterion = get_loss_fn(gamma=config.FOCAL_GAMMA, smoothing=config.LABEL_SMOOTHING)
    """
    return FocalLossWithLabelSmoothing(gamma=gamma, smoothing=smoothing,
                                        class_weights=class_weights)
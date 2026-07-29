"""
Tversky Loss.

As described in the manuscript:

    Tversky Loss = TP / (TP + alpha * FP + beta * FN)

with alpha = 0.3 and beta = 0.7. This asymmetric weighting penalizes
false negatives more heavily than false positives, prioritizing the
detection of small bronchial arteries.

The Tversky index generalizes the Dice coefficient by decoupling the
penalty on false positives (alpha) and false negatives (beta). When
alpha = beta = 0.5, it reduces to the Dice coefficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TverskyLoss(nn.Module):
    """
    Tversky loss for binary segmentation.

    Args:
        alpha: Weight for false positives (default: 0.3).
        beta: Weight for false negatives (default: 0.7).
        smooth: Smoothing factor to avoid division by zero.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, ...] raw logits (pre-sigmoid for binary).
            targets: [B, 1, ...] or [B, ...] binary ground truth.

        Returns:
            Scalar loss.
        """
        # Convert logits to probabilities
        if logits.shape[1] == 2:
            # Two-class output: take the foreground channel
            probs = F.softmax(logits, dim=1)[:, 1:2]
        else:
            probs = torch.sigmoid(logits)

        # Ensure targets match shape
        if targets.ndim < probs.ndim:
            targets = targets.unsqueeze(1)

        # Flatten
        probs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)

        # True positives, false positives, false negatives
        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Loss = 1 - Tversky index
        loss = 1.0 - tversky

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SoftTverskyLoss(nn.Module):
    """
    Soft Tversky loss that operates directly on probabilities.

    This variant does not apply sigmoid/softmax, expecting the caller
    to pass probabilities directly. Useful for deep supervision where
    intermediate outputs may already be activated.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim < probs.ndim:
            targets = targets.unsqueeze(1)

        probs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky).mean()

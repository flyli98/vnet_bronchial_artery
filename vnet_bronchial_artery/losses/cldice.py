"""
ClDice Loss (centerline Dice).

As described in the manuscript, ClDice enforces topological consistency
based on vessel centerline structure:

    ClDice = 2 * |Centerline(Y) ∩ Centerline(Y_hat)| /
                 (|Centerline(Y)| + |Centerline(Y_hat)|)

where Centerline(.) denotes the vessel centerline extracted via
skeletonization (soft-skeleton using iterative erosion/dilation).

The soft skeletonization is differentiable, enabling end-to-end training.

Reference: Shit et al., "clDice -- A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation" (CVPR 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Soft erosion using min pooling (3D).

    Erodes along each spatial axis separately via min-pooling, then takes
    the element-wise minimum across the three axes. This is the standard
    differentiable 3D erosion used in clDice.

    Args:
        img: [B, 1, D, H, W] probability map in [0, 1].
    """
    if img.ndim == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), 1, (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), 1, (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), 1, (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    elif img.ndim == 4:
        p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)
    else:
        raise ValueError(f"Unsupported tensor dimensionality: {img.ndim}")


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Soft dilation using max pooling (3D).

    Args:
        img: [B, 1, D, H, W] probability map in [0, 1].
    """
    if img.ndim == 5:
        return F.max_pool3d(img, (3, 3, 3), 1, (1, 1, 1))
    elif img.ndim == 4:
        return F.max_pool2d(img, (3, 3), 1, (1, 1))
    else:
        raise ValueError(f"Unsupported tensor dimensionality: {img.ndim}")


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft opening: erosion followed by dilation."""
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """
    Soft skeletonization via iterative open-residue.

    skeleton ≈ img - open(img), accumulated over iterations.

    Args:
        img: [B, 1, D, H, W] probability map.
        iter_: Number of iterations (controls skeleton thickness).
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)

    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)

    return skel


class ClDiceLoss(nn.Module):
    """
    Centerline Dice loss for tubular structure segmentation.

    Computes:
      1. Tprec = |skel(pred) ∩ target| / |skel(pred)|
      2. Tsens = |skel(target) ∩ pred| / |skel(target)|
      3. clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
      4. Loss = 1 - clDice

    Args:
        iter_: Number of skeletonization iterations.
        smooth: Smoothing factor.
        alpha: Weight for Tprec (precision).
        beta: Weight for Tsens (sensitivity).
    """

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1e-5,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, ...] raw logits.
            targets: [B, 1, ...] or [B, ...] binary ground truth.

        Returns:
            Scalar clDice loss.
        """
        # Convert to probabilities
        if logits.shape[1] == 2:
            probs = F.softmax(logits, dim=1)[:, 1:2]
        else:
            probs = torch.sigmoid(logits)

        if targets.ndim < probs.ndim:
            targets = targets.unsqueeze(1)

        # Compute skeletons
        skel_pred = soft_skel(probs, self.iter_)
        skel_target = soft_skel(targets, self.iter_)

        # Topology precision: fraction of predicted skeleton in target
        tprec = (
            (torch.sum(skel_pred * targets) + self.smooth)
            / (torch.sum(skel_pred) + self.smooth)
        )

        # Topology sensitivity: fraction of target skeleton in prediction
        tsens = (
            (torch.sum(skel_target * probs) + self.smooth)
            / (torch.sum(skel_target) + self.smooth)
        )

        # clDice
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)

        return 1.0 - cldice

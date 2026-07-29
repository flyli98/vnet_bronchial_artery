"""
Evaluation metrics for bronchial artery segmentation.

Implements:
  - Dice Similarity Coefficient (DSC)
  - 95th percentile Hausdorff Distance (95% HD)
  - Sensitivity (recall)
  - Specificity
  - Precision
"""

import numpy as np
from typing import Optional


def compute_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Compute Dice Similarity Coefficient.

    DSC = 2 * |pred ∩ gt| / (|pred| + |gt|)

    Args:
        pred: Binary prediction.
        gt: Binary ground truth.
        smooth: Smoothing factor.

    Returns:
        DSC value in [0, 1].
    """
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()

    if union == 0:
        return 1.0  # Both empty

    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.

    Uses scipy to compute the distance transform and then finds the
    95th percentile of the bidirectional surface distances.

    Args:
        pred: Binary prediction.
        gt: Binary ground truth.
        spacing: Voxel spacing (Z, Y, X) in mm.

    Returns:
        95% HD in mm. Returns NaN if either mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")

    # Distance from pred surface to gt surface
    dt_gt = distance_transform_edt(~gt_bool, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)

    # Surface voxels: boundary between mask and background
    pred_surface = pred_bool & ~_erode(pred_bool)
    gt_surface = gt_bool & ~_erode(gt_bool)

    # Bidirectional distances
    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Binary erosion (one voxel in all directions)."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(mask.ndim, mask.ndim)
    return binary_erosion(mask, structure=struct, border_value=1)


def compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Sensitivity (recall / true positive rate).

    Sens = TP / (TP + FN)
    """
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    if tp + fn == 0:
        return 1.0

    return float(tp / (tp + fn))


def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Specificity (true negative rate).

    Spec = TN / (TN + FP)
    """
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    tn = np.logical_and(~pred_bool, ~gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()

    if tn + fp == 0:
        return 1.0

    return float(tn / (tn + fp))


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Precision (positive predictive value).

    Prec = TP / (TP + FP)
    """
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Compute all segmentation metrics.

    Returns:
        Dictionary with keys: dice, hd95, sensitivity, specificity, precision.
    """
    return {
        "dice": compute_dice(pred, gt),
        "hd95": compute_hd95(pred, gt, spacing),
        "sensitivity": compute_sensitivity(pred, gt),
        "specificity": compute_specificity(pred, gt),
        "precision": compute_precision(pred, gt),
    }

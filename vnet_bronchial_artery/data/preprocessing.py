"""
Data preprocessing pipeline.

Preprocessing uses dataset intensity statistics and target spacing / patch
size from the configuration. The pipeline is tailored for the bronchial
artery CTA segmentation task.

Steps:
  1. Read image (SimpleITKIO)
  2. Resample to target spacing (trilinear for data,
     nearest-neighbor for segmentation)
  3. Intensity clipping using foreground percentiles (0.5th / 99.5th)
  4. Window width/level clipping (400 HU / 40 HU, per manuscript)
  5. ZScoreNormalization using foreground mean & std
  6. Optional cropping to patch size (with padding if needed)

The dataset intensity statistics provide:
  - mean: 145.38, std: 107.59
  - percentile_00_5: -207, percentile_99_5: 511
  - min: -1006, max: 828
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional

from config.plan_config import DatasetConfig, PlanConfig, IntensityProperties


# ---------------------------------------------------------------------------
# Image I/O (SimpleITKIO)
# ---------------------------------------------------------------------------
def read_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a medical image using SimpleITK.

    Returns:
        (data, spacing) where data is [C, D, H, W] float32 and
        spacing is [Z, Y, X] in mm.
    """
    sitk_img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(sitk_img)  # [D, H, W]
    spacing = np.array(sitk_img.GetSpacing()[::-1])  # reverse to [Z, Y, X]
    origin = np.array(sitk_img.GetOrigin()[::-1])
    direction = np.array(sitk_img.GetDirection())

    # Add channel dimension: [1, D, H, W]
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    data = data.astype(np.float32)

    return data, spacing


def read_segmentation(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a segmentation label image."""
    sitk_img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(sitk_img)  # [D, H, W]
    spacing = np.array(sitk_img.GetSpacing()[::-1])

    if data.ndim == 3:
        data = data[np.newaxis, ...]
    data = data.astype(np.int32)

    return data, spacing


def save_image(
    data: np.ndarray,
    spacing: np.ndarray,
    path: str,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Save a numpy array back to a medical image format."""
    if data.ndim == 4:
        data = data[0]  # remove channel dim
    sitk_img = sitk.GetImageFromArray(data)
    sitk_img.SetSpacing(spacing[::-1].tolist())  # back to [X, Y, Z]
    sitk_img.SetOrigin(origin)
    sitk.WriteImage(sitk_img, path)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def compute_new_shape(
    original_shape: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray,
) -> np.ndarray:
    """Compute the shape after resampling to target spacing."""
    new_shape = np.round(
        original_shape * (original_spacing / target_spacing)
    ).astype(int)
    return new_shape


def resample_data(
    data: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray,
    is_seg: bool = False,
    order: int = 3,
) -> np.ndarray:
    """
    Resample 3D data to target spacing.

    Uses scipy.ndimage.zoom for resampling. For segmentation, uses
    nearest-neighbor (order=0 or 1) to preserve label values.

    Args:
        data: [C, D, H, W] array
        original_spacing: [Z, Y, X] in mm
        target_spacing: [Z, Y, X] in mm
        is_seg: If True, use order=1 (no interpolation artifacts)
        order: Spline order for data (3 = cubic), overridden for seg.
    """
    from scipy.ndimage import zoom

    if is_seg:
        order = 1

    zoom_factors = (original_spacing / target_spacing).astype(float)
    # Don't zoom the channel dimension
    zoom_factors = np.insert(zoom_factors, 0, 1.0)

    resampled = zoom(data, zoom_factors, order=order, mode="nearest")
    return resampled.astype(np.float32 if not is_seg else np.int32)


# ---------------------------------------------------------------------------
# Intensity normalization (using dataset foreground statistics)
# ---------------------------------------------------------------------------
def apply_window_level(
    data: np.ndarray,
    window_width: float = 400.0,
    window_level: float = 40.0,
) -> np.ndarray:
    """
    Apply CT window width/level clipping.

    Per the manuscript: window width = 400 HU, window level = 40 HU.
    This clips intensities to [level - width/2, level + width/2].
    """
    lower = window_level - window_width / 2.0  # -160
    upper = window_level + window_width / 2.0  # 240
    data = np.clip(data, lower, upper)
    return data


def zscore_normalize(
    data: np.ndarray,
    intensity_props: IntensityProperties,
    use_percentile_clipping: bool = True,
    clip_to_window: bool = True,
) -> np.ndarray:
    """
    Z-score normalization using dataset foreground intensity properties.

    Pipeline:
      1. (Optional) Clip to foreground percentiles [0.5th, 99.5th]
      2. (Optional) Apply CT window width/level (400 HU / 40 HU)
      3. Z-score normalize: (x - mean) / std

    The mean (145.38) and std (107.59) come from the dataset foreground
    intensity statistics.

    Args:
        data: [C, D, H, W] float32 array (raw HU values)
        intensity_props: IntensityProperties with foreground statistics
        use_percentile_clipping: Clip to [percentile_00_5, percentile_99_5]
        clip_to_window: Also apply CT window (400/40 HU) from manuscript
    """
    # Step 1: Percentile clipping (foreground statistics)
    if use_percentile_clipping:
        p_low = intensity_props.percentile_00_5   # -207.0
        p_high = intensity_props.percentile_99_5  # 511.0
        data = np.clip(data, p_low, p_high)

    # Step 2: CT window width/level (from manuscript)
    if clip_to_window:
        data = apply_window_level(data, window_width=400.0, window_level=40.0)

    # Step 3: Z-score normalization (foreground statistics)
    mean = intensity_props.mean  # 145.38
    std = intensity_props.std    # 107.59
    data = (data - mean) / max(std, 1e-8)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Cropping / padding to patch size
# ---------------------------------------------------------------------------
def crop_and_pad_to_patch(
    data: np.ndarray,
    seg: Optional[np.ndarray],
    patch_size: Tuple[int, int, int],
    crop_offset: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]:
    """
    Crop or pad data to patch_size.

    If the data is larger than patch_size, crop (optionally at a given offset).
    If smaller, pad with zeros (data) or -1 (seg to mark padded regions).

    Returns:
        cropped_data, cropped_seg, offset_used
    """
    d, h, w = data.shape[-3:]
    pd, ph, pw = patch_size

    # Compute crop offsets
    if crop_offset is None:
        # Random crop for training, center crop for inference
        offset_d = max(0, (d - pd) // 2)
        offset_h = max(0, (h - ph) // 2)
        offset_w = max(0, (w - pw) // 2)
    else:
        offset_d, offset_h, offset_w = crop_offset

    # Crop
    slices_d = slice(offset_d, min(offset_d + pd, d))
    slices_h = slice(offset_h, min(offset_h + ph, h))
    slices_w = slice(offset_w, min(offset_w + pw, w))

    if data.ndim == 4:
        cropped_data = data[:, slices_d, slices_h, slices_w]
        if seg is not None:
            cropped_seg = seg[:, slices_d, slices_h, slices_w]
        else:
            cropped_seg = None
    else:
        cropped_data = data[slices_d, slices_h, slices_w]
        if seg is not None:
            cropped_seg = seg[slices_d, slices_h, slices_w]
        else:
            cropped_seg = None

    # Pad if needed
    actual_d = min(pd, d) if d < pd else pd
    actual_h = min(ph, h) if h < ph else ph
    actual_w = min(pw, w) if w < pw else pw

    pad_d = max(0, pd - cropped_data.shape[-3])
    pad_h = max(0, ph - cropped_data.shape[-2])
    pad_w = max(0, pw - cropped_data.shape[-1])

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        if data.ndim == 4:
            pad_width = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
        cropped_data = np.pad(cropped_data, pad_width, mode="constant", constant_values=0)
        if cropped_seg is not None:
            cropped_seg = np.pad(cropped_seg, pad_width, mode="constant", constant_values=-1)

    return cropped_data, cropped_seg, (offset_d, offset_h, offset_w)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_case(
    image_path: str,
    seg_path: Optional[str],
    dataset_config: DatasetConfig,
    plan_config: PlanConfig,
    crop_offset: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Full preprocessing for a single case.

    Pipeline:
      1. Read image (and optional segmentation)
      2. Resample to target spacing from plan_config
      3. Z-score normalize using foreground intensity statistics
      4. Crop/pad to patch size

    Args:
        image_path: Path to the CTA image file (.nii / .nii.gz)
        seg_path: Path to the segmentation label file (or None)
        dataset_config: DatasetConfig with foreground intensity properties
        plan_config: PlanConfig with target spacing & patch size
        crop_offset: Optional crop offset (random for train, None for center)

    Returns:
        data: [1, D, H, W] normalized float32
        seg: [1, D, H, W] int32 (or None)
    """
    # 1. Read
    data, original_spacing = read_image(image_path)

    if seg_path is not None:
        seg, _ = read_segmentation(seg_path)
    else:
        seg = None

    # 2. Resample to target spacing
    target_spacing = np.array(plan_config.spacing)
    data = resample_data(data, original_spacing, target_spacing, is_seg=False, order=3)
    if seg is not None:
        seg = resample_data(seg, original_spacing, target_spacing, is_seg=True)

    # 3. Z-score normalization (foreground statistics)
    data = zscore_normalize(
        data,
        dataset_config.intensity_properties,
        use_percentile_clipping=True,
        clip_to_window=True,
    )

    # 4. Crop/pad to patch size
    patch_size = tuple(plan_config.patch_size)
    data, seg, _ = crop_and_pad_to_patch(data, seg, patch_size, crop_offset)

    return data, seg

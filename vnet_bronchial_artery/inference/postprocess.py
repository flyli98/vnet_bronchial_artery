"""
Post-processing for bronchial artery segmentation.

As described in the manuscript:
  1. Remove 3D connected components with volume < 10 voxels
  2. Apply morphological closing (spherical structuring element, radius=1)
"""

import numpy as np
from scipy.ndimage import label, binary_closing, generate_binary_structure


def remove_small_components(
    mask: np.ndarray,
    min_volume: int = 10,
) -> np.ndarray:
    """
    Remove small 3D connected components from the segmentation.

    Only connected components with at least ``min_volume`` voxels are
    retained as segmented vessels.

    Args:
        mask: Binary segmentation mask.
        min_volume: Minimum number of voxels per component (default: 10).

    Returns:
        Cleaned binary mask.
    """
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask

    # 3D connectivity (26-connectivity)
    struct = generate_binary_structure(3, 3)
    labeled, n_components = label(mask_bool, structure=struct)

    if n_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # Count voxels per component
    component_sizes = np.bincount(labeled.ravel())

    # Keep only components >= min_volume (skip background = 0)
    keep = component_sizes >= min_volume
    keep[0] = False  # background

    cleaned = keep[labeled]

    return cleaned.astype(np.uint8)


def morphological_closing(
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """
    Apply morphological closing to fill small discontinuities.

    Uses a spherical (actually cubic for 3D) structuring element with
    the given radius. This fills small gaps in the vessel segmentation
    output.

    Args:
        mask: Binary segmentation mask.
        radius: Structuring element radius in voxels (default: 1).

    Returns:
        Closed binary mask.
    """
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask

    # Spherical structuring element approximation
    # For radius=1, this is a 3x3x3 cube (26-connectivity)
    struct = generate_binary_structure(3, 3)

    if radius > 1:
        # Dilate the structuring element for larger radii
        from scipy.ndimage import binary_dilation
        for _ in range(radius - 1):
            struct = binary_dilation(struct, structure=struct)

    closed = binary_closing(mask_bool, structure=struct, border_value=0)

    return closed.astype(np.uint8)


def postprocess_segmentation(
    mask: np.ndarray,
    min_volume: int = 10,
    closing_radius: int = 1,
) -> np.ndarray:
    """
    Full post-processing pipeline.

    1. Remove small connected components (< 10 voxels)
    2. Morphological closing (radius = 1 voxel)

    This matches the post-processing described in the manuscript:
    "only 3D connected components with a volume of at least 10 voxels
    were retained... a morphological closing operation (using a spherical
    structuring element with a radius of 1 voxel) was applied."

    Args:
        mask: Binary segmentation mask [1, D, H, W] or [D, H, W].
        min_volume: Minimum component volume in voxels.
        closing_radius: Structuring element radius for closing.

    Returns:
        Post-processed binary mask (same shape as input).
    """
    # Handle [1, D, H, W] format
    squeeze = False
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
        squeeze = True

    # Step 1: Remove small components
    mask = remove_small_components(mask, min_volume=min_volume)

    # Step 2: Morphological closing
    mask = morphological_closing(mask, radius=closing_radius)

    if squeeze:
        mask = mask[np.newaxis, ...]

    return mask

"""
Data augmentation for V-Net training.

Implements the augmentations described in the manuscript:
  1. Horizontal flipping (left-right mirroring)
  2. Vertical flipping (coronal-plane mirroring)
  3. Image scaling (factor range: 0.8 - 1.2)
  4. Random cropping

Additional augmentations:
  5. Gamma correction
  6. Gaussian noise
  7. Brightness adjustment

All augmentations are applied jointly to data and segmentation.
"""

import numpy as np
from typing import Tuple, Optional
import random


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            data, seg = t(data, seg)
        return data, seg


class RandomHorizontalFlip:
    """Flip along the W axis (left-right mirroring)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            data = np.flip(data, axis=-1).copy()
            seg = np.flip(seg, axis=-1).copy()
        return data, seg


class RandomVerticalFlip:
    """Flip along the H axis (coronal-plane mirroring)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            data = np.flip(data, axis=-2).copy()
            seg = np.flip(seg, axis=-2).copy()
        return data, seg


class RandomScaling:
    """
    Scale the image by a random factor in [0.8, 1.2].

    Uses scipy.ndimage.zoom. For segmentation, uses nearest-neighbor.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            from scipy.ndimage import zoom

            scale = random.uniform(*self.scale_range)
            # Scale only spatial dims, not channel
            zoom_factors = (1.0, scale, scale, scale)

            data = zoom(data, zoom_factors, order=3, mode="nearest")
            seg = zoom(seg, zoom_factors, order=0, mode="nearest")

        return data.astype(np.float32), seg.astype(np.int32)


class RandomCrop3D:
    """
    Random crop to patch_size. If the image is smaller than patch_size,
    pads with zeros (data) / -1 (seg).
    """

    def __init__(self, patch_size: Tuple[int, int, int]):
        self.patch_size = patch_size

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from data.preprocessing import crop_and_pad_to_patch

        # Random crop offset
        d, h, w = data.shape[-3:]
        pd, ph, pw = self.patch_size

        offset_d = random.randint(0, max(0, d - pd))
        offset_h = random.randint(0, max(0, h - ph))
        offset_w = random.randint(0, max(0, w - pw))

        data, seg, _ = crop_and_pad_to_patch(
            data, seg, self.patch_size, (offset_d, offset_h, offset_w)
        )
        return data, seg


class RandomGamma:
    """Gamma intensity augmentation (data only)."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.15):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            # data may have negative values after z-score normalization,
            # so shift to [0, 1] range, apply gamma, shift back
            d_min = data.min()
            d_max = data.max()
            d_range = max(d_max - d_min, 1e-8)
            normalized = (data - d_min) / d_range
            augmented = np.power(normalized, gamma)
            data = augmented * d_range + d_min
        return data.astype(np.float32), seg


class RandomGaussianNoise:
    """Add Gaussian noise to the data (seg unchanged)."""

    def __init__(self, noise_std: float = 0.1, p: float = 0.15):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, data.shape).astype(np.float32)
            data = data + noise
        return data.astype(np.float32), seg


class RandomBrightness:
    """Add a constant offset to the data (seg unchanged)."""

    def __init__(self, brightness_range: Tuple[float, float] = (-0.1, 0.1), p: float = 0.15):
        self.brightness_range = brightness_range
        self.p = p

    def __call__(self, data: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            offset = random.uniform(*self.brightness_range)
            data = data + offset
        return data.astype(np.float32), seg


def get_default_augmentation(
    patch_size: Tuple[int, int, int],
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> Compose:
    """
    Build the default augmentation pipeline.

    Augmentations from the manuscript:
      - Horizontal flip (left-right mirroring)
      - Vertical flip (coronal-plane mirroring)
      - Scaling (0.8 - 1.2)
      - Cropping (to patch_size)

    Additional augmentations:
      - Gamma correction
      - Gaussian noise
      - Brightness adjustment
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomScaling(scale_range=scale_range, p=0.3),
        RandomCrop3D(patch_size=patch_size),
        RandomGamma(gamma_range=(0.7, 1.5), p=0.15),
        RandomGaussianNoise(noise_std=0.1, p=0.15),
        RandomBrightness(brightness_range=(-0.1, 0.1), p=0.15),
    ])

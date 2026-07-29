"""
V-Net inference predictor.

Implements sliding-window inference for full-volume prediction, with
overlap-based stitching and optional test-time augmentation (TTA).

For cascade inference, supports bbox-restricted prediction where only
the region within the coarse model's bounding box is processed.
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from config.plan_config import PlanConfig, DatasetConfig
from models.vnet import build_vnet
from data.preprocessing import (
    read_image,
    resample_data,
    zscore_normalize,
    save_image,
)


class VNetPredictor:
    """
    Sliding-window inference predictor for V-Net.

    Args:
        plan_config: PlanConfig (e.g. 3d_fullres or 3d_lowres).
        dataset_config: DatasetConfig with foreground intensity properties.
        checkpoint_path: Path to model checkpoint.
        device: torch device.
        overlap: Overlap fraction between patches (0.0 - 0.5).
        use_tta: Enable test-time augmentation (horizontal/vertical flip).
    """

    def __init__(
        self,
        plan_config: PlanConfig,
        dataset_config: DatasetConfig,
        checkpoint_path: str,
        device: str = "cuda",
        overlap: float = 0.25,
        use_tta: bool = False,
    ):
        self.plan_config = plan_config
        self.dataset_config = dataset_config
        self.device = torch.device(device)
        self.patch_size = tuple(plan_config.patch_size)
        self.overlap = overlap
        self.use_tta = use_tta

        # Build and load model
        self.model = build_vnet(
            plan_config=plan_config,
            num_classes=2,
            deep_supervision=False,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def _predict_patch(self, patch: torch.Tensor) -> np.ndarray:
        """
        Predict a single patch.

        Returns:
            Probability map [1, D, H, W] for the foreground class.
        """
        patch = patch.unsqueeze(0).to(self.device)  # [1, 1, D, H, W]
        output = self.model(patch)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Softmax for 2-class output
        if output.shape[1] == 2:
            probs = torch.softmax(output, dim=1)[:, 1:2]
        else:
            probs = torch.sigmoid(output)

        return probs.squeeze().cpu().numpy()

    def _predict_patch_tta(self, patch: torch.Tensor) -> np.ndarray:
        """
        Predict a single patch with test-time augmentation.

        Averages predictions over original + horizontal flip + vertical flip.
        """
        pred_orig = self._predict_patch(patch)

        # Horizontal flip (W axis)
        patch_hflip = torch.flip(patch, dims=[-1])
        pred_hflip = self._predict_patch(patch_hflip)
        pred_hflip = np.flip(pred_hflip, axis=-1).copy()

        # Vertical flip (H axis)
        patch_vflip = torch.flip(patch, dims=[-2])
        pred_vflip = self._predict_patch(patch_vflip)
        pred_vflip = np.flip(pred_vflip, axis=-2).copy()

        return (pred_orig + pred_hflip + pred_vflip) / 3.0

    def predict_volume(
        self,
        data: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sliding-window prediction on a preprocessed volume.

        Args:
            data: [1, D, H, W] preprocessed (normalized) volume.
            bbox: Optional [d_min, d_max, h_min, h_max, w_min, w_max].
                If provided, only predict within this region.

        Returns:
            Probability map [1, D, H, W] at the same resolution as data.
        """
        d, h, w = data.shape[-3:]
        pd, ph, pw = self.patch_size

        # Determine prediction region
        if bbox is not None:
            d_min, d_max = max(0, bbox[0]), min(d, bbox[1])
            h_min, h_max = max(0, bbox[2]), min(h, bbox[3])
            w_min, w_max = max(0, bbox[4]), min(w, bbox[5])
        else:
            d_min, d_max = 0, d
            h_min, h_max = 0, h
            w_min, w_max = 0, w

        region_d = d_max - d_min
        region_h = h_max - h_min
        region_w = w_max - w_min

        # Initialize output
        output = np.zeros((1, d, h, w), dtype=np.float32)
        count = np.zeros((1, d, h, w), dtype=np.float32)

        # Compute step sizes
        step_d = max(1, int(pd * (1 - self.overlap)))
        step_h = max(1, int(ph * (1 - self.overlap)))
        step_w = max(1, int(pw * (1 - self.overlap)))

        # Generate patch positions
        d_positions = self._get_positions(region_d, pd, step_d)
        h_positions = self._get_positions(region_h, ph, step_h)
        w_positions = self._get_positions(region_w, pw, step_w)

        for dz in d_positions:
            for hz in h_positions:
                for wz in w_positions:
                    # Global coordinates
                    d_start = d_min + dz
                    h_start = h_min + hz
                    w_start = w_min + wz

                    d_end = min(d_start + pd, d_max)
                    h_end = min(h_start + ph, h_max)
                    w_end = min(w_start + pw, w_max)

                    # Extract patch (with padding if needed)
                    patch_d = d_end - d_start
                    patch_h = h_end - h_start
                    patch_w = w_end - w_start

                    patch = data[:, d_start:d_end, h_start:h_end, w_start:w_end]

                    # Pad to patch_size if at boundary
                    if (patch_d, patch_h, patch_w) != self.patch_size:
                        pad_d = pd - patch_d
                        pad_h = ph - patch_h
                        pad_w = pw - patch_w
                        patch = np.pad(
                            patch,
                            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                            mode="constant",
                        )

                    patch_tensor = torch.from_numpy(patch.astype(np.float32))

                    # Predict
                    if self.use_tta:
                        pred = self._predict_patch_tta(patch_tensor)
                    else:
                        pred = self._predict_patch(patch_tensor)

                    # Remove padding from prediction
                    pred = pred[:patch_d, :patch_h, :patch_w]

                    # Accumulate
                    output[:, d_start:d_end, h_start:h_end, w_start:w_end] += pred
                    count[:, d_start:d_end, h_start:h_end, w_start:w_end] += 1.0

        # Average overlapping regions
        count[count == 0] = 1.0
        output = output / count

        return output

    @staticmethod
    def _get_positions(region_size: int, patch_size: int, step: int) -> list:
        """Generate starting positions for sliding window."""
        if region_size <= patch_size:
            return [0]

        positions = list(range(0, region_size - patch_size + 1, step))
        # Ensure the last patch covers the end
        if positions[-1] + patch_size < region_size:
            positions.append(region_size - patch_size)
        return positions

    def predict_full_volume(self, image_path: str) -> np.ndarray:
        """
        Full prediction pipeline: read -> preprocess -> predict -> resample back.

        Args:
            image_path: Path to the CTA image.

        Returns:
            Binary segmentation [1, D, H, W] at the plan's target spacing.
        """
        # 1. Read
        data, original_spacing = read_image(image_path)

        # 2. Resample to target spacing
        target_spacing = np.array(self.plan_config.spacing)
        data = resample_data(data, original_spacing, target_spacing, is_seg=False, order=3)

        # 3. Normalize
        data = zscore_normalize(
            data,
            self.dataset_config.intensity_properties,
            use_percentile_clipping=True,
            clip_to_window=True,
        )

        # 4. Predict
        pred = self.predict_volume(data)

        # 5. Threshold
        binary = (pred > 0.5).astype(np.uint8)

        return binary

    def predict_with_bbox(
        self,
        image_path: str,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Prediction with optional bounding box restriction (for cascade).

        Args:
            image_path: Path to the CTA image.
            bbox: Optional bounding box at the fullres target spacing.

        Returns:
            Binary segmentation [1, D, H, W] at the plan's target spacing.
        """
        # 1. Read
        data, original_spacing = read_image(image_path)

        # 2. Resample to target spacing
        target_spacing = np.array(self.plan_config.spacing)
        data = resample_data(data, original_spacing, target_spacing, is_seg=False, order=3)

        # 3. Normalize
        data = zscore_normalize(
            data,
            self.dataset_config.intensity_properties,
            use_percentile_clipping=True,
            clip_to_window=True,
        )

        # 4. Predict (with bbox if provided)
        pred = self.predict_volume(data, bbox=bbox)

        # 5. Threshold
        binary = (pred > 0.5).astype(np.uint8)

        return binary

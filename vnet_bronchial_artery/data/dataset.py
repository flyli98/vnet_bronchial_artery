"""
PyTorch Dataset for bronchial artery segmentation.

Combines preprocessing (using dataset intensity statistics) and
augmentation (from the manuscript) into a single Dataset class that
supports both the coarse (3d_lowres) and fine (3d_fullres) cascade stages.

For the fine stage, sampling can be restricted to the bounding box
derived from the coarse model's prediction (cascade learning).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable

from config.plan_config import DatasetConfig, PlanConfig
from data.preprocessing import (
    read_image,
    read_segmentation,
    resample_data,
    zscore_normalize,
    crop_and_pad_to_patch,
    compute_new_shape,
)
from data.augmentation import get_default_augmentation, Compose


def find_data_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Find (image, label) pairs in the standard directory structure.

    Expected layout:
        data_dir/
            imagesTr/  case_0000.nii.gz, case_0001.nii.gz, ...
            labelsTr/  case_0000.nii.gz, case_0001.nii.gz, ...

    Returns:
        List of (image_path, label_path) tuples.
    """
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    if not os.path.isdir(images_dir):
        # Fallback: look for images directly
        images_dir = data_dir
        labels_dir = data_dir

    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.endswith((".nii", ".nii.gz")) and not fname.startswith("."):
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, fname)
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
            else:
                pairs.append((img_path, None))

    return pairs


class BraSegDataset(Dataset):
    """
    Bronchial Artery Segmentation Dataset.

    Preprocessing:
      1. Read image & label (SimpleITKIO)
      2. Resample to target spacing (from plan_config)
      3. Z-score normalize (using foreground intensity statistics)
      4. Crop/pad to patch size

    For training: random crop + augmentation.
    For validation: center crop, no augmentation.
    For cascade fine stage: optionally restrict sampling to bounding box
    from the coarse model prediction.
    """

    def __init__(
        self,
        data_pairs: List[Tuple[str, Optional[str]]],
        dataset_config: DatasetConfig,
        plan_config: PlanConfig,
        is_training: bool = True,
        augment: bool = True,
        bbox_provider: Optional[Callable[[str], np.ndarray]] = None,
        bbox_expansion: int = 20,
    ):
        """
        Args:
            data_pairs: List of (image_path, label_path) tuples.
            dataset_config: DatasetConfig with foreground intensity properties.
            plan_config: PlanConfig with target spacing & patch size.
            is_training: Whether this is for training (affects cropping).
            augment: Whether to apply data augmentation.
            bbox_provider: For cascade fine stage: a function that takes
                the image path and returns a bounding box [d_min, d_max,
                h_min, h_max, w_min, w_max] from the coarse model.
            bbox_expansion: Number of voxels to expand the bbox in each
                direction (manuscript: 20 voxels).
        """
        self.data_pairs = data_pairs
        self.dataset_config = dataset_config
        self.plan_config = plan_config
        self.is_training = is_training
        self.patch_size = tuple(plan_config.patch_size)
        self.target_spacing = np.array(plan_config.spacing)
        self.bbox_provider = bbox_provider
        self.bbox_expansion = bbox_expansion

        if augment and is_training:
            self.augmentation = get_default_augmentation(self.patch_size)
        else:
            self.augmentation = None

        # Cache for preprocessed data (lazy loading)
        self._cache = {}

    def __len__(self) -> int:
        return len(self.data_pairs)

    def _load_and_preprocess(
        self, image_path: str, label_path: Optional[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load and preprocess a single case (without cropping)."""
        # Read
        data, original_spacing = read_image(image_path)
        if label_path is not None:
            seg, _ = read_segmentation(label_path)
        else:
            seg = None

        # Resample to target spacing
        data = resample_data(data, original_spacing, self.target_spacing,
                             is_seg=False, order=3)
        if seg is not None:
            seg = resample_data(seg, original_spacing, self.target_spacing,
                                is_seg=True)

        # Z-score normalize (foreground statistics)
        data = zscore_normalize(
            data,
            self.dataset_config.intensity_properties,
            use_percentile_clipping=True,
            clip_to_window=True,
        )

        return data, seg

    def _get_bbox_offset(self, image_path: str, data_shape: tuple) -> Tuple[int, int, int]:
        """
        Compute crop offset based on bounding box (for cascade fine stage).

        If bbox_provider is set, the offset is centered on the bbox.
        Otherwise, random (training) or center (validation) offset.
        """
        d, h, w = data_shape[-3:]
        pd, ph, pw = self.patch_size

        if self.bbox_provider is not None:
            bbox = self.bbox_provider(image_path)
            # bbox = [d_min, d_max, h_min, h_max, w_min, w_max]
            d_center = (bbox[0] + bbox[1]) // 2
            h_center = (bbox[2] + bbox[3]) // 2
            w_center = (bbox[4] + bbox[5]) // 2

            offset_d = max(0, min(d_center - pd // 2, d - pd))
            offset_h = max(0, min(h_center - ph // 2, h - ph))
            offset_w = max(0, min(w_center - pw // 2, w - pw))
        elif self.is_training:
            offset_d = np.random.randint(0, max(1, d - pd))
            offset_h = np.random.randint(0, max(1, h - ph))
            offset_w = np.random.randint(0, max(1, w - pw))
        else:
            offset_d = max(0, (d - pd) // 2)
            offset_h = max(0, (h - ph) // 2)
            offset_w = max(0, (w - pw) // 2)

        return int(offset_d), int(offset_h), int(offset_w)

    def __getitem__(self, idx: int) -> dict:
        image_path, label_path = self.data_pairs[idx]

        # Load and preprocess
        data, seg = self._load_and_preprocess(image_path, label_path)

        # Crop to patch size
        offset = self._get_bbox_offset(image_path, data.shape)
        data, seg, _ = crop_and_pad_to_patch(data, seg, self.patch_size, offset)

        # Augmentation (training only)
        if self.augmentation is not None and seg is not None:
            data, seg = self.augmentation(data, seg)

        # Convert to tensors
        data_tensor = torch.from_numpy(data.astype(np.float32))
        if seg is not None:
            # Convert label to binary: 0 = background, 1 = bronchial artery
            seg_tensor = torch.from_numpy((seg > 0).astype(np.float32))
        else:
            seg_tensor = torch.zeros_like(data_tensor)

        return {
            "data": data_tensor,
            "target": seg_tensor,
            "image_path": image_path,
        }


class CascadeFineDataset(BraSegDataset):
    """
    Dataset for the fine stage of cascade learning.

    Sampling is restricted to the expanded 3D bounding box derived from
    the coarse model's prediction, as described in the manuscript:
    "Sampling for the fine network is restricted to the expanded 3D
    bounding box (extended by 20 voxels in all directions)."
    """

    def __init__(
        self,
        data_pairs: List[Tuple[str, Optional[str]]],
        dataset_config: DatasetConfig,
        plan_config: PlanConfig,
        coarse_predictions_dir: str,
        is_training: bool = True,
        bbox_expansion: int = 20,
    ):
        """
        Args:
            coarse_predictions_dir: Directory containing coarse model
                predictions (one .nii.gz per case).
            bbox_expansion: Voxels to expand bbox in each direction.
        """
        self.coarse_predictions_dir = coarse_predictions_dir
        super().__init__(
            data_pairs=data_pairs,
            dataset_config=dataset_config,
            plan_config=plan_config,
            is_training=is_training,
            augment=is_training,
            bbox_provider=self._get_bbox_from_coarse,
            bbox_expansion=bbox_expansion,
        )

    def _get_bbox_from_coarse(self, image_path: str) -> np.ndarray:
        """
        Extract bounding box from the coarse model's prediction.

        Returns: [d_min, d_max, h_min, h_max, w_min, w_max]
        """
        case_name = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
        coarse_path = os.path.join(self.coarse_predictions_dir, f"{case_name}.nii.gz")

        if not os.path.exists(coarse_path):
            # Fallback: return full volume bbox
            return np.array([0, -1, 0, -1, 0, -1])

        coarse_seg, _ = read_segmentation(coarse_path)
        mask = coarse_seg[0] > 0  # [D, H, W]

        if mask.sum() == 0:
            return np.array([0, -1, 0, -1, 0, -1])

        d_indices, h_indices, w_indices = np.where(mask)
        exp = self.bbox_expansion

        return np.array([
            max(0, d_indices.min() - exp),
            d_indices.max() + exp,
            max(0, h_indices.min() - exp),
            h_indices.max() + exp,
            max(0, w_indices.min() - exp),
            w_indices.max() + exp,
        ])

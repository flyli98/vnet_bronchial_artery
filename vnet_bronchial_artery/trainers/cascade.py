"""
Cascade learning pipeline.

Implements the two-stage cascade learning strategy described in the
manuscript:

  Stage 1 (Coarse): 3D V-Net at low resolution
    - Resampled to coarse spacing (3d_lowres)
    - Patch size: [128, 128, 128]
    - Global random sampling on the whole CT volume
    - Produces a coarse segmentation mask (ROI localization)

  Stage 2 (Fine): 3D V-Net at full resolution
    - Resampled to fine spacing (3d_fullres)
    - Patch size: [128, 128, 128]
    - Sampling restricted to expanded 3D bounding box (extended by 20
      voxels in all directions) derived from the coarse mask
    - Refines vessel boundary details

During inference:
  1. Coarse model generates localization mask
  2. Bounding box extracted and expanded
  3. Fine model processes the cropped region
  4. Result placed back into original image space

During training:
  The two models are trained independently using the same ground truth
  labels at their respective resolutions.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
from pathlib import Path

from config.plan_config import DatasetConfig, PlanConfig
from data.dataset import BraSegDataset, CascadeFineDataset, find_data_pairs
from data.preprocessing import (
    read_image,
    read_segmentation,
    resample_data,
    save_image,
)
from trainers.trainer import VNetTrainer
from inference.predictor import VNetPredictor
from inference.postprocess import postprocess_segmentation


class CascadePipeline:
    """
    Two-stage cascade learning pipeline (coarse -> fine).

    Orchestrates:
      1. Training the coarse model (3d_lowres)
      2. Running coarse inference to generate bounding boxes
      3. Training the fine model (3d_fullres) with bbox-restricted sampling
      4. Full cascade inference (coarse -> fine)
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./cascade_output",
        device: str = "cuda",
        n_folds: int = 5,
    ):
        """
        Args:
            data_dir: Directory containing imagesTr/ and labelsTr/.
            output_dir: Root output directory for checkpoints and predictions.
            device: torch device.
            n_folds: Number of cross-validation folds.

        All architecture and preprocessing parameters are hardcoded in
        config/plan_config.py — no external JSON files needed.
        """
        from config.plan_config import get_default_configs

        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.device = device
        self.n_folds = n_folds

        # Load configurations (all hardcoded)
        self.dataset_config, self.lowres_config, self.fullres_config = \
            get_default_configs()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_kfold(self):
        """
        Run k-fold cross-validation for both cascade stages.

        For each fold:
          1. Split data into train/val
          2. Train coarse model (3d_lowres)
          3. Run coarse inference on training set to generate bboxes
          4. Train fine model (3d_fullres) with CascadeFineDataset
        """
        data_pairs = find_data_pairs(self.data_dir)
        n = len(data_pairs)
        indices = np.arange(n)
        np.random.shuffle(indices)

        fold_size = n // self.n_folds

        for fold in range(self.n_folds):
            print(f"\n{'#'*60}")
            print(f"# Fold {fold + 1}/{self.n_folds}")
            print(f"{'#'*60}")

            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([
                indices[:fold * fold_size],
                indices[(fold + 1) * fold_size:],
            ])

            train_pairs = [data_pairs[i] for i in train_idx]
            val_pairs = [data_pairs[i] for i in val_idx]

            fold_dir = self.output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # --- Stage 1: Train coarse model ---
            self._train_coarse(train_pairs, val_pairs, fold_dir)

            # --- Stage 2: Generate coarse predictions for training set ---
            coarse_pred_dir = fold_dir / "coarse_predictions"
            self._generate_coarse_predictions(train_pairs, coarse_pred_dir, fold_dir)

            # --- Stage 3: Train fine model with bbox-restricted sampling ---
            self._train_fine(train_pairs, val_pairs, coarse_pred_dir, fold_dir)

    def _train_coarse(self, train_pairs, val_pairs, fold_dir):
        """Train the coarse (3d_lowres) model."""
        print("\n--- Stage 1: Training Coarse Model (3d_lowres) ---")

        train_ds = BraSegDataset(
            data_pairs=train_pairs,
            dataset_config=self.dataset_config,
            plan_config=self.lowres_config,
            is_training=True,
            augment=True,
        )
        val_ds = BraSegDataset(
            data_pairs=val_pairs,
            dataset_config=self.dataset_config,
            plan_config=self.lowres_config,
            is_training=False,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.lowres_config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=2,
        )

        trainer = VNetTrainer(
            plan_config=self.lowres_config,
            dataset_config=self.dataset_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            output_dir=str(fold_dir / "coarse"),
        )
        trainer.train()

    def _generate_coarse_predictions(self, data_pairs, pred_dir, fold_dir):
        """Run coarse model inference to generate bounding boxes."""
        print("\n--- Generating Coarse Predictions ---")
        pred_dir.mkdir(parents=True, exist_ok=True)

        predictor = VNetPredictor(
            plan_config=self.lowres_config,
            dataset_config=self.dataset_config,
            checkpoint_path=str(fold_dir / "coarse" / "best_model.pth"),
            device=self.device,
        )

        for img_path, _ in data_pairs:
            try:
                pred = predictor.predict_full_volume(img_path)
                case_name = os.path.basename(img_path).replace(".nii.gz", "").replace(".nii", "")
                save_image(
                    pred.astype(np.uint8),
                    np.array(self.lowres_config.spacing),
                    str(pred_dir / f"{case_name}.nii.gz"),
                )
            except Exception as e:
                print(f"  Warning: failed to predict {img_path}: {e}")

    def _train_fine(self, train_pairs, val_pairs, coarse_pred_dir, fold_dir):
        """Train the fine (3d_fullres) model with bbox-restricted sampling."""
        print("\n--- Stage 2: Training Fine Model (3d_fullres) ---")

        train_ds = CascadeFineDataset(
            data_pairs=train_pairs,
            dataset_config=self.dataset_config,
            plan_config=self.fullres_config,
            coarse_predictions_dir=str(coarse_pred_dir),
            is_training=True,
            bbox_expansion=20,
        )
        val_ds = BraSegDataset(
            data_pairs=val_pairs,
            dataset_config=self.dataset_config,
            plan_config=self.fullres_config,
            is_training=False,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.fullres_config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=2,
        )

        trainer = VNetTrainer(
            plan_config=self.fullres_config,
            dataset_config=self.dataset_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            output_dir=str(fold_dir / "fine"),
        )
        trainer.train()

    @torch.no_grad()
    def predict(self, image_path: str, fold_dir: str = None) -> np.ndarray:
        """
        Full cascade inference: coarse -> fine.

        1. Coarse model predicts ROI mask
        2. Bounding box extracted and expanded by 20 voxels
        3. Fine model processes the cropped region
        4. Result placed back into original image space

        Args:
            image_path: Path to the CTA image.
            fold_dir: Directory containing coarse/ and fine/ checkpoints.

        Returns:
            Binary segmentation mask at original image resolution.
        """
        if fold_dir is None:
            fold_dir = self.output_dir / "fold_0"

        # Stage 1: Coarse prediction
        coarse_predictor = VNetPredictor(
            plan_config=self.lowres_config,
            dataset_config=self.dataset_config,
            checkpoint_path=str(Path(fold_dir) / "coarse" / "best_model.pth"),
            device=self.device,
        )
        coarse_pred = coarse_predictor.predict_full_volume(image_path)

        # Extract bounding box from coarse prediction
        mask = coarse_pred[0] > 0
        if mask.sum() == 0:
            # Fallback: process entire volume
            bbox = None
        else:
            d_idx, h_idx, w_idx = np.where(mask)
            exp = 20
            bbox = np.array([
                max(0, d_idx.min() - exp), d_idx.max() + exp,
                max(0, h_idx.min() - exp), h_idx.max() + exp,
                max(0, w_idx.min() - exp), w_idx.max() + exp,
            ])

        # Stage 2: Fine prediction within bbox
        fine_predictor = VNetPredictor(
            plan_config=self.fullres_config,
            dataset_config=self.dataset_config,
            checkpoint_path=str(Path(fold_dir) / "fine" / "best_model.pth"),
            device=self.device,
        )
        fine_pred = fine_predictor.predict_with_bbox(image_path, bbox)

        # Post-processing
        fine_pred = postprocess_segmentation(fine_pred)

        return fine_pred

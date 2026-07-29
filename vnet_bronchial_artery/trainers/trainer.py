"""
V-Net training loop.

Implements the training procedure described in the manuscript:
  - AdamW optimizer (lr=1e-4, weight_decay=1e-5)
  - Cosine annealing learning rate schedule
  - Batch size 2 (from configuration)
  - Max 100 epochs with early stopping (patience=15)
  - 5-fold cross-validation
  - Combined Tversky + ClDice loss
  - Deep supervision during training
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict
from pathlib import Path

from config.plan_config import PlanConfig, DatasetConfig
from models.vnet import build_vnet
from losses.combined import CombinedLoss, get_default_loss
from utils.metrics import compute_dice, compute_hd95


class VNetTrainer:
    """
    Trainer for a single V-Net model (one cascade stage).

    Handles the full training loop with:
      - AdamW + cosine annealing
      - Early stopping (patience=15)
      - Deep supervision loss weighting
      - Validation metrics (DSC, 95% HD)
      - Best model checkpointing
    """

    def __init__(
        self,
        plan_config: PlanConfig,
        dataset_config: DatasetConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        output_dir: str = "./checkpoints",
        num_classes: int = 2,
        max_epochs: int = 100,
        patience: int = 15,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.plan_config = plan_config
        self.dataset_config = dataset_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Build model
        self.model = build_vnet(
            plan_config=plan_config,
            num_classes=num_classes,
            deep_supervision=True,
        ).to(self.device)

        # Loss function
        self.loss_fn = loss_fn if loss_fn is not None else get_default_loss()

        # Optimizer: AdamW (from manuscript)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # LR scheduler: cosine annealing (from manuscript)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        # Training state
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_dice": [],
            "val_hd95": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            data = batch["data"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward (with deep supervision during training)
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate and return (mean_dice, mean_hd95)."""
        self.model.eval()
        dice_scores = []
        hd95_scores = []

        for batch in self.val_loader:
            data = batch["data"].to(self.device)
            target = batch["target"].to(self.device)

            outputs = self.model(data)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Convert to binary prediction
            if outputs.shape[1] == 2:
                pred = torch.argmax(outputs, dim=1, keepdim=True).float()
            else:
                pred = (torch.sigmoid(outputs) > 0.5).float()

            # Compute metrics per sample
            for i in range(data.shape[0]):
                p = pred[i, 0].cpu().numpy()
                t = target[i, 0].cpu().numpy()
                dice = compute_dice(p, t)
                hd95 = compute_hd95(p, t)
                dice_scores.append(dice)
                if not np.isnan(hd95):
                    hd95_scores.append(hd95)

        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        mean_hd95 = np.mean(hd95_scores) if hd95_scores else float("inf")
        return mean_dice, mean_hd95

    def train(self) -> dict:
        """
        Full training loop with early stopping.

        Returns:
            Training history dictionary.
        """
        print(f"{'='*60}")
        print(f"Training V-Net ({self.plan_config.config_name})")
        print(f"  Patch size: {self.plan_config.patch_size}")
        print(f"  Batch size: {self.plan_config.batch_size}")
        print(f"  Spacing: {self.plan_config.spacing}")
        print(f"  Parameters: {self.model.get_parameter_count():,}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_dice, val_hd95 = self.validate()

            # Update LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_hd95"].append(val_hd95)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Dice: {val_dice:.4f} | "
                f"HD95: {val_hd95:.2f}mm | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpoint
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pth", epoch, val_dice)
                print(f"  -> New best Dice: {val_dice:.4f}, checkpoint saved.")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs).")
                break

        # Save final model
        self._save_checkpoint("final_model.pth", epoch, val_dice)

        print(f"\nTraining complete. Best Dice: {self.best_dice:.4f}")
        return self.history

    def _save_checkpoint(self, filename: str, epoch: int, val_dice: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_dice": val_dice,
            "plan_config": {
                "config_name": self.plan_config.config_name,
                "patch_size": self.plan_config.patch_size,
                "batch_size": self.plan_config.batch_size,
                "spacing": self.plan_config.spacing,
            },
        }
        path = self.output_dir / filename
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path} "
              f"(epoch {checkpoint['epoch']}, Dice {checkpoint['val_dice']:.4f})")

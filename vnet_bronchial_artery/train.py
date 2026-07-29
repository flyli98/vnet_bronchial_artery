"""
Main training entry point for V-Net bronchial artery segmentation.

All network architecture and preprocessing parameters are hardcoded in
config/plan_config.py. No external configuration files are needed.

Usage:
    # Train a single stage (3d_fullres)
    python train.py --data_dir /path/to/data --stage fullres

    # Train cascade (coarse + fine)
    python train.py --data_dir /path/to/data --stage cascade

    # Train with 5-fold cross-validation
    python train.py --data_dir /path/to/data --stage cascade --kfold 5
"""

import argparse
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Train V-Net for bronchial artery segmentation"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing imagesTr/ and labelsTr/",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--stage", type=str, default="fullres",
        choices=["lowres", "fullres", "cascade"],
        help="Training stage: lowres (coarse), fullres (fine), or cascade (both)",
    )
    parser.add_argument(
        "--kfold", type=int, default=0,
        help="Number of cross-validation folds (0 = no CV, single split)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5,
        help="Weight decay coefficient",
    )

    args = parser.parse_args()

    # All configs are hardcoded — no external files needed
    from config.plan_config import get_default_configs
    ds_config, lowres_cfg, fullres_cfg = get_default_configs()

    if args.stage == "cascade":
        from trainers.cascade import CascadePipeline

        if args.kfold > 0:
            pipeline = CascadePipeline(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                device=args.device,
                n_folds=args.kfold,
            )
            pipeline.run_kfold()
        else:
            _train_cascade_single(
                args, ds_config, lowres_cfg, fullres_cfg
            )
    else:
        plan_cfg = lowres_cfg if args.stage == "lowres" else fullres_cfg

        data_pairs = _find_data_pairs(args.data_dir)
        n = len(data_pairs)
        indices = np.arange(n)
        np.random.shuffle(indices)

        if args.kfold > 0:
            fold_size = n // args.kfold
            for fold in range(args.kfold):
                val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
                train_idx = np.concatenate([
                    indices[:fold * fold_size],
                    indices[(fold + 1) * fold_size:],
                ])
                _train_single(
                    [data_pairs[i] for i in train_idx],
                    [data_pairs[i] for i in val_idx],
                    ds_config, plan_cfg, args, fold,
                )
        else:
            split = int(0.8 * n)
            _train_single(
                [data_pairs[i] for i in indices[:split]],
                [data_pairs[i] for i in indices[split:]],
                ds_config, plan_cfg, args, 0,
            )


def _find_data_pairs(data_dir):
    from data.dataset import find_data_pairs
    return find_data_pairs(data_dir)


def _train_cascade_single(args, ds_config, lowres_cfg, fullres_cfg):
    """Train cascade with a single 80/20 split."""
    from data.dataset import find_data_pairs, BraSegDataset, CascadeFineDataset
    from trainers.trainer import VNetTrainer
    from inference.predictor import VNetPredictor
    from data.preprocessing import save_image
    from torch.utils.data import DataLoader

    data_pairs = find_data_pairs(args.data_dir)
    n = len(data_pairs)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)

    train_pairs = [data_pairs[i] for i in indices[:split]]
    val_pairs = [data_pairs[i] for i in indices[split:]]

    fold_dir = Path(args.output_dir) / "single_run"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Coarse
    print("\n=== Stage 1: Coarse (3d_lowres) ===")
    train_ds = BraSegDataset(train_pairs, ds_config, lowres_cfg,
                             is_training=True, augment=True)
    val_ds = BraSegDataset(val_pairs, ds_config, lowres_cfg,
                           is_training=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=lowres_cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    trainer = VNetTrainer(
        plan_config=lowres_cfg,
        dataset_config=ds_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=str(fold_dir / "coarse"),
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer.train()

    # Generate coarse predictions
    print("\n=== Generating Coarse Predictions ===")
    coarse_pred_dir = fold_dir / "coarse_predictions"
    coarse_pred_dir.mkdir(exist_ok=True)
    predictor = VNetPredictor(
        plan_config=lowres_cfg,
        dataset_config=ds_config,
        checkpoint_path=str(fold_dir / "coarse" / "best_model.pth"),
        device=args.device,
    )
    for img_path, _ in train_pairs:
        try:
            pred = predictor.predict_full_volume(img_path)
            case_name = os.path.basename(img_path).replace(".nii.gz", "").replace(".nii", "")
            save_image(pred.astype(np.uint8),
                       np.array(lowres_cfg.spacing),
                       str(coarse_pred_dir / f"{case_name}.nii.gz"))
        except Exception as e:
            print(f"  Warning: {e}")

    # Stage 2: Fine
    print("\n=== Stage 2: Fine (3d_fullres) ===")
    train_ds = CascadeFineDataset(
        train_pairs, ds_config, fullres_cfg,
        coarse_predictions_dir=str(coarse_pred_dir),
        is_training=True, bbox_expansion=20,
    )
    val_ds = BraSegDataset(val_pairs, ds_config, fullres_cfg,
                           is_training=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=fullres_cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    trainer = VNetTrainer(
        plan_config=fullres_cfg,
        dataset_config=ds_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=str(fold_dir / "fine"),
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer.train()


def _train_single(train_pairs, val_pairs, ds_config, plan_cfg, args, fold):
    from data.dataset import BraSegDataset
    from trainers.trainer import VNetTrainer
    from torch.utils.data import DataLoader

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BraSegDataset(train_pairs, ds_config, plan_cfg,
                             is_training=True, augment=True)
    val_ds = BraSegDataset(val_pairs, ds_config, plan_cfg,
                           is_training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=plan_cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    trainer = VNetTrainer(
        plan_config=plan_cfg,
        dataset_config=ds_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=str(fold_dir / args.stage),
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer.train()


if __name__ == "__main__":
    main()

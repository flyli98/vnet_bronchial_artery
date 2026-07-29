"""
Main inference entry point for V-Net bronchial artery segmentation.

All network architecture and preprocessing parameters are hardcoded in
config/plan_config.py. No external JSON files are needed.

Usage:
    # Single-stage prediction
    python predict.py --image /path/to/image.nii.gz \
        --checkpoint /path/to/best_model.pth --stage fullres --output prediction.nii.gz

    # Cascade prediction (coarse -> fine)
    python predict.py --image /path/to/image.nii.gz \
        --cascade --fold_dir /path/to/fold_0 --output prediction.nii.gz
"""

import argparse
import sys
import os
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Predict bronchial artery segmentation with V-Net"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the CTA image (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--output", type=str, default="prediction.nii.gz",
        help="Output file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.25,
        help="Sliding window overlap fraction (0-0.5)",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable test-time augmentation",
    )

    # Single-stage options
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (single-stage mode)",
    )
    parser.add_argument(
        "--stage", type=str, default="fullres",
        choices=["lowres", "fullres"],
        help="Which plan configuration to use",
    )

    # Cascade options
    parser.add_argument(
        "--cascade", action="store_true",
        help="Use cascade prediction (coarse -> fine)",
    )
    parser.add_argument(
        "--fold_dir", type=str, default=None,
        help="Directory containing coarse/ and fine/ checkpoints (cascade mode)",
    )

    args = parser.parse_args()

    from config.plan_config import get_default_configs, LOWRES_CONFIG, FULLRES_CONFIG
    from inference.predictor import VNetPredictor
    from inference.postprocess import postprocess_segmentation
    from data.preprocessing import save_image

    ds_config, lowres_cfg, fullres_cfg = get_default_configs()

    if args.cascade:
        # Cascade prediction
        from trainers.cascade import CascadePipeline

        pipeline = CascadePipeline(
            data_dir=".",  # not used for prediction
            output_dir=args.fold_dir or "./output",
            device=args.device,
        )
        pred = pipeline.predict(args.image, fold_dir=args.fold_dir)
        plan_cfg = fullres_cfg
    else:
        # Single-stage prediction
        plan_cfg = lowres_cfg if args.stage == "lowres" else fullres_cfg
        predictor = VNetPredictor(
            plan_config=plan_cfg,
            dataset_config=ds_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            overlap=args.overlap,
            use_tta=args.tta,
        )
        pred = predictor.predict_full_volume(args.image)

    # Post-processing
    pred = postprocess_segmentation(pred, min_volume=10, closing_radius=1)

    # Save
    save_image(
        pred.astype(np.uint8),
        np.array(plan_cfg.spacing),
        args.output,
    )
    print(f"Prediction saved to {args.output}")
    print(f"  Voxel count: {pred.sum()}")
    print(f"  Shape: {pred.shape}")


if __name__ == "__main__":
    main()

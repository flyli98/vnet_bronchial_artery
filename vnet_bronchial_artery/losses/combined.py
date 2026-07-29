"""
Combined loss function: Tversky Loss + ClDice Loss.

As described in the manuscript:

    Loss = Tversky Loss + ClDice Loss

with equal weighting (1:1). The Tversky loss (alpha=0.3, beta=0.7)
penalizes false negatives more heavily, prioritizing detection of
small bronchial arteries. The ClDice loss enforces topological
consistency based on vessel centerline structure, reducing
discontinuities in segmentation outputs.

This module also supports deep supervision: when the model produces
auxiliary outputs, the combined loss is applied to each with
exponentially decaying weights.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union

from losses.tversky import TverskyLoss
from losses.cldice import ClDiceLoss


class CombinedLoss(nn.Module):
    """
    Combined Tversky + ClDice loss.

    Args:
        tversky_alpha: FP weight (default: 0.3).
        tversky_beta: FN weight (default: 0.7).
        tversky_weight: Weight for Tversky loss (default: 1.0).
        cldice_weight: Weight for ClDice loss (default: 1.0).
        cldice_iter: Skeletonization iterations (default: 3).
        deep_supervision_weights: Weights for deep supervision outputs.
            If None, uses exponential decay [0.5^0, 0.5^1, ...].
    """

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_weight: float = 1.0,
        cldice_weight: float = 1.0,
        cldice_iter: int = 3,
        deep_supervision_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.cldice_loss = ClDiceLoss(iter_=cldice_iter)
        self.tversky_weight = tversky_weight
        self.cldice_weight = cldice_weight
        self.deep_supervision_weights = deep_supervision_weights

    def forward(
        self,
        outputs: Union[torch.Tensor, tuple],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss, optionally with deep supervision.

        Args:
            outputs: Model output. If a tuple, the first element is the
                main output and subsequent elements are deep supervision
                outputs at progressively lower resolutions.
            targets: Ground truth labels.

        Returns:
            Scalar total loss.
        """
        if isinstance(outputs, (tuple, list)):
            # Deep supervision
            main_output = outputs[0]
            ds_outputs = outputs[1:]

            # Main loss
            loss = self._compute_single(main_output, targets)

            # Deep supervision losses with decaying weights
            if len(ds_outputs) > 0:
                if self.deep_supervision_weights is None:
                    weights = [0.5 ** (i + 1) for i in range(len(ds_outputs))]
                else:
                    weights = self.deep_supervision_weights

                for ds_out, w in zip(ds_outputs, weights):
                    # Downsample target to match ds output size if needed
                    ds_target = targets
                    if ds_out.shape[2:] != targets.shape[2:]:
                        ds_target = torch.nn.functional.interpolate(
                            targets.float(),
                            size=ds_out.shape[2:],
                            mode="nearest",
                        )
                    loss += w * self._compute_single(ds_out, ds_target)

            return loss
        else:
            return self._compute_single(outputs, targets)

    def _compute_single(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Tversky + ClDice for a single output."""
        tv = self.tversky_loss(logits, targets)
        cd = self.cldice_loss(logits, targets)
        return self.tversky_weight * tv + self.cldice_weight * cd


def get_default_loss() -> CombinedLoss:
    """
    Build the default combined loss as specified in the manuscript.

    Tversky(alpha=0.3, beta=0.7) + ClDice(iter=3), weight 1:1.
    """
    return CombinedLoss(
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_weight=1.0,
        cldice_weight=1.0,
        cldice_iter=3,
    )

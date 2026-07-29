"""
Attention gates for V-Net skip connections.

As described in the manuscript, an attention mechanism is integrated into
the skip connection pathways. The attention gate adaptively amplifies
feature weights corresponding to vascular regions while suppressing
background tissue signals, improving sensitivity to slender bronchial
arteries.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for
the Pancreas" (MIDL 2018).
"""

import torch
import torch.nn as nn


class AttentionGate3D(nn.Module):
    """
    3D attention gate for skip connections.

    Computes attention coefficients from both the decoder feature (g)
    and the encoder skip feature (x):

        q = W_g * g + W_x * x + b
        alpha = sigmoid(W_psi * ReLU(q) + b_psi)
        output = alpha * x

    The gating signal g comes from the decoder (lower resolution, more
    semantic), and x is the encoder skip (higher resolution, more detail).
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int,
        norm_op: str = "InstanceNorm3d",
        norm_op_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}

        Norm = nn.InstanceNorm3d if norm_op == "InstanceNorm3d" else nn.BatchNorm3d

        # Gate (decoder) transform: 1x1x1 conv
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, **norm_op_kwargs),
        )

        # Skip (encoder) transform: 1x1x1 conv
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, **norm_op_kwargs),
        )

        # Attention coefficient: 1x1x1 conv -> sigmoid
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: decoder feature [B, C_g, D', H', W'] (lower resolution)
            skip: encoder feature [B, C_x, D, H, W] (higher resolution)

        Returns:
            Attention-weighted skip feature [B, C_x, D, H, W]
        """
        # Transform gate to inter_channels
        g = self.W_gate(gate)

        # Transform skip to inter_channels
        x = self.W_skip(skip)

        # Upsample gate to match skip spatial size if needed
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(
                g, size=x.shape[2:], mode="trilinear", align_corners=False
            )

        # Combine and compute attention
        combined = self.relu(g + x)
        alpha = self.psi(combined)

        # Apply attention to original skip (preserve original channels)
        return skip * alpha


class AttentionGate2D(nn.Module):
    """2D variant of the attention gate (for 2d configuration)."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int,
        norm_op: str = "InstanceNorm2d",
        norm_op_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}

        Norm = nn.InstanceNorm2d if norm_op == "InstanceNorm2d" else nn.BatchNorm2d

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, **norm_op_kwargs),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            Norm(inter_channels, **norm_op_kwargs),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        x = self.W_skip(skip)

        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(
                g, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        combined = self.relu(g + x)
        alpha = self.psi(combined)
        return skip * alpha


def build_attention_gate(
    conv_op: str = "Conv3d",
    gate_channels: int = 320,
    skip_channels: int = 64,
    inter_channels: int | None = None,
    norm_op: str = "InstanceNorm3d",
    norm_op_kwargs: dict | None = None,
) -> nn.Module:
    """
    Factory function to build the appropriate attention gate.

    inter_channels defaults to skip_channels // 2 if not specified.
    """
    if inter_channels is None:
        inter_channels = max(skip_channels // 2, 1)

    if conv_op == "Conv3d":
        return AttentionGate3D(
            gate_channels=gate_channels,
            skip_channels=skip_channels,
            inter_channels=inter_channels,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
        )
    else:
        return AttentionGate2D(
            gate_channels=gate_channels,
            skip_channels=skip_channels,
            inter_channels=inter_channels,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
        )

"""
V-Net with attention-gated skip connections.

This implementation follows the 3D V-Net architecture described in the
manuscript, with structural parameters (n_stages, features_per_stage,
kernel_sizes, strides, n_conv_per_stage, etc.) defined in the
configuration.

Key architectural features:
  1. Residual blocks in every encoder/decoder stage (V-Net characteristic)
  2. Strided-conv downsampling (not max pooling)
  3. Transposed-conv upsampling
  4. Attention gates on skip connections (from manuscript)
  5. InstanceNorm3d + LeakyReLU
  6. Deep supervision output heads (optional)

The network is fully parameterized by PlanConfig, allowing the same code
to serve both the coarse (3d_lowres) and fine (3d_fullres) cascade stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from config.plan_config import PlanConfig
from models.blocks import (
    ConvBlock,
    ResidualStage,
    DownStage,
    UpStage,
    _match_size,
)
from models.attention import build_attention_gate


class VNet(nn.Module):
    """
    3D V-Net with attention-gated skip connections.

    Architecture (encoder-decoder with residual blocks):

        Input
          |
        [Encoder Stage 0]  ----skip 0 (attention)----+
          | (stride 1, no downsample)                |
        [Encoder Stage 1]  ----skip 1 (attention)----+
          | (downsample)                             |
        [Encoder Stage 2]  ----skip 2 (attention)----+
          | (downsample)                             |
          ...                                        |
        [Bottleneck]                                 |
          |                                          |
        [Decoder Stage N-1] <---cat(skip N-1)-------+
          | (upsample)
          ...
        [Decoder Stage 1]   <---cat(skip 1)---------+
          | (upsample)
        [Decoder Stage 0]   <---cat(skip 0)--------+
          |
        [Output Conv 1x1] -> logits

    Parameters are driven by PlanConfig.
    """

    def __init__(self, plan_config: PlanConfig, num_classes: int = 2,
                 deep_supervision: bool = True):
        """
        Args:
            plan_config: PlanConfig (e.g. 3d_fullres).
            num_classes: Number of output classes (background + foreground).
            deep_supervision: If True, produce auxiliary outputs at each
                decoder stage for training-time deep supervision.
        """
        super().__init__()
        self.plan_config = plan_config
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        arch = plan_config.architecture
        conv_op = arch.conv_op
        conv_bias = arch.conv_bias
        norm_op = arch.norm_op
        norm_op_kwargs = arch.norm_op_kwargs
        nonlin = arch.nonlin
        nonlin_kwargs = arch.nonlin_kwargs

        n_stages = arch.n_stages
        features = arch.features_per_stage
        kernel_sizes = arch.kernel_sizes
        strides = arch.strides
        n_convs_enc = arch.n_conv_per_stage
        n_convs_dec = arch.n_conv_per_stage_decoder

        in_channels = 1  # single-channel CTA

        # ---- Encoder ----
        self.encoder_stages = nn.ModuleList()
        self.encoder_skips = nn.ModuleList()  # attention gates

        prev_ch = in_channels
        for i in range(n_stages):
            ks = kernel_sizes[i][0]  # assume isotropic kernel
            st = strides[i][0]

            if i == 0:
                # First stage: no downsampling (stride=1), residual block only
                stage = ResidualStage(
                    in_channels=prev_ch,
                    out_channels=features[i],
                    kernel_size=ks,
                    n_convs=n_convs_enc[i],
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            else:
                # DownStage: strided conv + residual block
                stage = DownStage(
                    in_channels=prev_ch,
                    out_channels=features[i],
                    kernel_size=ks,
                    stride=st,
                    n_convs=n_convs_enc[i],
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )

            self.encoder_stages.append(stage)
            prev_ch = features[i]

        # ---- Attention gates (for skip connections, stages 0..n-2) ----
        bottleneck_ch = features[-1]
        for i in range(n_stages - 1):
            gate = build_attention_gate(
                conv_op=conv_op,
                gate_channels=features[i + 1] if i + 1 < n_stages else bottleneck_ch,
                skip_channels=features[i],
                inter_channels=max(features[i] // 2, 1),
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
            )
            self.encoder_skips.append(gate)

        # ---- Decoder ----
        self.decoder_stages = nn.ModuleList()
        n_dec_stages = len(n_convs_dec)  # = n_stages - 1

        for i in range(n_dec_stages):
            # Decoder stage i upsamples from level (n_stages-1-i) to (n_stages-2-i)
            enc_idx = n_stages - 1 - i  # current decoder input level
            skip_idx = enc_idx - 1       # corresponding encoder skip level

            ks = kernel_sizes[enc_idx][0]
            st = strides[enc_idx][0]

            up = UpStage(
                in_channels=features[enc_idx],
                skip_channels=features[skip_idx],
                out_channels=features[skip_idx],
                kernel_size=ks,
                stride=st,
                n_convs=n_convs_dec[i],
                conv_op=conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            self.decoder_stages.append(up)

        # ---- Output heads ----
        Conv = nn.Conv3d if conv_op == "Conv3d" else nn.Conv2d

        # Final output: 1x1 conv to num_classes
        self.final_conv = Conv(features[0], num_classes, kernel_size=1, bias=True)

        # Deep supervision heads (one per decoder stage except the last)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for i in range(n_dec_stages - 1):
                enc_idx = n_stages - 1 - i
                skip_idx = enc_idx - 1
                self.ds_heads.append(
                    Conv(features[skip_idx], num_classes, kernel_size=1, bias=True)
                )
        else:
            self.ds_heads = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming / Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1, D, H, W] (or [B, 1, H, W] for 2d).

        Returns:
            If deep_supervision and training:
                Tuple of (final_output, ds_output_0, ds_output_1, ...)
            Otherwise:
                final_output [B, num_classes, D, H, W]
        """
        # ---- Encoder: collect skip features ----
        skips = []
        feat = x
        for i, stage in enumerate(self.encoder_stages):
            feat = stage(feat)
            if i < len(self.encoder_stages) - 1:
                skips.append(feat)

        # bottleneck = feat (last encoder output)

        # ---- Decoder with attention-gated skip connections ----
        ds_outputs = []
        for i, up_stage in enumerate(self.decoder_stages):
            skip_idx = len(skips) - 1 - i

            # Get the gating signal (current decoder feature) for attention
            # The gate is the feature before upsampling
            gate_feat = feat

            # Apply attention to the skip connection
            skip = skips[skip_idx]
            attended_skip = self.encoder_skips[skip_idx](gate_feat, skip)

            # Upsample and concatenate
            feat = up_stage(feat, attended_skip)

            # Deep supervision output
            if self.deep_supervision and self.training and i < len(self.decoder_stages) - 1:
                ds_outputs.append(self.ds_heads[i](feat))

        # ---- Final output ----
        output = self.final_conv(feat)

        if self.deep_supervision and self.training and len(ds_outputs) > 0:
            return (output, *ds_outputs)
        return output

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vnet(
    plan_config: PlanConfig,
    num_classes: int = 2,
    deep_supervision: bool = True,
) -> VNet:
    """
    Factory: build a VNet from a PlanConfig.

    This is the primary entry point used by the training and inference
    pipelines. The same function serves both cascade stages (coarse
    and fine) by passing different PlanConfig objects.
    """
    return VNet(
        plan_config=plan_config,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
    )


# ---------------------------------------------------------------------------
# Quick smoke test (parameters are hardcoded, no external files needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from config.plan_config import get_default_configs

    ds_config, lowres_cfg, fullres_cfg = get_default_configs()

    print("=== 3D Fullres V-Net ===")
    model = build_vnet(fullres_cfg, num_classes=2, deep_supervision=True)
    print(f"Parameters: {model.get_parameter_count():,}")

    # Test forward pass
    patch = tuple(fullres_cfg.patch_size)
    x = torch.randn(1, 1, *patch)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    print("\n=== 3D Lowres V-Net ===")
    model_lr = build_vnet(lowres_cfg, num_classes=2, deep_supervision=True)
    print(f"Parameters: {model_lr.get_parameter_count():,}")
    patch_lr = tuple(lowres_cfg.patch_size)
    x_lr = torch.randn(1, 1, *patch_lr)
    model_lr.eval()
    with torch.no_grad():
        out_lr = model_lr(x_lr)
    print(f"Input:  {x_lr.shape}")
    print(f"Output: {out_lr.shape}")

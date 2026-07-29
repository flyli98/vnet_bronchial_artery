"""
Residual convolution blocks for V-Net.

V-Net differs from plain U-Net by using residual connections within each
encoder/decoder stage. Each stage applies n convolutions, and the input
is added to the output (with a projection if channel dimensions differ).

The block parameters (kernel size, number of convolutions, normalization,
activation) are driven by the architecture configuration.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single Conv -> Norm -> ReLU unit.

    Supports both Conv3d (default) and Conv2d, selected by conv_op string.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        conv_op: str = "Conv3d",
        conv_bias: bool = True,
        norm_op: str = "InstanceNorm3d",
        norm_op_kwargs: dict | None = None,
        nonlin: str = "LeakyReLU",
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        Conv = nn.Conv3d if conv_op == "Conv3d" else nn.Conv2d
        Norm = self._get_norm(norm_op, conv_op)
        Act = self._get_act(nonlin)

        padding = kernel_size // 2
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        self.norm = Norm(out_channels, **norm_op_kwargs)
        self.act = Act(**nonlin_kwargs)

    @staticmethod
    def _get_norm(name: str, conv_op: str):
        if name == "InstanceNorm3d":
            return nn.InstanceNorm3d
        elif name == "InstanceNorm2d":
            return nn.InstanceNorm2d
        elif name == "BatchNorm3d":
            return nn.BatchNorm3d
        elif name == "BatchNorm2d":
            return nn.BatchNorm2d
        elif name in ("None", None):
            return nn.Identity
        raise ValueError(f"Unsupported norm_op: {name}")

    @staticmethod
    def _get_act(name: str):
        if name == "LeakyReLU":
            return nn.LeakyReLU
        elif name == "ReLU":
            return nn.ReLU
        elif name == "PReLU":
            return nn.PReLU
        raise ValueError(f"Unsupported nonlin: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualStage(nn.Module):
    """
    V-Net residual stage: applies n convolutions with a residual connection.

    The input is projected to ``out_channels`` if the channel count differs,
    then each convolution transforms the features. The projected input is
    added to the final output, forming the residual path:

        output = f_n(...f_2(f_1(x))...) + project(x)

    This is the core building block of the V-Net encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_convs: int = 2,
        conv_op: str = "Conv3d",
        conv_bias: bool = True,
        norm_op: str = "InstanceNorm3d",
        norm_op_kwargs: dict | None = None,
        nonlin: str = "LeakyReLU",
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        Conv = nn.Conv3d if conv_op == "Conv3d" else nn.Conv2d

        # Residual projection (1x1 conv) if channels change
        if in_channels != out_channels:
            self.residual = Conv(
                in_channels, out_channels, kernel_size=1, stride=1, bias=conv_bias
            )
        else:
            self.residual = nn.Identity()

        # Stack of conv blocks
        blocks = []
        prev_ch = in_channels
        for _ in range(n_convs):
            blocks.append(
                ConvBlock(
                    in_channels=prev_ch,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    conv_op=conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            prev_ch = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.blocks(x)
        return out + residual


class DownStage(nn.Module):
    """
    Encoder stage: strided downsampling convolution + residual conv block.

    The downsampling is performed by a strided convolution (as in V-Net),
    not by max pooling. The stride comes from the configuration.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        n_convs: int = 2,
        conv_op: str = "Conv3d",
        conv_bias: bool = True,
        norm_op: str = "InstanceNorm3d",
        norm_op_kwargs: dict | None = None,
        nonlin: str = "LeakyReLU",
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        Conv = nn.Conv3d if conv_op == "Conv3d" else nn.Conv2d
        Norm = ConvBlock._get_norm(norm_op, conv_op)
        Act = ConvBlock._get_act(nonlin)

        padding = kernel_size // 2
        self.down_conv = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        self.down_norm = Norm(out_channels, **norm_op_kwargs)
        self.down_act = Act(**nonlin_kwargs)

        self.res_block = ResidualStage(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_convs=n_convs,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_act(self.down_norm(self.down_conv(x)))
        x = self.res_block(x)
        return x


class UpStage(nn.Module):
    """
    Decoder stage: transposed convolution upsampling + concatenation
    with skip connection + residual conv block.

    The upsampling factor is driven by the stride from the configuration.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        n_convs: int = 2,
        conv_op: str = "Conv3d",
        conv_bias: bool = True,
        norm_op: str = "InstanceNorm3d",
        norm_op_kwargs: dict | None = None,
        nonlin: str = "LeakyReLU",
        nonlin_kwargs: dict | None = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        ConvTranspose = nn.ConvTranspose3d if conv_op == "Conv3d" else nn.ConvTranspose2d
        Norm = ConvBlock._get_norm(norm_op, conv_op)
        Act = ConvBlock._get_act(nonlin)

        padding = kernel_size // 2
        output_padding = stride - 1

        self.up_conv = ConvTranspose(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
            output_padding=output_padding,
            bias=conv_bias,
        )
        self.up_norm = Norm(out_channels, **norm_op_kwargs)
        self.up_act = Act(**nonlin_kwargs)

        # After concatenation: out_channels (up) + skip_channels
        self.res_block = ResidualStage(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_convs=n_convs,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_act(self.up_norm(self.up_conv(x)))

        # Handle size mismatch from odd-sized inputs
        if x.shape[2:] != skip.shape[2:]:
            x = _match_size(x, skip.shape[2:])

        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


def _match_size(x: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Crop or pad x to match target spatial dimensions."""
    for dim in range(2, x.ndim):
        diff = x.shape[dim] - target_size[dim - 2]
        if diff > 0:
            # Crop
            x = x.narrow(dim, diff // 2, target_size[dim - 2])
        elif diff < 0:
            # Pad
            pad = [0] * (2 * (x.ndim - 2))
            pad_idx = 2 * (x.ndim - 2) - 2 * (dim - 2) - 1
            pad[pad_idx] = -diff
            x = torch.nn.functional.pad(x, pad)
    return x

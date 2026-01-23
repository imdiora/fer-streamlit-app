from __future__ import annotations

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


def l2_norm(x: torch.Tensor, axis: int = 1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=axis, keepdim=True).clamp(min=eps)
    return x / norm


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.conv = ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = ConvBlock(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return shortcut + x if self.residual else x


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention (lightweight).

    This follows the common CoordAtt pattern used in mobile networks.
    """

    def __init__(self, inp: int, oup: int, groups: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)

        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)
        return identity * a_w * a_h


class MDConv(nn.Module):
    """Mixed Depthwise Convolution.

    Splits channels and applies depthwise conv with different kernel sizes.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int],
        split_out_channels: list[int],
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        if sum(split_out_channels) != channels:
            raise ValueError(
                f"MDConv split_out_channels must sum to channels ({channels}); got {sum(split_out_channels)}"
            )
        if len(kernel_sizes) != len(split_out_channels):
            raise ValueError("MDConv kernel_sizes and split_out_channels must have same length")

        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    c,
                    c,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2,
                    groups=c,
                    bias=False,
                )
                for c, k in zip(self.split_channels, kernel_sizes)
            ]
        )
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.split_channels, dim=1)
        out = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        out = torch.cat(out, dim=1)
        return self.prelu(self.bn(out))


class MixDepthWise(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
        groups: int = 1,
        kernel_sizes: list[int] = [3, 5, 7],
        split_out_channels: list[int] = [64, 32, 32],
    ) -> None:
        super().__init__()
        self.residual = residual

        # 1x1 expand
        self.conv = ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # mixed depthwise
        self.conv_dw = MDConv(channels=groups, kernel_sizes=kernel_sizes, split_out_channels=split_out_channels, stride=stride)
        # coordinate attention
        self.ca = CoordAtt(groups, groups)
        # projection
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.ca(x)
        x = self.project(x)
        return shortcut + x if self.residual else x


class Residual(nn.Module):
    def __init__(
        self,
        c: int,
        num_block: int,
        groups: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        blocks = [DepthWise(c, c, residual=True, kernel=kernel, stride=stride, padding=padding, groups=groups) for _ in range(num_block)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixResidual(nn.Module):
    def __init__(
        self,
        c: int,
        num_block: int,
        groups: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
        kernel_sizes: list[int] = [3, 5],
        split_out_channels: list[int] = [64, 64],
    ) -> None:
        super().__init__()
        blocks = [
            MixDepthWise(
                c,
                c,
                residual=True,
                kernel=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                kernel_sizes=kernel_sizes,
                split_out_channels=split_out_channels,
            )
            for _ in range(num_block)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixedFeatureNet(nn.Module):
    """Backbone network used by DDAMFN.

    Designed for 112x112 input.

    The full forward returns a normalized embedding, but in this repository we mostly
    reuse intermediate features (512 x 7 x 7) for emotion classification.
    """

    def __init__(self, embedding_size: int = 256, out_h: int = 7, out_w: int = 7) -> None:
        super().__init__()

        # 112x112 -> 56x56
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        # 56x56
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        # 56x56 -> 28x28
        self.conv_23 = MixDepthWise(
            64,
            64,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=128,
            kernel_sizes=[3, 5, 7],
            split_out_channels=[64, 32, 32],
        )

        # 28x28
        self.conv_3 = MixResidual(
            64,
            num_block=9,
            groups=128,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            kernel_sizes=[3, 5],
            split_out_channels=[96, 32],
        )

        # 28x28 -> 14x14
        self.conv_34 = MixDepthWise(
            64,
            128,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=256,
            kernel_sizes=[3, 5, 7],
            split_out_channels=[128, 64, 64],
        )

        # 14x14
        self.conv_4 = MixResidual(
            128,
            num_block=16,
            groups=256,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            kernel_sizes=[3, 5],
            split_out_channels=[192, 64],
        )

        # 14x14 -> 7x7
        # Note: groups and split channels follow the pattern used in many DDAMFN implementations:
        # allocate one chunk per kernel size.
        self.conv_45 = MixDepthWise(
            128,
            256,
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=256 * 4,
            kernel_sizes=[3, 5, 7, 9],
            split_out_channels=[256, 256, 256, 256],
        )

        # 7x7
        self.conv_5 = MixResidual(
            256,
            num_block=6,
            groups=512,
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            kernel_sizes=[3, 5, 7],
            split_out_channels=[172, 170, 170],
        )

        # 7x7 -> 7x7
        self.conv_6_sep = ConvBlock(256, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        # 7x7 -> 1x1 (depthwise)
        self.conv_6_dw = LinearBlock(512, 512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0), groups=512)
        self.conv_6_flatten = Flatten()

        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)

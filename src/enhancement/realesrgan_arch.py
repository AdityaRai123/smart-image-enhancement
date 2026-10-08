"""
Real-ESRGAN generator architectures (PyTorch).

Re-implemented here (parameter names identical to the official code) so that the
official pretrained checkpoints from https://github.com/xinntao/Real-ESRGAN load with
``strict=True`` without installing ``basicsr`` / ``realesrgan`` (those packages pin old
torchvision APIs and frequently fail to install on current Python versions).

* :class:`SRVGGNetCompact` - lightweight VGG-style network used by
  ``realesr-general-x4v3`` (~1.2 M parameters). Default for CPU inference.
* :class:`RRDBNet`         - Residual-in-Residual Dense Block network used by
  ``RealESRGAN_x4plus`` (~16.7 M parameters). Higher quality, much slower on CPU.

Reference: X. Wang, L. Xie, C. Dong, Y. Shan, "Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data", ICCVW 2021.

This module imports torch at import time; it is only imported lazily by
``src/enhancement/super_resolution.py`` once torch is known to be available.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Compact SR network: conv + PReLU body, pixel-shuffle upsampling, nearest-neighbour skip."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_conv: int = 32, upscale: int = 4) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # The network learns the residual over a nearest-neighbour upsampled input.
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    """Five densely connected convolutions with a scaled residual connection."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN / Real-ESRGAN x4 generator."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                 num_block: int = 23, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def build_network(architecture: str, scale: int, num_feat: int, num_conv: int, num_block: int) -> nn.Module:
    """Instantiate the generator described by a model spec."""
    if architecture == "srvgg":
        return SRVGGNetCompact(num_feat=num_feat, num_conv=num_conv, upscale=scale)
    if architecture == "rrdb":
        if scale != 4:
            raise ValueError("Only x4 RRDBNet checkpoints are supported.")
        return RRDBNet(num_feat=num_feat, num_block=num_block)
    raise ValueError(f"Unknown Real-ESRGAN architecture '{architecture}'.")

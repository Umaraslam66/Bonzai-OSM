"""Strided CNN encoder mapping (B, 9, 512, 512) raster -> (B, output_dim, 32, 32) features.

Output is consumed by Stage B (Inker) via cross-attention. Frozen
during Inker training in production (initialised from the diffusion-trained
Stage A encoder); for Phase 0b smoke we train it from scratch alongside
the Inker.

The 3-layer (smoke) variant compresses 512 -> 256 -> 128 -> 64 (stride-2 each)
then a stride-2 stride-1 mix to 32x32. The 4-layer (production) variant uses
4 stride-2 conv layers for cleaner 16x compression.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bonzai_genai.models.configs import RasterEncoderConfig


def _strided_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class RasterEncoder(nn.Module):
    """Strided conv stack + 1x1 projection to ``output_dim`` channels at 32x32."""

    def __init__(self, cfg: RasterEncoderConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        # 4 stride-2 layers always: 512 -> 256 -> 128 -> 64 -> 32.
        layers = [_strided_conv(cfg.in_channels, ch)]   # 512 -> 256
        cur = ch
        layers.append(_strided_conv(cur, cur * 2))      # 256 -> 128
        cur *= 2
        layers.append(_strided_conv(cur, cur * 2))      # 128 -> 64
        cur *= 2
        layers.append(_strided_conv(cur, cur * 2 if cfg.num_layers >= 4 else cur))  # 64 -> 32
        if cfg.num_layers >= 4:
            cur *= 2
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(cur, cfg.output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))

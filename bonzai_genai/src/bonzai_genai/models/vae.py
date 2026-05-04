"""9-channel VAE for the Sketcher pipeline.

Compresses (B, 9, 512, 512) raster -> (B, latent_dim, 64, 64) latent
(8x spatial compression, latent_dim=4). Channel-aware reconstruction
loss (BCE on binary masks, MSE on density channel) + KL regulariser.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bonzai_genai.models.configs import VAEConfig


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class VAEEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        # Initial 1x1 lift
        self.stem = nn.Conv2d(cfg.in_channels, ch, kernel_size=1)
        # Three stride-2 then one stride-1 down-block. 512 -> 256 -> 128 -> 64
        # Each block doubles channels.
        blocks = []
        cur_ch = ch
        for i in range(cfg.num_down_blocks):
            stride = 2 if i < 3 else 1
            blocks.append(_conv_block(cur_ch, cur_ch * 2, stride=stride))
            cur_ch *= 2
        self.down = nn.Sequential(*blocks)
        # Project to 2*latent_dim channels (mu + logvar concatenated)
        self.head = nn.Conv2d(cur_ch, 2 * cfg.latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.down(h)
        h = self.head(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar


# Decoder + VAE forward come in Task 4.
class VAE(nn.Module):
    """Placeholder; decoder + forward land in Task 4."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAEEncoder(cfg)
        # Decoder added in Task 4.

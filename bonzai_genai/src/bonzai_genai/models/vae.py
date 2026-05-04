"""9-channel VAE for the Sketcher pipeline.

Compresses (B, 9, 512, 512) raster -> (B, latent_dim, 64, 64) latent
(8x spatial compression, latent_dim=4). Channel-aware reconstruction
loss (BCE on binary masks, MSE on density channel) + KL regulariser.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from bonzai_genai.models.configs import VAEConfig


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class VAEEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        self.stem = nn.Conv2d(cfg.in_channels, ch, kernel_size=1)
        # Three stride-2 then one stride-1 down-block. 512 -> 256 -> 128 -> 64.
        blocks = []
        cur_ch = ch
        for i in range(cfg.num_down_blocks):
            stride = 2 if i < 3 else 1
            blocks.append(_conv_block(cur_ch, cur_ch * 2, stride=stride))
            cur_ch *= 2
        self.down = nn.Sequential(*blocks)
        self.head = nn.Conv2d(cur_ch, 2 * cfg.latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.down(h)
        h = self.head(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        cur_ch = ch * (2 ** cfg.num_down_blocks)  # 32 * 16 = 512 for tiny
        self.stem = nn.Conv2d(cfg.latent_dim, cur_ch, kernel_size=3, padding=1)
        # Mirror the encoder: 1 stride-1 then 3 up-blocks. 64 -> 128 -> 256 -> 512.
        blocks = []
        for i in range(cfg.num_down_blocks):
            if i == 0:
                blocks.append(_conv_block(cur_ch, cur_ch // 2, stride=1))
            else:
                blocks.append(_up_block(cur_ch, cur_ch // 2))
            cur_ch //= 2
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(cur_ch, cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        h = self.up(h)
        return self.head(h)


class VAE(nn.Module):
    """Full VAE: encode -> reparam -> decode + channel-aware loss.

    Channel layout (must match data/rasteriser.py output):
      0: motorway, 1: trunk, 2: primary, 3: residential (binary)
      4: all-roads (binary)
      5: building density (continuous)
      6: water mask (binary)
      7: green mask (binary)
      8: urban mask (binary)
    """

    BINARY_CHANNELS = (0, 1, 2, 3, 4, 6, 7, 8)
    CONTINUOUS_CHANNELS = (5,)

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAEEncoder(cfg)
        self.decoder = VAEDecoder(cfg)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def compute_losses(
        self, x: torch.Tensor, out: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        recon = out["recon"]
        mu = out["mu"]
        logvar = out["logvar"]
        # Binary channels: BCE-with-logits (decoder outputs raw logits)
        bin_idx = list(self.BINARY_CHANNELS)
        bce = F.binary_cross_entropy_with_logits(
            recon[:, bin_idx], x[:, bin_idx], reduction="mean",
        )
        # Continuous channels: MSE
        cont_idx = list(self.CONTINUOUS_CHANNELS)
        mse = F.mse_loss(recon[:, cont_idx], x[:, cont_idx], reduction="mean")
        # KL divergence to N(0, I), per-element mean
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return {"recon_bce": bce, "recon_mse": mse, "kl": kl}

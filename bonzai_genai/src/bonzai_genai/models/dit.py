"""DiT (Diffusion Transformer) for the Sketcher.

Operates on the VAE latent (B, latent_dim, 64, 64); patches into 1024
transformer tokens (patch_size=2). AdaLN-Zero conditioning per
DiT-XL/2 (Peebles & Xie 2023). Reference: §5.4 of the global design spec.

Built progressively across Plan 2 Tasks 9-12:
    Task  9: PatchEmbed + SinusoidalTimeEmbed
    Task 10: AdaLN-Zero attention block (DiTBlock)
    Task 11: FinalLayer + DiT main module
    Task 12: EDM diffusion + DPM-Solver++ sampler + LightningModule
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from bonzai_genai.models.configs import DiTConfig


class PatchEmbed(nn.Module):
    """Patchify (B, C, H, W) latent into (B, N, hidden_dim) sequence."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        return rearrange(h, "b c h w -> b (h w) c")


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DiT(nn.Module):
    """Stub; full module assembled in Task 11."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.time_embed = SinusoidalTimeEmbed(cfg.cond_dim)

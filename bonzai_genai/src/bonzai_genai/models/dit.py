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


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT transformer block with AdaLN-Zero conditioning.

    Per Peebles & Xie (2023): 6 modulation parameters per block
    (shift/scale/gate × {attn, mlp}); zero-initialised so the
    residual path is identity at init.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        ffn_dim = cfg.hidden_dim * cfg.ffn_expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, cfg.hidden_dim),
        )
        # 6 modulation params per AdaLN-Zero block
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, 6 * cfg.hidden_dim, bias=True),
        )
        # Zero-init the projection to ensure identity residual path at start
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.adaLN(cond)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=-1)
        x_attn = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out
        x_mlp = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class FinalLayer(nn.Module):
    """Final modulation + linear projection back to patch tokens."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(
            cfg.hidden_dim, cfg.patch_size * cfg.patch_size * cfg.in_channels,
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, 2 * cfg.hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class DiT(nn.Module):
    """DiT diffusion transformer over a 64x64 VAE latent.

    Conditioning paths:
      - time embedding (always)
      - text embedding (optional; null vector when not provided)
      - region tag embedding (optional; null vector when not provided)
    All three are summed into a single ``cond_dim`` vector before
    being projected to AdaLN-Zero modulation parameters per block.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.time_embed = SinusoidalTimeEmbed(cfg.cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        # Null cond vectors (learned) for unconditional path
        self.null_text = nn.Parameter(torch.zeros(cfg.cond_dim))
        self.null_tags = nn.Parameter(torch.zeros(cfg.cond_dim))
        # Positional embedding (learned, fixed length = grid_size^2)
        grid = 64 // cfg.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, cfg.hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Transformer blocks
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.final = FinalLayer(cfg)

    def _build_cond(
        self,
        t: torch.Tensor,
        cond_text: torch.Tensor | None,
        cond_tags: torch.Tensor | None,
    ) -> torch.Tensor:
        bs = t.shape[0]
        time = self.time_mlp(self.time_embed(t))
        text = cond_text if cond_text is not None else self.null_text.expand(bs, -1)
        tags = cond_tags if cond_tags is not None else self.null_tags.expand(bs, -1)
        return time + text + tags

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond_text: torch.Tensor | None = None,
        cond_tags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._build_cond(t, cond_text, cond_tags)
        x = self.patch_embed(z) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        x = self.final(x, cond)
        # Unpatchify: (B, N, p*p*C) -> (B, C, H, W)
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        grid = 64 // p
        x = rearrange(
            x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=grid, w=grid, p1=p, p2=p, c=c,
        )
        return x

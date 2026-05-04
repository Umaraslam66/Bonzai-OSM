"""Stage B — autoregressive transformer ("Inker") with cross-attention to a raster encoder.

Architecture per global spec §6.3:
    - Token embedding (vocab_size -> hidden_dim)
    - RoPE on Q, K
    - 12 (smoke) / 24-32 (production) decoder layers, each with:
        - Causal self-attention (RoPE-applied)
        - Cross-attention to raster encoder feature grid
        - FFN
    - Output head: hidden_dim -> vocab_size

Built progressively in Plan 2 Tasks 14-17:
    Task 14: TokenEmbed + RoPE
    Task 15: Inker block (self+cross+ffn) + full module
    Task 16: Constrained decoding masks
    Task 17: Lightning Stage B + samplers
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bonzai_genai.models.configs import InkerConfig


class TokenEmbed(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cosine and sine RoPE caches.

    Returns ``(cos, sin)`` each of shape ``(seq_len, head_dim)`` with
    each consecutive pair ``[c_k, c_k]`` and ``[s_k, s_k]`` for k = 0..head_dim/2-1.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim/2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` with shape (..., seq, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]
    rotated = torch.stack([x1 * cos1 - x2 * sin1, x1 * sin1 + x2 * cos1], dim=-1)
    return rotated.flatten(-2)


class InkerSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE."""

    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:s], sin[:s])
        k = apply_rope(k, cos[:s], sin[:s])
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.hidden_dim)
        return self.out_proj(out)


class InkerCrossAttention(nn.Module):
    """Cross-attention from token sequence to a flat raster feature grid."""

    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.raster_feat_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.raster_feat_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, raster_feat: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        n = raster_feat.shape[1]
        q = self.q_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(raster_feat).view(b, n, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(raster_feat).view(b, n, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.hidden_dim)
        return self.out_proj(out)


class InkerBlock(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.hidden_dim)
        self.self_attn = InkerSelfAttention(cfg)
        self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = InkerCrossAttention(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)
        ffn = cfg.hidden_dim * cfg.ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, ffn),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn, cfg.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        raster_feat: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x), cos, sin)
        x = x + self.cross_attn(self.cross_norm(x), raster_feat)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Inker(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbed(cfg)
        self.blocks = nn.ModuleList([InkerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        head_dim = cfg.hidden_dim // cfg.num_heads
        cos, sin = build_rope_cache(cfg.max_context_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self, tokens: torch.Tensor, raster_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, raster_feat, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Constrained decoding (Plan 2 Task 16). Mandatory subset only:
#   - x -> y coordinate pair completion
#   - Layer-order enforcement (LAND < ROADS < BUILDINGS < POIS)
# Deferred to Plan 3+:
#   - Polygon non-self-intersection
#   - Road-edge node-ref bounds
#   - Building-field ordering (class -> height -> coords)
# ---------------------------------------------------------------------------

from bonzai_genai.config import COORD_BINS  # noqa: E402
from bonzai_genai.vocab.attributes import AttributeVocab  # noqa: E402
from bonzai_genai.vocab.tokens import (  # noqa: E402
    NUM_SPECIAL_TOKENS,
    SpecialToken,
)


def build_constrained_mask(
    state: dict, vocab_size: int, attr_vocab: AttributeVocab,
) -> torch.Tensor:
    """Return a boolean mask of length ``vocab_size``; True = allowed.

    ``state`` keys:
        - ``last_token``: int — last emitted token id (drives x->y enforcement)
        - ``layer``: str — one of "land" / "roads" / "buildings" / "pois"
        - ``phase``: str — coarse decoder phase tag (informational)
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    last = state.get("last_token")
    x_lo = NUM_SPECIAL_TOKENS
    x_hi = x_lo + COORD_BINS
    y_lo = x_hi
    y_hi = y_lo + COORD_BINS
    if last is not None and x_lo <= last < x_hi:
        # After an x-coord, only y-coords are valid.
        mask[y_lo:y_hi] = True
        return mask
    layer = state.get("layer")
    blocked: set[int] = set()
    if layer is not None:
        idx_to_blocked = {
            "land": (),
            "roads": (SpecialToken.LAYER_LAND,),
            "buildings": (SpecialToken.LAYER_LAND, SpecialToken.LAYER_ROADS),
            "pois": (
                SpecialToken.LAYER_LAND,
                SpecialToken.LAYER_ROADS,
                SpecialToken.LAYER_BUILDINGS,
            ),
        }
        for tok in idx_to_blocked.get(layer, ()):
            blocked.add(int(tok))
    mask[:] = True
    for b in blocked:
        mask[b] = False
    return mask

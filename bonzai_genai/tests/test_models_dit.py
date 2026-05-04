"""Tests for DiT components."""
import pytest
import torch

from bonzai_genai.models.configs import DiTConfig, TinyPreset
from bonzai_genai.models.dit import (
    DiT,
    PatchEmbed,
    SinusoidalTimeEmbed,
)


@pytest.fixture
def cfg():
    return DiTConfig.from_preset(TinyPreset)


def test_patch_embed_token_count(cfg):
    pe = PatchEmbed(cfg)
    z = torch.randn(2, cfg.in_channels, 64, 64)
    out = pe(z)
    # 64/2 = 32, 32*32 = 1024
    assert out.shape == (2, 1024, cfg.hidden_dim)


def test_sinusoidal_time_embed_dim(cfg):
    te = SinusoidalTimeEmbed(cfg.cond_dim)
    t = torch.tensor([0.0, 1.5, 999.0])
    out = te(t)
    assert out.shape == (3, cfg.cond_dim)
    assert torch.isfinite(out).all()


def test_dit_block_preserves_shape(cfg):
    from bonzai_genai.models.dit import DiTBlock
    block = DiTBlock(cfg)
    x = torch.randn(2, 1024, cfg.hidden_dim)
    cond = torch.randn(2, cfg.cond_dim)
    out = block(x, cond)
    assert out.shape == x.shape


def test_dit_block_zero_init_residual_path():
    """AdaLN-Zero: at init, gate parameters are 0 so the block is identity."""
    from bonzai_genai.models.dit import DiTBlock
    cfg_small = DiTConfig(hidden_dim=32, num_layers=1, num_heads=4, cond_dim=32, patch_size=2)
    block = DiTBlock(cfg_small)
    x = torch.randn(1, 16, 32)
    cond = torch.zeros(1, 32)
    with torch.no_grad():
        out = block(x, cond)
    # With zero cond + zero-init gates, output ~= input
    assert torch.allclose(out, x, atol=1e-5), f"max diff = {(out-x).abs().max()}"

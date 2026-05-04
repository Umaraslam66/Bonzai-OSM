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

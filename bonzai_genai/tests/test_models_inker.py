"""Tests for the Inker (Stage B autoregressive transformer)."""
import pytest
import torch

from bonzai_genai.models.configs import (
    InkerConfig,
    RasterEncoderConfig,
    TinyPreset,
)


@pytest.fixture
def inker_cfg():
    return InkerConfig.from_preset(TinyPreset)


@pytest.fixture
def raster_cfg():
    return RasterEncoderConfig.from_preset(TinyPreset)


def test_token_embed_output_shape(inker_cfg):
    from bonzai_genai.models.inker import TokenEmbed
    emb = TokenEmbed(inker_cfg)
    tokens = torch.randint(0, inker_cfg.vocab_size, (2, 32))
    out = emb(tokens)
    assert out.shape == (2, 32, inker_cfg.hidden_dim)


def test_rope_applies_rotation(inker_cfg):
    from bonzai_genai.models.inker import build_rope_cache
    head_dim = inker_cfg.hidden_dim // inker_cfg.num_heads
    cos, sin = build_rope_cache(seq_len=64, head_dim=head_dim)
    assert cos.shape == (64, head_dim)
    assert sin.shape == (64, head_dim)
    # cos[0] should be ~1 (no rotation at position 0)
    assert torch.allclose(cos[0], torch.ones(head_dim), atol=1e-5)


def test_inker_forward_returns_logits(inker_cfg, raster_cfg):
    from bonzai_genai.models.inker import Inker
    inker = Inker(inker_cfg)
    tokens = torch.randint(0, inker_cfg.vocab_size, (2, 32))
    raster_feat = torch.randn(2, 32 * 32, raster_cfg.output_dim)
    logits = inker(tokens, raster_feat)
    assert logits.shape == (2, 32, inker_cfg.vocab_size)


def test_inker_causal_mask_blocks_future_tokens(inker_cfg, raster_cfg):
    from bonzai_genai.models.inker import Inker
    inker = Inker(inker_cfg)
    inker.eval()
    tokens = torch.randint(0, inker_cfg.vocab_size, (1, 16))
    raster_feat = torch.randn(1, 32 * 32, raster_cfg.output_dim)
    with torch.no_grad():
        out_full = inker(tokens, raster_feat)
        tokens_perturbed = tokens.clone()
        tokens_perturbed[:, 8:] = 0
        out_perturbed = inker(tokens_perturbed, raster_feat)
    # First 8 positions identical because causal mask blocks future
    assert torch.allclose(out_full[:, :8], out_perturbed[:, :8], atol=1e-5)

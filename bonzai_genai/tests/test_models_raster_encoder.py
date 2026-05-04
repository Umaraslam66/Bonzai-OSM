"""Tests for the small CNN raster encoder used by Stage B for cross-attention."""
import pytest
import torch

from bonzai_genai.models.configs import RasterEncoderConfig, TinyPreset
from bonzai_genai.models.raster_encoder import RasterEncoder


@pytest.fixture
def cfg():
    return RasterEncoderConfig.from_preset(TinyPreset)


def test_output_shape_is_32x32_with_output_dim_channels(cfg):
    enc = RasterEncoder(cfg)
    x = torch.randn(2, cfg.in_channels, 512, 512)
    feat = enc(x)
    assert feat.shape == (2, cfg.output_dim, 32, 32)


def test_grid_can_be_flattened_for_cross_attention(cfg):
    enc = RasterEncoder(cfg)
    x = torch.randn(1, 9, 512, 512)
    feat = enc(x)
    flat = feat.flatten(2).transpose(1, 2)  # (B, 1024, output_dim)
    assert flat.shape == (1, 1024, cfg.output_dim)

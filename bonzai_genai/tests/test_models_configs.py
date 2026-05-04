"""Tests for model configuration dataclasses."""
import pytest

from bonzai_genai.models.configs import (
    DiTConfig,
    InkerConfig,
    ProductionPreset,
    RasterEncoderConfig,
    TinyPreset,
    VAEConfig,
)


def test_tiny_preset_has_smaller_params_than_production():
    tiny_dit = DiTConfig.from_preset(TinyPreset)
    prod_dit = DiTConfig.from_preset(ProductionPreset)
    assert tiny_dit.hidden_dim < prod_dit.hidden_dim
    assert tiny_dit.num_layers < prod_dit.num_layers


def test_vae_latent_shape_consistent():
    cfg = VAEConfig.from_preset(TinyPreset)
    assert cfg.latent_dim == 4
    assert cfg.spatial_compression == 8


def test_dit_patch_count_consistent():
    """Patch size 2 over 64x64 latent -> 32x32 = 1024 transformer tokens."""
    cfg = DiTConfig.from_preset(TinyPreset)
    assert cfg.patch_size == 2
    grid = 64 // cfg.patch_size
    assert grid * grid == 1024


def test_inker_context_length():
    tiny = InkerConfig.from_preset(TinyPreset)
    prod = InkerConfig.from_preset(ProductionPreset)
    assert tiny.max_context_len == 4096
    assert prod.max_context_len == 16384


def test_raster_encoder_output_dim():
    tiny = RasterEncoderConfig.from_preset(TinyPreset)
    prod = RasterEncoderConfig.from_preset(ProductionPreset)
    assert tiny.output_dim == 256
    assert prod.output_dim == 768


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        DiTConfig.from_preset("medium")

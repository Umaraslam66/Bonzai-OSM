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


from bonzai_genai.models.configs import Plan3Preset  # noqa: E402


def test_plan3_preset_is_registered():
    cfg = DiTConfig.from_preset(Plan3Preset)
    assert isinstance(cfg, DiTConfig)


def test_plan3_dit_sits_between_tiny_and_production():
    tiny = DiTConfig.from_preset(TinyPreset)
    plan3 = DiTConfig.from_preset(Plan3Preset)
    prod = DiTConfig.from_preset(ProductionPreset)
    assert tiny.hidden_dim < plan3.hidden_dim < prod.hidden_dim
    assert tiny.num_layers < plan3.num_layers < prod.num_layers


def test_plan3_inker_sits_between_tiny_and_production():
    tiny = InkerConfig.from_preset(TinyPreset)
    plan3 = InkerConfig.from_preset(Plan3Preset)
    prod = InkerConfig.from_preset(ProductionPreset)
    assert tiny.hidden_dim <= plan3.hidden_dim < prod.hidden_dim
    assert tiny.num_layers < plan3.num_layers <= prod.num_layers
    assert tiny.max_context_len <= plan3.max_context_len < prod.max_context_len


def test_plan3_inker_context_is_8k():
    plan3 = InkerConfig.from_preset(Plan3Preset)
    assert plan3.max_context_len == 8192


def test_plan3_raster_encoder_output_dim():
    plan3 = RasterEncoderConfig.from_preset(Plan3Preset)
    assert plan3.output_dim == 512


def test_plan3_vae_matches_tiny_for_smoke_ckpt_reuse():
    """Plan 3 warm-starts from the Phase 0b smoke VAE checkpoint
    (base_channels=32). The shapes must match or load_state_dict will
    fail with a `size mismatch` error on Leonardo. A bigger VAE is
    deferred to Phase 3 production training.
    """
    tiny = VAEConfig.from_preset(TinyPreset)
    plan3 = VAEConfig.from_preset(Plan3Preset)
    assert plan3.base_channels == tiny.base_channels
    assert plan3.latent_dim == tiny.latent_dim
    assert plan3.spatial_compression == tiny.spatial_compression
    assert plan3.num_down_blocks == tiny.num_down_blocks

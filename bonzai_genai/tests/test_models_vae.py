"""Tests for the VAE."""
import pytest
import torch

from bonzai_genai.models.configs import TinyPreset, VAEConfig
from bonzai_genai.models.vae import VAE, VAEEncoder


@pytest.fixture
def cfg():
    return VAEConfig.from_preset(TinyPreset)


def test_encoder_output_shape(cfg):
    enc = VAEEncoder(cfg)
    x = torch.randn(2, 9, 512, 512)
    mu, logvar = enc(x)
    # 8x spatial compression: 512 -> 64
    assert mu.shape == (2, cfg.latent_dim, 64, 64)
    assert logvar.shape == (2, cfg.latent_dim, 64, 64)


def test_encoder_logvar_is_clamped_for_stability(cfg):
    enc = VAEEncoder(cfg)
    x = torch.randn(1, 9, 512, 512) * 1e6  # huge inputs
    _, logvar = enc(x)
    assert torch.all(logvar >= -10) and torch.all(logvar <= 10), "logvar must be clamped"

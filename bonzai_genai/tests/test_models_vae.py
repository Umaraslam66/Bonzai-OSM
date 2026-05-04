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


def test_decoder_output_shape(cfg):
    from bonzai_genai.models.vae import VAEDecoder
    dec = VAEDecoder(cfg)
    z = torch.randn(2, cfg.latent_dim, 64, 64)
    out = dec(z)
    assert out.shape == (2, cfg.in_channels, 512, 512)


def test_vae_forward_returns_recon_mu_logvar(cfg):
    vae = VAE(cfg)
    x = torch.randn(2, 9, 512, 512)
    out = vae(x)
    assert "recon" in out and "mu" in out and "logvar" in out
    assert out["recon"].shape == x.shape


def test_reconstruction_loss_breaks_down_by_channel(cfg):
    vae = VAE(cfg)
    x = torch.zeros(1, 9, 512, 512)
    x[:, :5] = (torch.rand_like(x[:, :5]) > 0.5).float()  # binary masks
    x[:, 5] = torch.rand_like(x[:, 5])                     # density continuous
    x[:, 6:] = (torch.rand_like(x[:, 6:]) > 0.5).float()
    out = vae(x)
    losses = vae.compute_losses(x, out)
    assert "recon_bce" in losses and "recon_mse" in losses and "kl" in losses
    assert all(torch.isfinite(v) for v in losses.values())


def test_kl_loss_is_zero_for_unit_gaussian(cfg):
    vae = VAE(cfg)
    out = {
        "recon": torch.zeros(1, 9, 512, 512),
        "mu": torch.zeros(1, 4, 64, 64),
        "logvar": torch.zeros(1, 4, 64, 64),
    }
    losses = vae.compute_losses(torch.zeros(1, 9, 512, 512), out)
    assert losses["kl"].abs() < 1e-6

"""Tests for diffusion samplers."""
import torch

from bonzai_genai.models.configs import DiTConfig, TinyPreset
from bonzai_genai.models.dit import DiT
from bonzai_genai.training.samplers import dpmpp_sample


def test_dpmpp_returns_tensor_of_latent_shape():
    cfg = DiTConfig.from_preset(TinyPreset)
    dit = DiT(cfg)
    dit.eval()
    samples = dpmpp_sample(
        dit, batch_size=2, num_steps=10, latent_shape=(cfg.in_channels, 64, 64),
        device=torch.device("cpu"),
    )
    assert samples.shape == (2, cfg.in_channels, 64, 64)
    assert torch.isfinite(samples).all()

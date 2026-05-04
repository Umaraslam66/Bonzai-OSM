"""Lightning module smoke tests: each module should run forward + backward + step on tiny inputs."""
import pytest
import torch

from bonzai_genai.models.configs import TinyPreset, VAEConfig


@pytest.fixture
def synth_raster_batch():
    # Tiny batch shaped like the real one (B, 9, 512, 512) but small B=1
    x = torch.zeros(1, 9, 512, 512)
    x[:, :5] = (torch.rand_like(x[:, :5]) > 0.7).float()
    x[:, 5] = torch.rand_like(x[:, 5])
    x[:, 6:] = (torch.rand_like(x[:, 6:]) > 0.7).float()
    return x


def test_lit_vae_one_training_step(synth_raster_batch):
    from bonzai_genai.training.lit_vae import LitVAE
    lit = LitVAE(vae_config=VAEConfig.from_preset(TinyPreset), kl_weight=0.01, lr=1e-4)
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    loss = lit.training_step(synth_raster_batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

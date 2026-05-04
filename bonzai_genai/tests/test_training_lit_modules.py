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


def test_lit_stage_a_one_training_step(synth_raster_batch):
    from bonzai_genai.models.configs import DiTConfig
    from bonzai_genai.training.lit_stage_a import LitStageA
    lit = LitStageA(
        dit_config=DiTConfig.from_preset(TinyPreset),
        vae_config=VAEConfig.from_preset(TinyPreset),
        cfg_dropout_prob=0.1,
        lr=1e-4,
    )
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    loss = lit.training_step(synth_raster_batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_lit_stage_b_one_training_step():
    from bonzai_genai.models.configs import (
        InkerConfig,
        RasterEncoderConfig,
    )
    from bonzai_genai.training.lit_stage_b import LitStageB
    lit = LitStageB(
        inker_config=InkerConfig.from_preset(TinyPreset),
        raster_encoder_config=RasterEncoderConfig.from_preset(TinyPreset),
        lr=3e-4,
    )
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    batch = {
        "raster": torch.randn(1, 9, 512, 512),
        "tokens": torch.randint(0, 1000, (1, 64)),
        "token_lens": torch.tensor([64]),
    }
    loss = lit.training_step(batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

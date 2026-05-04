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


def test_greedy_inker_sample_returns_token_sequence():
    from bonzai_genai.models.configs import InkerConfig, RasterEncoderConfig
    from bonzai_genai.models.inker import Inker
    from bonzai_genai.models.raster_encoder import RasterEncoder
    from bonzai_genai.training.samplers import greedy_inker_sample

    icfg = InkerConfig.from_preset(TinyPreset)
    rcfg = RasterEncoderConfig.from_preset(TinyPreset)
    inker = Inker(icfg)
    enc = RasterEncoder(rcfg)
    raster = torch.randn(1, 9, 512, 512)
    seq = greedy_inker_sample(
        inker, enc, raster, max_tokens=16, bos_id=0, eos_id=1,
    )
    assert seq.shape[0] == 1
    assert seq.shape[1] <= 17  # bos + up to 16 generated

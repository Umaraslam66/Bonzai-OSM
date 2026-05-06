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


def test_kv_cached_greedy_matches_recompute_greedy():
    """The KV-cached sampler must produce the IDENTICAL token sequence as
    the slow recompute sampler. Otherwise the cache is wrong.

    Argmax is deterministic; the only allowable diff is floating-point
    associativity in the attention sums, which doesn't flip an argmax in
    practice on this scale. We use a fixed-seed model and a small max_tokens
    so the test runs in <2 s on CPU.
    """
    torch.manual_seed(0)
    from bonzai_genai.models.configs import InkerConfig, RasterEncoderConfig
    from bonzai_genai.models.inker import Inker
    from bonzai_genai.models.raster_encoder import RasterEncoder
    from bonzai_genai.training.samplers import (
        greedy_inker_sample,
        greedy_inker_sample_cached,
    )

    icfg = InkerConfig.from_preset(TinyPreset)
    rcfg = RasterEncoderConfig.from_preset(TinyPreset)
    inker = Inker(icfg)
    enc = RasterEncoder(rcfg)
    raster = torch.randn(2, 9, 512, 512)

    seq_slow = greedy_inker_sample(
        inker, enc, raster, max_tokens=24, bos_id=0, eos_id=1,
    )
    seq_fast = greedy_inker_sample_cached(
        inker, enc, raster, max_tokens=24, bos_id=0, eos_id=1,
    )
    assert seq_slow.shape == seq_fast.shape
    assert torch.equal(seq_slow, seq_fast), (
        f"cached sampler diverged from recompute sampler:\n"
        f"slow={seq_slow}\nfast={seq_fast}"
    )


def test_kv_cached_greedy_handles_eos_stop():
    """Cached sampler must stop early when all batch elements emit EOS.

    Stub forward_step to always return logits that pick EOS (id=1). This
    exercises the loop's early-exit path independent of model weights.
    """
    torch.manual_seed(0)
    from bonzai_genai.models.configs import InkerConfig, RasterEncoderConfig
    from bonzai_genai.models.inker import Inker
    from bonzai_genai.models.raster_encoder import RasterEncoder
    from bonzai_genai.training.samplers import greedy_inker_sample_cached

    icfg = InkerConfig.from_preset(TinyPreset)
    rcfg = RasterEncoderConfig.from_preset(TinyPreset)
    inker = Inker(icfg)
    enc = RasterEncoder(rcfg)
    raster = torch.randn(1, 9, 512, 512)

    eos_id = 1
    vocab = icfg.vocab_size
    real_forward_step = inker.forward_step

    def force_eos_step(token, cross_kvs, past_self_kvs, position):
        # Run the real step so caches stay shape-correct, then overwrite logits.
        _, new_kvs = real_forward_step(token, cross_kvs, past_self_kvs, position)
        logits = torch.full((token.shape[0], 1, vocab), -1e9)
        logits[..., eos_id] = 1.0
        return logits, new_kvs

    inker.forward_step = force_eos_step  # type: ignore[assignment]
    seq = greedy_inker_sample_cached(
        inker, enc, raster, max_tokens=64, bos_id=0, eos_id=eos_id,
    )
    # bos + one generated EOS → length 2
    assert seq.shape[1] == 2
    assert int(seq[0, 1]) == eos_id

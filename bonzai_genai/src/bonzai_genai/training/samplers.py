"""Diffusion samplers and AR samplers.

DPM-Solver++ for Stage A (DiT) sampling. Full reference: §5.7 of the
global design spec. We implement the second-order multistep variant
(``dpmpp_2m``); 50 steps gives high quality for Phase 4 production
and 10-25 steps suffices for smoke runs.

Greedy + beam-search Inker samplers added in Plan 2 Task 17.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def greedy_inker_sample(
    inker,
    raster_encoder,
    raster: torch.Tensor,
    *,
    max_tokens: int,
    bos_id: int,
    eos_id: int,
    constrained: bool = False,
) -> torch.Tensor:
    """Greedy decode from BOS until EOS or ``max_tokens``.

    Returns ``(B, T)`` token tensor including the BOS prefix.
    Constrained-decoding logit masking is applied when ``constrained=True``.
    """
    inker.eval()
    raster_encoder.eval()
    device = raster.device
    bs = raster.shape[0]
    feat = raster_encoder(raster)                 # (B, D, 32, 32)
    feat_seq = feat.flatten(2).transpose(1, 2)    # (B, 1024, D)
    tokens = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(max_tokens):
        logits = inker(tokens, feat_seq)
        next_logits = logits[:, -1]
        if constrained:
            from bonzai_genai.models.inker import build_constrained_mask
            from bonzai_genai.vocab.attributes import load_default_vocab
            attr_vocab = load_default_vocab()
            for b in range(bs):
                state = {
                    "phase": "header",
                    "layer": None,
                    "last_token": int(tokens[b, -1]),
                }
                mask = build_constrained_mask(
                    state, next_logits.shape[-1], attr_vocab,
                ).to(device)
                next_logits[b, ~mask] = -1e9
        nxt = next_logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
        if (nxt == eos_id).all():
            break
    return tokens


@torch.no_grad()
def dpmpp_sample(
    model: nn.Module,
    *,
    batch_size: int,
    num_steps: int,
    latent_shape: tuple[int, int, int],
    device: torch.device,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    cond_text: torch.Tensor | None = None,
    cond_tags: torch.Tensor | None = None,
) -> torch.Tensor:
    """DPM-Solver++ 2M sampler over an EDM noise schedule.

    Returns ``(B, C, H, W)`` denoised latents.
    """
    # Karras EDM noise schedule
    ramp = torch.linspace(0, 1, num_steps + 1, device=device)
    sigmas = (
        sigma_max ** (1 / rho)
        + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])  # final sigma = 0
    x = torch.randn(batch_size, *latent_shape, device=device) * sigma_max
    old_denoised: torch.Tensor | None = None
    for i in range(num_steps):
        sigma = sigmas[i]
        t = sigma.expand(batch_size)
        denoised = model(x, t, cond_text=cond_text, cond_tags=cond_tags)
        if old_denoised is None or i == num_steps - 1:
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        else:
            h = sigmas[i + 1] - sigma
            r = sigmas[i] / sigmas[i - 1]
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            d = (x - denoised_d) / sigma
            x = x + d * h
        old_denoised = denoised
    return x

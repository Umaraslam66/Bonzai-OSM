"""End-to-end metrics: pipe DiT raster -> Inker -> decoded GeoJSON -> re-rasterise -> IoU."""
from __future__ import annotations

import torch

from bonzai_genai.data.rasteriser import rasterise
from bonzai_genai.eval.stage_a import channel_iou
from bonzai_genai.vocab.attributes import AttributeVocab
from bonzai_genai.vocab.tokeniser import Tokeniser


def end_to_end_channel_iou(
    sampled_rasters: torch.Tensor,
    sampled_token_sequences: list[list[int]],
    vocab: AttributeVocab,
) -> dict[int, float]:
    """For each (sampled_raster, sampled_tokens) pair, decode tokens, re-rasterise, IoU.

    Measures whether Stage B's sampled tokens are *consistent* with the Stage A raster.
    """
    tok = Tokeniser(vocab)
    re_rasters = []
    for seq in sampled_token_sequences:
        try:
            geom = tok.decode(list(seq))
            re_rasters.append(torch.from_numpy(rasterise(geom)).float())
        except Exception:
            re_rasters.append(torch.zeros_like(sampled_rasters[0]))
    re_raster_t = torch.stack(re_rasters)
    return channel_iou(re_raster_t, sampled_rasters)

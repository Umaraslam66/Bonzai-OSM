"""LightningModule for Stage B (Inker) training.

Cross-entropy on next-token prediction with teacher-forcing. Cross-attention
to the raster CNN encoder's output (for Phase 0b smoke we use the
ground-truth raster — no domain gap; that's Experiment 3's job).
"""
from __future__ import annotations

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from torch.optim import AdamW

from bonzai_genai.models.configs import InkerConfig, RasterEncoderConfig
from bonzai_genai.models.inker import Inker
from bonzai_genai.models.raster_encoder import RasterEncoder


class LitStageB(L.LightningModule):
    def __init__(
        self,
        inker_config: InkerConfig,
        raster_encoder_config: RasterEncoderConfig,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["inker_config", "raster_encoder_config"])
        self.inker = Inker(inker_config)
        self.encoder = RasterEncoder(raster_encoder_config)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        raster = batch["raster"]
        tokens = batch["tokens"]      # (B, T)
        lens = batch["token_lens"]    # (B,)
        feat = self.encoder(raster).flatten(2).transpose(1, 2)
        # Teacher-forcing: predict tokens[1:] from tokens[:-1]
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = self.inker(inputs, feat)
        bs, t, v = logits.shape
        # Per-position validity mask: position i is valid iff i < lens - 1
        pos = torch.arange(t, device=logits.device).unsqueeze(0).expand(bs, -1)
        valid = pos < (lens - 1).unsqueeze(1)
        loss = F.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            reduction="none",
        )
        loss = (loss * valid.reshape(-1).float()).sum() / valid.sum().clamp(min=1)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return AdamW(
            list(self.inker.parameters()) + list(self.encoder.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

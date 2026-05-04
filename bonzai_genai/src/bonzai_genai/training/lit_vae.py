"""LightningModule for VAE reconstruction training (Phase 0b smoke + Phase 3 production)."""
from __future__ import annotations

import lightning as L  # noqa: N812
import torch
from torch.optim import AdamW

from bonzai_genai.models.configs import VAEConfig
from bonzai_genai.models.vae import VAE


class LitVAE(L.LightningModule):
    def __init__(
        self,
        vae_config: VAEConfig,
        kl_weight: float = 0.01,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vae_config"])
        self.vae_config = vae_config
        self.vae = VAE(vae_config)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.vae(batch)
        losses = self.vae.compute_losses(batch, out)
        total = (
            losses["recon_bce"]
            + losses["recon_mse"]
            + self.hparams.kl_weight * losses["kl"]
        )
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=False)
        self.log("train/loss", total, prog_bar=True)
        return total

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self.vae(batch)
        losses = self.vae.compute_losses(batch, out)
        total = (
            losses["recon_bce"]
            + losses["recon_mse"]
            + self.hparams.kl_weight * losses["kl"]
        )
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, prog_bar=False)
        self.log("val/loss", total, prog_bar=True)
        return total

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

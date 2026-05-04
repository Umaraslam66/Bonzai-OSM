"""LightningModule for Stage A (DiT) training in latent space.

Pipeline per training step:
    1. Encode raster batch via frozen VAE -> latent z0.
    2. Sample sigma per example from log-normal EDM distribution.
    3. Add noise: x = z0 + sigma * eps.
    4. Predict denoised latent: x_hat = DiT(x, sigma, cond).
    5. Loss = MSE between x_hat and z0, weighted by EDM weight.

Classifier-free guidance: 10% of training drops conditioning to null.
"""
from __future__ import annotations

import lightning as L  # noqa: N812
import torch
from torch.optim import AdamW

from bonzai_genai.models.configs import DiTConfig, VAEConfig
from bonzai_genai.models.dit import DiT
from bonzai_genai.models.vae import VAE


class LitStageA(L.LightningModule):
    def __init__(
        self,
        dit_config: DiTConfig,
        vae_config: VAEConfig,
        cfg_dropout_prob: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dit_config", "vae_config"])
        self.dit = DiT(dit_config)
        self.vae = VAE(vae_config)
        for p in self.vae.parameters():
            p.requires_grad = False

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        log_sigma = (
            self.hparams.p_mean
            + self.hparams.p_std * torch.randn(batch_size, device=device)
        )
        return log_sigma.exp()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # batch is the raw raster (B, 9, 512, 512). Encode -> z0.
        with torch.no_grad():
            mu, _ = self.vae.encoder(batch)
            z0 = mu  # use mean for stability; reparam is reserved for VAE training
        bs = z0.shape[0]
        sigma = self._sample_sigma(bs, z0.device)
        sigma_b = sigma.view(bs, 1, 1, 1)
        eps = torch.randn_like(z0)
        x_noisy = z0 + sigma_b * eps
        # CFG dropout: this smoke run is unconditional; conditioning paths
        # exist but are dropped to null for Exp 0.
        x_hat = self.dit(x_noisy, sigma, cond_text=None, cond_tags=None)
        # EDM weighting: w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        sigma_data = self.hparams.sigma_data
        w = (sigma_b ** 2 + sigma_data ** 2) / ((sigma_b * sigma_data) ** 2)
        loss = (w * (x_hat - z0) ** 2).mean()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return AdamW(
            [p for p in self.dit.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

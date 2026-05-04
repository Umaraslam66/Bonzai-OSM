"""Shared Lightning trainer driver for VAE / Stage A / Stage B training.

Driven entirely by env-vars set in the sbatch script:
  BONZAI_STAGE                 "vae" | "stage_a" | "stage_b"
  BONZAI_PRESET                "tiny" | "plan3" | "production"
  BONZAI_TRAIN_URL             WebDataset shard glob for training
  BONZAI_VAL_URL               WebDataset shard glob for validation
  BONZAI_OUT                   directory for checkpoints + logs
  BONZAI_BATCH_SIZE            integer
  BONZAI_MAX_EPOCHS            integer (smoke = 1; production = 50-100)
  BONZAI_VAE_CKPT              (stage_a only) frozen VAE checkpoint path
  BONZAI_GRAD_ACCUM            (optional) gradient accumulation steps
  BONZAI_DDP_DEVICES           (optional) >1 enables DDP across that many GPUs
  BONZAI_COUNTRY_WEIGHTS_JSON  (optional) JSON dict country -> sampling weight
                               When set, train DataLoader uses
                               country_balance_filter for per-country balance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Editable-install fallback (same workaround as conftest.py / prepare_tiles_local.py)
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import lightning as L  # noqa: E402, N812

from bonzai_genai.models.configs import (  # noqa: E402
    DiTConfig,
    InkerConfig,
    RasterEncoderConfig,
    VAEConfig,
)
from bonzai_genai.training.data_module import TileDataModule  # noqa: E402


def main() -> None:
    stage = os.environ["BONZAI_STAGE"]
    preset = os.environ.get("BONZAI_PRESET", "tiny")
    train_url = os.environ["BONZAI_TRAIN_URL"]
    val_url = os.environ["BONZAI_VAL_URL"]
    out_dir = Path(os.environ["BONZAI_OUT"])
    batch_size = int(os.environ.get("BONZAI_BATCH_SIZE", "8"))
    max_epochs = int(os.environ.get("BONZAI_MAX_EPOCHS", "1"))

    out_dir.mkdir(parents=True, exist_ok=True)

    weights_json = os.environ.get("BONZAI_COUNTRY_WEIGHTS_JSON")
    country_weights: dict[str, float] | None = (
        json.loads(weights_json) if weights_json else None
    )
    balance_by_country = country_weights is not None

    if stage == "vae":
        from bonzai_genai.training.lit_vae import LitVAE
        lit = LitVAE(vae_config=VAEConfig.from_preset(preset))
        dm = TileDataModule(
            train_url=train_url, val_url=val_url, batch_size=batch_size,
            return_tokens=False, num_workers=4,
            balance_by_country=balance_by_country,
            country_weights=country_weights,
        )
    elif stage == "stage_a":
        from bonzai_genai.training.lit_stage_a import LitStageA
        lit = LitStageA(
            dit_config=DiTConfig.from_preset(preset),
            vae_config=VAEConfig.from_preset(preset),
        )
        ckpt_path = os.environ.get("BONZAI_VAE_CKPT")
        if ckpt_path:
            from bonzai_genai.training.lit_vae import LitVAE
            vae_lit = LitVAE.load_from_checkpoint(
                ckpt_path, vae_config=VAEConfig.from_preset(preset),
            )
            lit.vae.load_state_dict(vae_lit.vae.state_dict())
            for p in lit.vae.parameters():
                p.requires_grad = False
        dm = TileDataModule(
            train_url=train_url, val_url=val_url, batch_size=batch_size,
            return_tokens=False, num_workers=4,
            balance_by_country=balance_by_country,
            country_weights=country_weights,
        )
    elif stage == "stage_b":
        from bonzai_genai.training.lit_stage_b import LitStageB
        lit = LitStageB(
            inker_config=InkerConfig.from_preset(preset),
            raster_encoder_config=RasterEncoderConfig.from_preset(preset),
        )
        dm = TileDataModule(
            train_url=train_url, val_url=val_url, batch_size=batch_size,
            return_tokens=True, num_workers=4,
            max_token_len=InkerConfig.from_preset(preset).max_context_len,
            balance_by_country=balance_by_country,
            country_weights=country_weights,
        )
    else:
        raise SystemExit(f"unknown BONZAI_STAGE: {stage}")

    devices = int(os.environ.get("BONZAI_DDP_DEVICES", "1"))
    trainer_kwargs: dict = dict(
        max_epochs=max_epochs,
        default_root_dir=str(out_dir),
        log_every_n_steps=10,
        accumulate_grad_batches=int(os.environ.get("BONZAI_GRAD_ACCUM", "1")),
        precision="bf16-mixed",
    )
    if devices > 1:
        # Saturate the full Leonardo node — boost_usr_prod bills per node
        # regardless of GPU count. See feedback_leonardo_full_node.md.
        trainer_kwargs.update(
            accelerator="gpu",
            devices=devices,
            strategy="ddp",
            sync_batchnorm=True,
            # We use streaming WebDataset with wds.split_by_node; Lightning
            # must NOT replace our sampler with its own DistributedSampler.
            use_distributed_sampler=False,
        )
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()

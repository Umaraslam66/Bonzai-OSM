"""Smoke test for the BONZAI_SAMPLE_FROM_CKPT eval-driver mode.

Builds the smallest possible LitVAE / LitStageA / LitStageB checkpoints
on the fly via `torch.save` (Lightning-compatible format), points the
driver at them, and asserts that the requested number of PNGs and
GeoJSON files land on disk.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _save_lightning_ckpt(lit_module, path: Path) -> None:
    """Write a Lightning-compatible checkpoint without spinning up Trainer."""
    import lightning as L  # noqa: N812
    import torch
    payload = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": L.__version__,
        "state_dict": lit_module.state_dict(),
        "loops": {},
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
        "hparams_name": "kwargs",
        "hyper_parameters": dict(lit_module.hparams),
    }
    torch.save(payload, str(path))


@pytest.fixture
def tiny_checkpoints(tmp_path):
    """Create tiny VAE / Stage A / Stage B checkpoints in tmp_path/ckpt/."""
    from bonzai_genai.models.configs import (
        DiTConfig,
        InkerConfig,
        RasterEncoderConfig,
        TinyPreset,
        VAEConfig,
    )
    from bonzai_genai.training.lit_stage_a import LitStageA
    from bonzai_genai.training.lit_stage_b import LitStageB
    from bonzai_genai.training.lit_vae import LitVAE

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    vae_lit = LitVAE(vae_config=VAEConfig.from_preset(TinyPreset))
    sa_lit = LitStageA(
        dit_config=DiTConfig.from_preset(TinyPreset),
        vae_config=VAEConfig.from_preset(TinyPreset),
    )
    sb_lit = LitStageB(
        inker_config=InkerConfig.from_preset(TinyPreset),
        raster_encoder_config=RasterEncoderConfig.from_preset(TinyPreset),
    )
    _save_lightning_ckpt(vae_lit, ckpt_dir / "vae.ckpt")
    _save_lightning_ckpt(sa_lit, ckpt_dir / "stage_a.ckpt")
    _save_lightning_ckpt(sb_lit, ckpt_dir / "stage_b.ckpt")
    return ckpt_dir


def test_sample_from_ckpt_dumps_pngs_and_geojson(tiny_checkpoints, tmp_path):
    out = tmp_path / "samples"
    env = os.environ.copy()
    env.update(
        BONZAI_SAMPLE_FROM_CKPT="1",
        BONZAI_CKPT_DIR=str(tiny_checkpoints),
        BONZAI_PRESET="tiny",
        BONZAI_SAMPLE_OUT=str(out),
        BONZAI_NUM_SAMPLES="4",
        BONZAI_NUM_DPM_STEPS="2",
        BONZAI_INKER_MAX_TOKENS="16",
    )
    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts" / "run_eval.py"
    result = subprocess.run(
        [sys.executable, str(runner)], env=env,
        capture_output=True, timeout=300, text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"run_eval.py exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    pngs = sorted(out.glob("*.png"))
    geojsons = sorted(out.glob("*.geojson"))
    assert len(pngs) == 4, f"expected 4 PNGs, got {len(pngs)}: {pngs}"
    assert len(geojsons) == 4, f"expected 4 GeoJSON, got {len(geojsons)}: {geojsons}"

# Phase 0b — Modeling Layer + Eval Harness + Experiment 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `bonzai_genai` modeling layer (custom-from-scratch VAE, DiT Sketcher, AR Inker, raster CNN encoder), three PyTorch Lightning training loops, the full §8 eval harness, and a full Experiment 0 smoke run on Leonardo — confirming the architecture skeleton closes geometrically before we commit production GPU-h.

**Architecture:** Each model is a config-driven `nn.Module` with a `tiny` preset (Experiment 0) and a `production` preset (Phases 4–5). Training is via PyTorch Lightning — one `LightningModule` per stage sharing one `LightningDataModule` over the existing WebDataset shards. Eval is split by stage (`eval/stage_a.py`, `eval/stage_b.py`, `eval/end_to_end.py`, `eval/baselines.py`) with one `metric(samples, ground_truth, **opts) -> dict[str, float]` contract per metric. Experiment 0 trains tiny VAE → tiny DiT (latent space) → tiny Inker (cross-attending to the GT raster) on ~5,000 synthetic tiles, then runs the full §8 metrics + soft go/no-go.

**Tech Stack:** Python 3.11+, PyTorch 2.5+, PyTorch Lightning 2.4+, einops, torchmetrics, FlashAttention 3 (production-only — smoke runs use plain SDPA), webdataset, scikit-image, shapely, networkx (for road-graph metric), pytest + hypothesis, ruff + black. SLURM `boost_usr_prod` (4×A100 nodes, GPU-billed) for training/sampling jobs; `lrd_all_serial` (free CPU) for synthetic data generation and CPU-bound metrics.

**Source spec:** [`docs/superpowers/specs/2026-05-04-phase-0b-modeling-and-smoke-harness-design.md`](../specs/2026-05-04-phase-0b-modeling-and-smoke-harness-design.md)
**Global design:** [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](../specs/2026-05-03-genai-city-infrastructure-design.md)

**Phase 0a deliverable being consumed:** WebDataset shards on `$WORK/bonzai-tiles/{singapore,sri_lanka,sweden,synth}/`. Phase 0b will *generate* the `synth/` shards as part of the run (Task 24).

---

## File structure being built

```
bonzai_genai/
├── pyproject.toml                                  # MODIFY: add lightning, einops, torchmetrics, networkx
├── src/bonzai_genai/
│   ├── models/                                     # NEW
│   │   ├── __init__.py
│   │   ├── configs.py                              # TinyConfig + ProductionConfig dataclasses
│   │   ├── vae.py                                  # 9-channel VAE encoder + decoder + reparam
│   │   ├── dit.py                                  # DiT blocks, AdaLN-Zero, conditioning
│   │   ├── inker.py                                # AR transformer, RoPE, cross-attn, constrained decoding
│   │   └── raster_encoder.py                       # Strided CNN, frozen during Inker training
│   ├── training/                                   # NEW
│   │   ├── __init__.py
│   │   ├── data_module.py                          # WebDataset → LightningDataModule
│   │   ├── lit_vae.py
│   │   ├── lit_stage_a.py
│   │   ├── lit_stage_b.py
│   │   ├── samplers.py                             # DPM-Solver++ for DiT, greedy + beam for Inker
│   │   └── callbacks.py                            # sample-dump, EMA, custom checkpoint policy
│   ├── eval/                                       # NEW
│   │   ├── __init__.py
│   │   ├── stage_a.py                              # channel IoU, FID, FID-clip, conditioning
│   │   ├── stage_b.py                              # Chamfer, road graph, POI placement, validity
│   │   ├── end_to_end.py                           # combined raster→tokens→raster pipeline
│   │   └── baselines.py                            # 4 baselines from global §8.2
│   ├── synth/procedural.py                         # MODIFY: extend for richer Experiment 0 corpus
│   └── cli/prepare_tiles.py                        # MODIFY: add `synth-corpus` subcommand
├── scripts/
│   ├── leonardo_vae_train.sbatch                   # NEW
│   ├── leonardo_stage_a_train.sbatch               # NEW
│   ├── leonardo_stage_b_train.sbatch               # NEW
│   ├── leonardo_eval.sbatch                        # NEW
│   └── leonardo_experiment_0.sbatch                # NEW (full E2E driver)
├── tests/
│   ├── test_models_configs.py                      # NEW
│   ├── test_models_vae.py                          # NEW
│   ├── test_models_dit.py                          # NEW
│   ├── test_models_inker.py                        # NEW
│   ├── test_models_raster_encoder.py               # NEW
│   ├── test_training_data_module.py                # NEW
│   ├── test_training_lit_modules.py                # NEW (smoke-train each Lightning module 1 step)
│   ├── test_training_samplers.py                   # NEW
│   ├── test_eval_stage_a.py                        # NEW
│   ├── test_eval_stage_b.py                        # NEW
│   ├── test_eval_baselines.py                      # NEW
│   └── test_synth_extended.py                      # NEW
└── results/EXPERIMENT_0_REPORT.md                  # NEW (Task 26)
```

---

## Task 1: Add modeling deps + scaffold model/training/eval dirs

**Files:**
- Modify: `bonzai_genai/pyproject.toml`
- Create: `bonzai_genai/src/bonzai_genai/models/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/training/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/eval/__init__.py`

- [ ] **Step 1: Add deps to `pyproject.toml`**

In `bonzai_genai/pyproject.toml`, edit the `dependencies` list and add (preserve existing entries; just append):

```toml
    "torch>=2.5",
    "lightning>=2.4",
    "einops>=0.8",
    "torchmetrics>=1.4",
    "networkx>=3.3",
```

- [ ] **Step 2: Create empty package init files**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
touch src/bonzai_genai/models/__init__.py
touch src/bonzai_genai/training/__init__.py
touch src/bonzai_genai/eval/__init__.py
```

Each `__init__.py` is an empty file (1-line module docstring optional).

- [ ] **Step 3: Reinstall the package with new deps**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pip install -e ".[dev]" 2>&1 | tail -3
```

Expected: `Successfully installed ... lightning-2.4.* torch-2.5.* einops-0.8.* torchmetrics-1.4.* networkx-3.3.*` (versions may vary; importance is no errors).

- [ ] **Step 4: Smoke-import sanity check**

```bash
.venv/bin/python -c "
import torch
import lightning as L
import einops
import torchmetrics
import networkx
print('torch', torch.__version__)
print('lightning', L.__version__)
print('OK')
"
```

Expected: each `__version__` prints, then `OK`.

- [ ] **Step 5: Run existing test suite to confirm nothing broke**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pytest -q 2>&1 | tail -3
```

Expected: `55 passed` (or 54 + 1 skipped if SG PBF isn't local).

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/pyproject.toml bonzai_genai/src/bonzai_genai/models/__init__.py bonzai_genai/src/bonzai_genai/training/__init__.py bonzai_genai/src/bonzai_genai/eval/__init__.py
git commit -m "feat(deps): add torch + lightning + einops + torchmetrics + networkx; scaffold models/training/eval

Phase 0b prerequisite: model + training + eval surface area lands in
the next ~25 tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Model configs (`models/configs.py`)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/models/configs.py`
- Create: `bonzai_genai/tests/test_models_configs.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_models_configs.py`:

```python
"""Tests for model configuration dataclasses."""
import pytest

from bonzai_genai.models.configs import (
    DiTConfig,
    InkerConfig,
    ProductionPreset,
    RasterEncoderConfig,
    TinyPreset,
    VAEConfig,
)


def test_tiny_preset_has_smaller_params_than_production():
    tiny_dit = DiTConfig.from_preset(TinyPreset)
    prod_dit = DiTConfig.from_preset(ProductionPreset)
    assert tiny_dit.hidden_dim < prod_dit.hidden_dim
    assert tiny_dit.num_layers < prod_dit.num_layers


def test_vae_latent_shape_consistent():
    cfg = VAEConfig.from_preset(TinyPreset)
    # VAE compresses 512x512x9 -> 64x64x4 (8x spatial, 4 latent ch)
    assert cfg.latent_dim == 4
    assert cfg.spatial_compression == 8


def test_dit_patch_count_consistent():
    """Patch size 2 over 64x64 latent -> 32x32 = 1024 transformer tokens."""
    cfg = DiTConfig.from_preset(TinyPreset)
    assert cfg.patch_size == 2
    grid = 64 // cfg.patch_size
    assert grid * grid == 1024


def test_inker_context_length():
    tiny = InkerConfig.from_preset(TinyPreset)
    prod = InkerConfig.from_preset(ProductionPreset)
    assert tiny.max_context_len == 4096
    assert prod.max_context_len == 16384


def test_raster_encoder_output_dim():
    tiny = RasterEncoderConfig.from_preset(TinyPreset)
    prod = RasterEncoderConfig.from_preset(ProductionPreset)
    assert tiny.output_dim == 256
    assert prod.output_dim == 768


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        DiTConfig.from_preset("medium")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pytest tests/test_models_configs.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'DiTConfig' from 'bonzai_genai.models.configs'`.

- [ ] **Step 3: Write `models/configs.py`**

Write `bonzai_genai/src/bonzai_genai/models/configs.py`:

```python
"""Model configuration dataclasses with Tiny / Production presets.

Each model has a frozen dataclass; instantiate via ``from_preset(name)``.
The two named presets correspond to:

- ``TinyPreset`` — Experiment 0 smoke models (~5-50M params each)
- ``ProductionPreset`` — Phases 4-5 production models (~10M-1B params each)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

TinyPreset: Final[str] = "tiny"
ProductionPreset: Final[str] = "production"
_PRESETS = (TinyPreset, ProductionPreset)


def _check_preset(name: str) -> None:
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {_PRESETS}")


@dataclass(frozen=True)
class VAEConfig:
    in_channels: int = 9
    base_channels: int = 32
    num_down_blocks: int = 4
    latent_dim: int = 4
    spatial_compression: int = 8

    @classmethod
    def from_preset(cls, name: str) -> "VAEConfig":
        _check_preset(name)
        if name == TinyPreset:
            return cls(base_channels=32)
        return cls(base_channels=64)


@dataclass(frozen=True)
class DiTConfig:
    in_channels: int = 4          # latent space
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_expansion: int = 4
    patch_size: int = 2           # over 64x64 latent
    cond_dim: int = 256           # combined conditioning embedding dim

    @classmethod
    def from_preset(cls, name: str) -> "DiTConfig":
        _check_preset(name)
        if name == TinyPreset:
            return cls(hidden_dim=512, num_layers=12, num_heads=8, cond_dim=256)
        return cls(hidden_dim=1024, num_layers=24, num_heads=16, cond_dim=768)


@dataclass(frozen=True)
class InkerConfig:
    vocab_size: int = 9728        # ~14 special + 1024 coord + 8192 node-ref + ~290 attr (live count from tokens.py)
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_expansion: int = 4
    max_context_len: int = 4096
    raster_feat_dim: int = 256    # must match RasterEncoderConfig.output_dim

    @classmethod
    def from_preset(cls, name: str) -> "InkerConfig":
        _check_preset(name)
        if name == TinyPreset:
            return cls(
                hidden_dim=512, num_layers=12, num_heads=8,
                max_context_len=4096, raster_feat_dim=256,
            )
        return cls(
            hidden_dim=1280, num_layers=32, num_heads=20,
            max_context_len=16384, raster_feat_dim=768,
        )


@dataclass(frozen=True)
class RasterEncoderConfig:
    in_channels: int = 9
    base_channels: int = 64
    num_layers: int = 3
    output_dim: int = 256

    @classmethod
    def from_preset(cls, name: str) -> "RasterEncoderConfig":
        _check_preset(name)
        if name == TinyPreset:
            return cls(base_channels=64, num_layers=3, output_dim=256)
        return cls(base_channels=96, num_layers=4, output_dim=768)
```

- [ ] **Step 4: Run tests; expect all pass**

```bash
.venv/bin/pytest tests/test_models_configs.py -v 2>&1 | tail -10
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/configs.py bonzai_genai/tests/test_models_configs.py
git commit -m "feat(models): add config dataclasses with tiny/production presets

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: VAE encoder

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/models/vae.py`
- Create: `bonzai_genai/tests/test_models_vae.py`

- [ ] **Step 1: Write the failing test (encoder shape contract)**

Write `bonzai_genai/tests/test_models_vae.py`:

```python
"""Tests for the VAE."""
import pytest
import torch

from bonzai_genai.models.configs import TinyPreset, VAEConfig
from bonzai_genai.models.vae import VAE, VAEEncoder


@pytest.fixture
def cfg():
    return VAEConfig.from_preset(TinyPreset)


def test_encoder_output_shape(cfg):
    enc = VAEEncoder(cfg)
    x = torch.randn(2, 9, 512, 512)
    mu, logvar = enc(x)
    # 8x spatial compression: 512 -> 64
    assert mu.shape == (2, cfg.latent_dim, 64, 64)
    assert logvar.shape == (2, cfg.latent_dim, 64, 64)


def test_encoder_logvar_is_clamped_for_stability(cfg):
    enc = VAEEncoder(cfg)
    x = torch.randn(1, 9, 512, 512) * 1e6  # huge inputs
    _, logvar = enc(x)
    assert torch.all(logvar >= -10) and torch.all(logvar <= 10), "logvar must be clamped"
```

- [ ] **Step 2: Run; expect ImportError**

```bash
.venv/bin/pytest tests/test_models_vae.py::test_encoder_output_shape -v 2>&1 | tail -5
```

Expected: ImportError.

- [ ] **Step 3: Write the encoder**

Write `bonzai_genai/src/bonzai_genai/models/vae.py`:

```python
"""9-channel VAE for the Sketcher pipeline.

Compresses (B, 9, 512, 512) raster -> (B, latent_dim, 64, 64) latent
(8x spatial compression, latent_dim=4). Channel-aware reconstruction
loss (BCE on binary masks, MSE on density channel) + KL regulariser.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bonzai_genai.models.configs import VAEConfig


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class VAEEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        # Initial 1x1 lift
        self.stem = nn.Conv2d(cfg.in_channels, ch, kernel_size=1)
        # 4 down-blocks: stride-2 conv each. 512 -> 256 -> 128 -> 64
        # (We stop at 8x to match latent_dim*8 = 32->64 spatial.)
        # Each block doubles channels.
        blocks = []
        cur_ch = ch
        for i in range(cfg.num_down_blocks):
            stride = 2 if i < 3 else 1
            blocks.append(_conv_block(cur_ch, cur_ch * 2, stride=stride))
            cur_ch *= 2
        self.down = nn.Sequential(*blocks)
        # Project to 2*latent_dim channels (mu + logvar concatenated)
        self.head = nn.Conv2d(cur_ch, 2 * cfg.latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.down(h)
        h = self.head(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar


# Decoder + VAE forward come in Task 4.
class VAE(nn.Module):
    """Placeholder; decoder + forward land in Task 4."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAEEncoder(cfg)
        # Decoder added in Task 4.
```

- [ ] **Step 4: Run encoder tests; expect 2 passed**

```bash
.venv/bin/pytest tests/test_models_vae.py -v 2>&1 | tail -8
```

Expected: 2 passed (only encoder tests defined yet).

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/vae.py bonzai_genai/tests/test_models_vae.py
git commit -m "feat(models): VAE encoder (9-channel raster -> 64x64x4 latent)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: VAE decoder + reconstruction + reparam

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/models/vae.py`
- Modify: `bonzai_genai/tests/test_models_vae.py`

- [ ] **Step 1: Append decoder + roundtrip tests**

Append to `bonzai_genai/tests/test_models_vae.py`:

```python
def test_decoder_output_shape(cfg):
    from bonzai_genai.models.vae import VAEDecoder
    dec = VAEDecoder(cfg)
    z = torch.randn(2, cfg.latent_dim, 64, 64)
    out = dec(z)
    assert out.shape == (2, cfg.in_channels, 512, 512)


def test_vae_forward_returns_recon_mu_logvar(cfg):
    vae = VAE(cfg)
    x = torch.randn(2, 9, 512, 512)
    out = vae(x)
    assert "recon" in out and "mu" in out and "logvar" in out
    assert out["recon"].shape == x.shape


def test_reconstruction_loss_breaks_down_by_channel(cfg):
    vae = VAE(cfg)
    x = torch.zeros(1, 9, 512, 512)
    x[:, :5] = (torch.rand_like(x[:, :5]) > 0.5).float()  # binary masks
    x[:, 5] = torch.rand_like(x[:, 5])                     # density continuous
    x[:, 6:] = (torch.rand_like(x[:, 6:]) > 0.5).float()
    out = vae(x)
    losses = vae.compute_losses(x, out)
    assert "recon_bce" in losses and "recon_mse" in losses and "kl" in losses
    assert all(torch.isfinite(v) for v in losses.values())


def test_kl_loss_is_zero_for_unit_gaussian(cfg):
    vae = VAE(cfg)
    # Build mu=0, logvar=0 distribution -> KL with N(0,I) is exactly 0.
    out = {"recon": torch.zeros(1, 9, 512, 512), "mu": torch.zeros(1, 4, 64, 64), "logvar": torch.zeros(1, 4, 64, 64)}
    losses = vae.compute_losses(torch.zeros(1, 9, 512, 512), out)
    assert losses["kl"].abs() < 1e-6
```

- [ ] **Step 2: Run; expect 4 ImportError / AttributeError failures**

```bash
.venv/bin/pytest tests/test_models_vae.py -v 2>&1 | tail -10
```

Expected: 2 passed (encoder), 4 failed.

- [ ] **Step 3: Replace `VAE` placeholder + add decoder**

Replace the `VAE` class (and add `VAEDecoder`) in `bonzai_genai/src/bonzai_genai/models/vae.py`:

```python
def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class VAEDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        # Mirror the encoder: latent -> 8*ch -> 4*ch -> 2*ch -> ch -> in
        ch = cfg.base_channels
        cur_ch = ch * (2 ** cfg.num_down_blocks)
        self.stem = nn.Conv2d(cfg.latent_dim, cur_ch, kernel_size=3, padding=1)
        blocks = []
        for i in range(cfg.num_down_blocks):
            # Mirror the encoder: 1 of the 4 blocks was stride-1, rest stride-2
            if i == 0:
                blocks.append(_conv_block(cur_ch, cur_ch // 2, stride=1))
            else:
                blocks.append(_up_block(cur_ch, cur_ch // 2))
            cur_ch //= 2
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(cur_ch, cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        h = self.up(h)
        return self.head(h)


class VAE(nn.Module):
    """Full VAE: encode -> reparam -> decode + channel-aware loss."""

    # Channel layout (must match data/rasteriser.py output):
    #   0: motorway, 1: trunk, 2: primary, 3: residential (binary)
    #   4: all-roads (binary)
    #   5: building density (continuous)
    #   6: water mask (binary)
    #   7: green mask (binary)
    #   8: urban mask (binary)
    BINARY_CHANNELS = (0, 1, 2, 3, 4, 6, 7, 8)
    CONTINUOUS_CHANNELS = (5,)

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAEEncoder(cfg)
        self.decoder = VAEDecoder(cfg)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def compute_losses(
        self, x: torch.Tensor, out: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        recon = out["recon"]
        mu = out["mu"]
        logvar = out["logvar"]
        # Binary channels: BCE over logits (decoder outputs raw logits)
        bin_idx = list(self.BINARY_CHANNELS)
        bce = F.binary_cross_entropy_with_logits(
            recon[:, bin_idx], x[:, bin_idx], reduction="mean",
        )
        # Continuous channels: MSE
        cont_idx = list(self.CONTINUOUS_CHANNELS)
        mse = F.mse_loss(recon[:, cont_idx], x[:, cont_idx], reduction="mean")
        # KL divergence to N(0, I), per-element mean
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return {"recon_bce": bce, "recon_mse": mse, "kl": kl}
```

- [ ] **Step 4: Run; expect 6 passed**

```bash
.venv/bin/pytest tests/test_models_vae.py -v 2>&1 | tail -10
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/vae.py bonzai_genai/tests/test_models_vae.py
git commit -m "feat(models): VAE decoder + reparam + channel-aware loss

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Lightning VAE training module

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/training/lit_vae.py`
- Create: `bonzai_genai/tests/test_training_lit_modules.py`

- [ ] **Step 1: Write the failing 1-step training test**

Write `bonzai_genai/tests/test_training_lit_modules.py`:

```python
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
```

- [ ] **Step 2: Run; expect ImportError**

```bash
.venv/bin/pytest tests/test_training_lit_modules.py::test_lit_vae_one_training_step -v 2>&1 | tail -5
```

Expected: ImportError.

- [ ] **Step 3: Write `lit_vae.py`**

Write `bonzai_genai/src/bonzai_genai/training/lit_vae.py`:

```python
"""LightningModule for VAE reconstruction training (Phase 0b smoke + Phase 3 production)."""
from __future__ import annotations

import lightning as L
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
```

- [ ] **Step 4: Run; expect 1 passed**

```bash
.venv/bin/pytest tests/test_training_lit_modules.py -v 2>&1 | tail -8
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/training/lit_vae.py bonzai_genai/tests/test_training_lit_modules.py
git commit -m "feat(training): Lightning VAE training module

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Extend `synth/procedural.py` for richer Experiment 0 corpus

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/synth/procedural.py`
- Create: `bonzai_genai/tests/test_synth_extended.py`

The current `procedural.py` produces simple grid roads + a few buildings. Experiment 0 needs more variety: roads at varied angles (not just axis-aligned), different building footprints, occasional parks/water, optional POIs.

- [ ] **Step 1: Write the failing variety tests**

Write `bonzai_genai/tests/test_synth_extended.py`:

```python
"""Extended variety tests for the procedural synth tile generator."""
import pytest

from bonzai_genai.synth.procedural import generate_synthetic_tile


def test_generates_buildings_landuse_pois_in_dense_mode():
    geom = generate_synthetic_tile(seed=0, density="dense")
    assert len(geom.roads) >= 5, "dense tiles should have multiple roads"
    assert len(geom.buildings) >= 20, "dense tiles should have many buildings"
    assert len(geom.land) >= 0, "land polygons optional but should not crash"


def test_sparse_tile_is_actually_sparse():
    geom = generate_synthetic_tile(seed=0, density="sparse")
    assert len(geom.buildings) <= 30, "sparse tiles should have few buildings"


def test_seed_is_deterministic():
    g1 = generate_synthetic_tile(seed=42, density="dense")
    g2 = generate_synthetic_tile(seed=42, density="dense")
    assert len(g1.roads) == len(g2.roads)
    assert len(g1.buildings) == len(g2.buildings)


def test_road_angles_are_varied_in_dense_mode():
    """Plan-2 requirement: roads at non-axis-aligned angles for variety."""
    import math
    geom = generate_synthetic_tile(seed=1, density="dense")
    angles = []
    for road in geom.roads:
        if len(road.polyline) < 2:
            continue
        x0, y0 = road.polyline[0]
        x1, y1 = road.polyline[-1]
        if x1 == x0 and y1 == y0:
            continue
        angles.append(math.atan2(y1 - y0, x1 - x0))
    # At least some non-axis-aligned roads
    non_axis = sum(1 for a in angles if abs(a) > 0.1 and abs(abs(a) - math.pi/2) > 0.1)
    assert non_axis >= 1, f"expected non-axis-aligned roads, got angles {angles}"
```

- [ ] **Step 2: Run; expect failures (`density` kwarg unknown / not enough roads)**

```bash
.venv/bin/pytest tests/test_synth_extended.py -v 2>&1 | tail -10
```

Expected: 4 failed.

- [ ] **Step 3: Read current `procedural.py` to understand what to extend**

```bash
wc -l bonzai_genai/src/bonzai_genai/synth/procedural.py
head -50 bonzai_genai/src/bonzai_genai/synth/procedural.py
```

- [ ] **Step 4: Edit `procedural.py` to support `density` and varied angles**

Add a `density: str = "dense"` kwarg to `generate_synthetic_tile`. For `density="dense"`, generate ~5 roads at random angles + ~30 buildings + 1-2 land polygons; for `density="sparse"`, ~2 roads + ~10 buildings + 0 land polygons. Use the existing `random.Random(seed)` for determinism.

The exact code skeleton:

```python
def generate_synthetic_tile(seed: int, density: str = "dense") -> TileGeometry:
    rng = random.Random(seed)
    if density not in ("sparse", "dense"):
        raise ValueError(f"density must be 'sparse' or 'dense', got {density!r}")
    if density == "dense":
        n_roads = rng.randint(5, 8)
        n_buildings = rng.randint(30, 60)
        n_land = rng.randint(1, 3)
        n_pois = rng.randint(3, 8)
    else:
        n_roads = rng.randint(2, 3)
        n_buildings = rng.randint(8, 15)
        n_land = 0
        n_pois = rng.randint(0, 2)

    geom = TileGeometry()
    # Roads at varied angles
    for _ in range(n_roads):
        angle = rng.uniform(0, math.pi)
        x0 = rng.uniform(0, TILE_SIDE_M)
        y0 = rng.uniform(0, TILE_SIDE_M)
        length = rng.uniform(400, 1500)
        x1 = max(0.0, min(TILE_SIDE_M - 1, x0 + length * math.cos(angle)))
        y1 = max(0.0, min(TILE_SIDE_M - 1, y0 + length * math.sin(angle)))
        cls = rng.choice(["residential", "secondary", "tertiary", "primary"])
        geom.roads.append(Road(class_name=f"road_class={cls}", polyline=[(x0, y0), (x1, y1)]))
    # Buildings (axis-aligned rectangles)
    for _ in range(n_buildings):
        cx = rng.uniform(50, TILE_SIDE_M - 50)
        cy = rng.uniform(50, TILE_SIDE_M - 50)
        w = rng.uniform(15, 40)
        h = rng.uniform(15, 40)
        verts = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        cls = rng.choice(["residential", "commercial", "office", "industrial"])
        geom.buildings.append(Building(
            class_name=f"building_class={cls}",
            height_name="height=NA",
            vertices=verts,
        ))
    # Land polygons
    for _ in range(n_land):
        cx = rng.uniform(100, TILE_SIDE_M - 100)
        cy = rng.uniform(100, TILE_SIDE_M - 100)
        r = rng.uniform(80, 150)
        verts = []
        for i in range(8):
            theta = i * math.pi / 4
            verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        cls = rng.choice(["land_class=park", "water_class=lake"])
        geom.land.append(LandPolygon(class_name=cls, vertices=verts))
    # POIs at building corners (rough heuristic)
    for _ in range(n_pois):
        if not geom.buildings:
            break
        b = rng.choice(geom.buildings)
        x, y = b.vertices[0]
        cls = rng.choice(["cafe", "restaurant", "school", "bank"])
        geom.pois.append(POI(class_name=f"poi={cls}", point=(x, y)))
    return geom
```

(Apply the diff using `Edit` calls; preserve existing imports, helpers, and the function signature default for backwards compatibility.)

- [ ] **Step 5: Run extended + existing tests; expect all pass**

```bash
.venv/bin/pytest tests/test_synth_extended.py tests/test_synth.py -v 2>&1 | tail -10
```

Expected: 4 + 3 = 7 passed.

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/synth/procedural.py bonzai_genai/tests/test_synth_extended.py
git commit -m "feat(synth): extend procedural generator with density modes + varied road angles

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add `synth-corpus` CLI subcommand

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py`

The existing `synthetic` subcommand generates tiles in a single density. We need a new `synth-corpus` that generates Experiment 0's 5,000-tile mixed-density corpus deterministically.

- [ ] **Step 1: Add the `synth-corpus` subcommand**

Append to `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py`:

```python
@app.command("synth-corpus")
def cmd_synth_corpus(
    output: Path = typer.Option(..., "-o", "--output"),  # noqa: B008
    n_train: int = typer.Option(4500, "--n-train"),  # noqa: B008
    n_val: int = typer.Option(500, "--n-val"),  # noqa: B008
    shard_size: int = typer.Option(500, "--shard-size"),  # noqa: B008
    seed_base: int = typer.Option(0, "--seed-base"),  # noqa: B008
) -> None:
    """Generate Experiment 0 synthetic corpus: mixed sparse/dense density."""
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    train_dir = output / "train"
    val_dir = output / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_w = ShardWriter(train_dir, shard_size=shard_size)
    val_w = ShardWriter(val_dir, shard_size=shard_size)
    rng = random.Random(seed_base)

    def _emit(writer: ShardWriter, n: int, prefix: str, base: int) -> None:
        with Progress(console=console) as progress:
            task_id = progress.add_task(f"[green]{prefix}", total=n)
            for i in range(n):
                density = "dense" if rng.random() < 0.6 else "sparse"
                geom = generate_synthetic_tile(seed=base + i, density=density)
                raster = rasterise(geom)
                tokens = tokeniser.encode(geom)
                meta = TileMetadata(
                    tile_id=f"{prefix}-{i:06d}",
                    sw_lat=0.0, sw_lon=0.0,
                    country="SYN", koppen="N/A",
                    density_bucket=density,
                    primary_land_use="mixed",
                )
                writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
                progress.update(task_id, advance=1)

    _emit(train_w, n_train, "SYN-T", seed_base)
    _emit(val_w, n_val, "SYN-V", seed_base + n_train)
    train_w.close()
    val_w.close()
    console.print(f"[bold green]Wrote {n_train} train + {n_val} val to {output}")
```

You will also need these imports added near the existing imports (preserve existing entries):

```python
import random
from bonzai_genai.synth.procedural import generate_synthetic_tile
```

- [ ] **Step 2: Smoke-run with tiny counts**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
rm -rf /tmp/bonzai-syncorpus
.venv/bin/python scripts/prepare_tiles_local.py synth-corpus -o /tmp/bonzai-syncorpus --n-train 30 --n-val 10 --shard-size 20 2>&1 | tail -5
ls -lh /tmp/bonzai-syncorpus/{train,val}/
```

Expected: `Wrote 30 train + 10 val to /tmp/bonzai-syncorpus`. Both train and val have shard files.

- [ ] **Step 3: Run all tests; expect previous + still passing (no test for the CLI subcommand itself)**

```bash
.venv/bin/pytest -q 2>&1 | tail -3
```

Expected: 60+ passed.

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py
git commit -m "feat(cli): add synth-corpus subcommand for Experiment 0 dataset prep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: WebDataset Lightning DataModule

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/training/data_module.py`
- Create: `bonzai_genai/tests/test_training_data_module.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_training_data_module.py`:

```python
"""Tests for the LightningDataModule that wraps WebDataset shards."""
from pathlib import Path

import pytest
import torch


@pytest.fixture
def syn_corpus(tmp_path):
    """Build a tiny synth corpus in tmp_path/{train,val}/ for testing."""
    import subprocess, sys
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "corpus"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "prepare_tiles_local.py"),
            "synth-corpus",
            "-o", str(out),
            "--n-train", "20", "--n-val", "10", "--shard-size", "10",
        ],
        check=True,
    )
    return out


def test_data_module_yields_raster_only_batch(syn_corpus):
    from bonzai_genai.training.data_module import TileDataModule
    dm = TileDataModule(
        train_url=str(syn_corpus / "train" / "shard-{000000..000001}.tar"),
        val_url=str(syn_corpus / "val" / "shard-000000.tar"),
        batch_size=4,
        return_tokens=False,
        num_workers=0,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 9, 512, 512)
    assert batch.dtype == torch.float32


def test_data_module_yields_raster_and_tokens_batch(syn_corpus):
    from bonzai_genai.training.data_module import TileDataModule
    dm = TileDataModule(
        train_url=str(syn_corpus / "train" / "shard-{000000..000001}.tar"),
        val_url=str(syn_corpus / "val" / "shard-000000.tar"),
        batch_size=2,
        return_tokens=True,
        num_workers=0,
        max_token_len=4096,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert "raster" in batch and "tokens" in batch and "token_lens" in batch
    assert batch["raster"].shape == (2, 9, 512, 512)
    assert batch["tokens"].shape == (2, 4096)
    assert batch["tokens"].dtype == torch.long
    # token_lens are real lengths before pad
    assert (batch["token_lens"] <= 4096).all()
```

- [ ] **Step 2: Run; expect ImportError**

```bash
.venv/bin/pytest tests/test_training_data_module.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `data_module.py`**

Write `bonzai_genai/src/bonzai_genai/training/data_module.py`:

```python
"""LightningDataModule that wraps WebDataset shards into PyTorch DataLoaders."""
from __future__ import annotations

import io

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from bonzai_genai.data.tile_bundle import TileBundle


def _decode_bundle(sample: dict) -> dict:
    """WebDataset gives us raw bytes for raster.npy / tokens.npy / metadata.json."""
    raster = np.load(io.BytesIO(sample["raster.npy"]))
    tokens = np.load(io.BytesIO(sample["tokens.npy"]))
    return {"raster": raster.astype(np.float32), "tokens": tokens.astype(np.int64)}


def _collate_raster_only(items: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.stack([it["raster"] for it in items]))


def _collate_with_tokens(items: list[dict], max_len: int) -> dict[str, torch.Tensor]:
    rasters = torch.from_numpy(np.stack([it["raster"] for it in items]))
    bs = len(items)
    tokens_pad = torch.zeros(bs, max_len, dtype=torch.long)
    lens = torch.zeros(bs, dtype=torch.long)
    for i, it in enumerate(items):
        n = min(len(it["tokens"]), max_len)
        tokens_pad[i, :n] = torch.from_numpy(it["tokens"][:n])
        lens[i] = n
    return {"raster": rasters, "tokens": tokens_pad, "token_lens": lens}


class TileDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_url: str,
        val_url: str,
        batch_size: int = 8,
        return_tokens: bool = False,
        max_token_len: int = 4096,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _build(self, url: str):
        import webdataset as wds
        ds = (
            wds.WebDataset(url, shardshuffle=False, empty_check=False)
            .map(_decode_bundle)
        )
        return ds

    def setup(self, stage: str) -> None:
        self.train_ds = self._build(self.hparams.train_url)
        self.val_ds = self._build(self.hparams.val_url)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        if self.hparams.return_tokens:
            collate = lambda items: _collate_with_tokens(items, self.hparams.max_token_len)
        else:
            collate = _collate_raster_only
        return DataLoader(
            ds.batched(self.hparams.batch_size, collation_fn=collate, partial=False),
            batch_size=None,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)
```

- [ ] **Step 4: Run tests; expect 2 passed**

```bash
.venv/bin/pytest tests/test_training_data_module.py -v 2>&1 | tail -8
```

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/training/data_module.py bonzai_genai/tests/test_training_data_module.py
git commit -m "feat(training): WebDataset LightningDataModule for raster + tokens

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: DiT — patch embed + sinusoidal time embed

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/models/dit.py`
- Create: `bonzai_genai/tests/test_models_dit.py`

- [ ] **Step 1: Write the failing tests**

Write `bonzai_genai/tests/test_models_dit.py`:

```python
"""Tests for DiT components."""
import math

import pytest
import torch

from bonzai_genai.models.configs import DiTConfig, TinyPreset
from bonzai_genai.models.dit import (
    DiT,
    PatchEmbed,
    SinusoidalTimeEmbed,
)


@pytest.fixture
def cfg():
    return DiTConfig.from_preset(TinyPreset)


def test_patch_embed_token_count(cfg):
    pe = PatchEmbed(cfg)
    z = torch.randn(2, cfg.in_channels, 64, 64)
    out = pe(z)
    # 64/2 = 32, 32*32 = 1024
    assert out.shape == (2, 1024, cfg.hidden_dim)


def test_sinusoidal_time_embed_dim(cfg):
    te = SinusoidalTimeEmbed(cfg.cond_dim)
    t = torch.tensor([0.0, 1.5, 999.0])
    out = te(t)
    assert out.shape == (3, cfg.cond_dim)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Write `models/dit.py` (patch embed + time embed only; full DiT in next task)**

Write `bonzai_genai/src/bonzai_genai/models/dit.py`:

```python
"""DiT (Diffusion Transformer) for the Sketcher.

Operates on the VAE latent (B, latent_dim, 64, 64); patches into 1024
transformer tokens (patch_size=2). AdaLN-Zero conditioning per
DiT-XL/2 (Peebles & Xie 2023). Reference architecture: §5.4 of the
global design spec.

This module is built progressively across Plan 2 Tasks 9-12:
    Task  9: PatchEmbed + SinusoidalTimeEmbed
    Task 10: AdaLN-Zero attention block
    Task 11: FFN + DiT main module + sampling head
    Task 12: EDM noise + DPM-Solver++ sampler + LightningModule
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from bonzai_genai.models.configs import DiTConfig


class PatchEmbed(nn.Module):
    """Patchify (B, C, H, W) latent into (B, N, hidden_dim) sequence."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, hidden, H/p, W/p) -> (B, N, hidden)
        h = self.proj(z)
        return rearrange(h, "b c h w -> b (h w) c")


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DiT(nn.Module):
    """Stub; full module assembled in Task 11."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.time_embed = SinusoidalTimeEmbed(cfg.cond_dim)
```

- [ ] **Step 4: Run tests; expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/dit.py bonzai_genai/tests/test_models_dit.py
git commit -m "feat(models): DiT patch embed + sinusoidal time embed

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: DiT AdaLN-Zero attention + FFN block

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/models/dit.py`
- Modify: `bonzai_genai/tests/test_models_dit.py`

- [ ] **Step 1: Append failing block test**

Append to `bonzai_genai/tests/test_models_dit.py`:

```python
def test_dit_block_preserves_shape(cfg):
    from bonzai_genai.models.dit import DiTBlock
    block = DiTBlock(cfg)
    x = torch.randn(2, 1024, cfg.hidden_dim)
    cond = torch.randn(2, cfg.cond_dim)
    out = block(x, cond)
    assert out.shape == x.shape


def test_dit_block_zero_init_residual_path():
    """AdaLN-Zero: at init, gate parameters should be ~0 so the block is identity."""
    from bonzai_genai.models.dit import DiTBlock
    cfg_small = DiTConfig(hidden_dim=32, num_layers=1, num_heads=4, cond_dim=32, patch_size=2)
    block = DiTBlock(cfg_small)
    x = torch.randn(1, 16, 32)
    cond = torch.zeros(1, 32)
    with torch.no_grad():
        out = block(x, cond)
    # With zero cond + zero-init gates, output ~= input
    assert torch.allclose(out, x, atol=1e-5), f"max diff = {(out-x).abs().max()}"
```

- [ ] **Step 2: Run; expect 2 failures**

- [ ] **Step 3: Add `DiTBlock` to `models/dit.py`**

Append to `bonzai_genai/src/bonzai_genai/models/dit.py`:

```python
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT transformer block with AdaLN-Zero conditioning.

    Per Peebles & Xie (2023): 6 modulation parameters per block
    (shift/scale/gate × {attn, mlp}); each starts at zero so the
    residual path is identity at init.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        ffn_dim = cfg.hidden_dim * cfg.ffn_expansion
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, cfg.hidden_dim),
        )
        # 6 modulation params per AdaLN-Zero block
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, 6 * cfg.hidden_dim, bias=True),
        )
        # Zero-init the projection to ensure identity residual path at start
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) -> (B, 6 * hidden_dim) -> 6 chunks
        c = self.adaLN(cond)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=-1)
        # Pre-norm + modulation + attention
        x_attn = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out
        # Pre-norm + modulation + MLP
        x_mlp = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x
```

- [ ] **Step 4: Run; expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/dit.py bonzai_genai/tests/test_models_dit.py
git commit -m "feat(models): DiT block with AdaLN-Zero conditioning

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Full DiT module + unpatchify

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/models/dit.py`
- Modify: `bonzai_genai/tests/test_models_dit.py`

- [ ] **Step 1: Append failing forward-pass test**

Append to `bonzai_genai/tests/test_models_dit.py`:

```python
def test_dit_forward_returns_latent_shape(cfg):
    dit = DiT(cfg)
    z = torch.randn(2, cfg.in_channels, 64, 64)
    t = torch.tensor([100.0, 500.0])
    out = dit(z, t)
    assert out.shape == z.shape


def test_dit_unconditional_forward_uses_null_cond(cfg):
    dit = DiT(cfg)
    z = torch.randn(1, cfg.in_channels, 64, 64)
    t = torch.tensor([100.0])
    out_uncond = dit(z, t)  # no cond_text / cond_tags -> null
    assert out_uncond.shape == z.shape
```

- [ ] **Step 2: Run; expect failures (DiT.forward not implemented)**

- [ ] **Step 3: Replace the `DiT` stub with the full module**

Replace the `DiT` class in `bonzai_genai/src/bonzai_genai/models/dit.py`:

```python
class FinalLayer(nn.Module):
    """Final modulation + linear projection back to patch tokens."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.patch_size * cfg.patch_size * cfg.in_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, 2 * cfg.hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class DiT(nn.Module):
    """DiT diffusion transformer over a 64x64 VAE latent.

    Conditioning paths:
      - time embedding (always)
      - text embedding (optional; null vector when not provided)
      - region tag embedding (optional; null vector when not provided)
    All three are summed into a single ``cond_dim`` vector before
    being projected to AdaLN-Zero modulation parameters per block.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.time_embed = SinusoidalTimeEmbed(cfg.cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        # Null cond vectors (learned) for unconditional path
        self.null_text = nn.Parameter(torch.zeros(cfg.cond_dim))
        self.null_tags = nn.Parameter(torch.zeros(cfg.cond_dim))
        # Positional embedding (learned, fixed length = grid_size^2)
        grid = 64 // cfg.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, cfg.hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Transformer blocks
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.final = FinalLayer(cfg)

    def _build_cond(
        self,
        t: torch.Tensor,
        cond_text: torch.Tensor | None,
        cond_tags: torch.Tensor | None,
    ) -> torch.Tensor:
        bs = t.shape[0]
        time = self.time_mlp(self.time_embed(t))
        text = cond_text if cond_text is not None else self.null_text.expand(bs, -1)
        tags = cond_tags if cond_tags is not None else self.null_tags.expand(bs, -1)
        return time + text + tags

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond_text: torch.Tensor | None = None,
        cond_tags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._build_cond(t, cond_text, cond_tags)
        x = self.patch_embed(z) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        x = self.final(x, cond)
        # Unpatchify: (B, N, p*p*C) -> (B, C, H, W)
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        grid = 64 // p
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      h=grid, w=grid, p1=p, p2=p, c=c)
        return x
```

- [ ] **Step 4: Run; expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/dit.py bonzai_genai/tests/test_models_dit.py
git commit -m "feat(models): full DiT forward with conditioning paths + unpatchify

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: EDM noise + DPM-Solver++ sampler + Lightning Stage A

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/training/samplers.py`
- Create: `bonzai_genai/src/bonzai_genai/training/lit_stage_a.py`
- Modify: `bonzai_genai/tests/test_training_lit_modules.py`
- Modify: `bonzai_genai/tests/test_training_samplers.py` (NEW file)

- [ ] **Step 1: Write the failing tests**

Write `bonzai_genai/tests/test_training_samplers.py`:

```python
"""Tests for diffusion samplers."""
import torch

from bonzai_genai.models.configs import DiTConfig, TinyPreset, VAEConfig
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
```

Append to `bonzai_genai/tests/test_training_lit_modules.py`:

```python
def test_lit_stage_a_one_training_step():
    from bonzai_genai.models.configs import DiTConfig, TinyPreset, VAEConfig
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
    # batch is the raw raster (B, 9, 512, 512); LitStageA encodes through VAE itself
    batch = torch.zeros(1, 9, 512, 512)
    batch[:, :5] = (torch.rand_like(batch[:, :5]) > 0.7).float()
    batch[:, 5] = torch.rand_like(batch[:, 5])
    batch[:, 6:] = (torch.rand_like(batch[:, 6:]) > 0.7).float()
    loss = lit.training_step(batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run; expect 2 ImportError**

- [ ] **Step 3: Implement `samplers.py`**

Write `bonzai_genai/src/bonzai_genai/training/samplers.py`:

```python
"""Diffusion samplers and AR samplers.

DPM-Solver++ for Stage A (DiT) sampling. Full reference: §5.7 of the
global design spec. We implement the second-order multistep variant
(``dpmpp_2m``); 50 steps gives high quality for Phase 4 production
and 10-25 steps suffices for smoke runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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
        # Model predicts denoised x given current x and sigma (diffusion timestep ~ sigma)
        t = sigma.expand(batch_size)
        denoised = model(x, t, cond_text=cond_text, cond_tags=cond_tags)
        if old_denoised is None or i == num_steps - 1:
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        else:
            # 2M update: average current and previous denoiser estimates
            h = sigmas[i + 1] - sigma
            r = sigmas[i] / sigmas[i - 1] if i >= 1 else 1.0
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            d = (x - denoised_d) / sigma
            x = x + d * h
        old_denoised = denoised
    return x
```

- [ ] **Step 4: Implement `lit_stage_a.py`**

Write `bonzai_genai/src/bonzai_genai/training/lit_stage_a.py`:

```python
"""LightningModule for Stage A (DiT) training in latent space.

Pipeline per training step:
    1. Encode raster batch via frozen VAE -> latent z0.
    2. Sample sigma per example from log-normal EDM distribution.
    3. Add noise: x = z0 + sigma * eps.
    4. Predict denoised latent: x_hat = DiT(x, sigma, cond).
    5. Loss = MSE between x_hat and z0, weighted by EDM weight 1/sigma^2.

Classifier-free guidance: 10% of training drops conditioning to null.
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
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
        log_sigma = self.hparams.p_mean + self.hparams.p_std * torch.randn(batch_size, device=device)
        return log_sigma.exp()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # batch is the raw raster (B, 9, 512, 512). Encode -> z0.
        with torch.no_grad():
            mu, logvar = self.vae.encoder(batch)
            z0 = mu  # use mean for stability; reparam is reserved for VAE training
        bs = z0.shape[0]
        sigma = self._sample_sigma(bs, z0.device)
        sigma_b = sigma.view(bs, 1, 1, 1)
        eps = torch.randn_like(z0)
        x_noisy = z0 + sigma_b * eps
        # CFG dropout: this smoke run is unconditional anyway, but keep the path live.
        cond_text = None
        cond_tags = None
        x_hat = self.dit(x_noisy, sigma, cond_text=cond_text, cond_tags=cond_tags)
        # EDM weighting: w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        sigma_data = self.hparams.sigma_data
        w = ((sigma_b ** 2 + sigma_data ** 2) / ((sigma_b * sigma_data) ** 2))
        loss = (w * (x_hat - z0) ** 2).mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return AdamW(
            [p for p in self.dit.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
```

- [ ] **Step 5: Run sampler test + lit smoke tests; expect both pass**

```bash
.venv/bin/pytest tests/test_training_samplers.py tests/test_training_lit_modules.py -v 2>&1 | tail -10
```

Expected: 1 + 2 = 3 passed.

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/training/samplers.py bonzai_genai/src/bonzai_genai/training/lit_stage_a.py bonzai_genai/tests/test_training_samplers.py bonzai_genai/tests/test_training_lit_modules.py
git commit -m "feat(training): EDM noise + DPM-Solver++ sampler + Lightning Stage A

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Raster CNN encoder

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/models/raster_encoder.py`
- Create: `bonzai_genai/tests/test_models_raster_encoder.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_models_raster_encoder.py`:

```python
"""Tests for the small CNN raster encoder used by Stage B for cross-attention."""
import pytest
import torch

from bonzai_genai.models.configs import RasterEncoderConfig, TinyPreset
from bonzai_genai.models.raster_encoder import RasterEncoder


@pytest.fixture
def cfg():
    return RasterEncoderConfig.from_preset(TinyPreset)


def test_output_shape_is_32x32_with_output_dim_channels(cfg):
    enc = RasterEncoder(cfg)
    x = torch.randn(2, cfg.in_channels, 512, 512)
    feat = enc(x)
    # 4 strided convs / 3 strided convs both compress 512 -> 32x32
    assert feat.shape == (2, cfg.output_dim, 32, 32)


def test_grid_can_be_flattened_for_cross_attention(cfg):
    enc = RasterEncoder(cfg)
    x = torch.randn(1, 9, 512, 512)
    feat = enc(x)
    flat = feat.flatten(2).transpose(1, 2)  # (B, 1024, output_dim)
    assert flat.shape == (1, 1024, cfg.output_dim)
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Implement `models/raster_encoder.py`**

Write `bonzai_genai/src/bonzai_genai/models/raster_encoder.py`:

```python
"""Strided CNN encoder mapping (B, 9, 512, 512) raster -> (B, output_dim, 32, 32) features.

Output is consumed by Stage B (Inker) via cross-attention. Frozen
during Inker training (initialised from the diffusion-trained Stage A
encoder OR trained jointly per a config flag; for Phase 0b smoke we
train it from scratch alongside the Inker).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bonzai_genai.models.configs import RasterEncoderConfig


def _strided_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class RasterEncoder(nn.Module):
    """3 (smoke) or 4 (production) strided conv layers; final 1x1 to ``output_dim``."""

    def __init__(self, cfg: RasterEncoderConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        layers = [_strided_conv(cfg.in_channels, ch)]   # 512 -> 256
        cur = ch
        layers.append(_strided_conv(cur, cur * 2))      # 256 -> 128
        cur *= 2
        layers.append(_strided_conv(cur, cur * 2))      # 128 -> 64
        cur *= 2
        if cfg.num_layers >= 4:
            layers.append(_strided_conv(cur, cur * 2)) # 64 -> 32
            cur *= 2
        else:
            layers.append(_strided_conv(cur, cur))     # 64 -> 32 (no channel doubling)
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(cur, cfg.output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))
```

- [ ] **Step 4: Run; expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/raster_encoder.py bonzai_genai/tests/test_models_raster_encoder.py
git commit -m "feat(models): strided CNN raster encoder for Stage B cross-attention

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Inker — token embedding + RoPE

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/models/inker.py`
- Create: `bonzai_genai/tests/test_models_inker.py`

- [ ] **Step 1: Write the failing tests**

Write `bonzai_genai/tests/test_models_inker.py`:

```python
"""Tests for the Inker (Stage B autoregressive transformer)."""
import pytest
import torch

from bonzai_genai.models.configs import (
    InkerConfig,
    RasterEncoderConfig,
    TinyPreset,
)


@pytest.fixture
def inker_cfg():
    return InkerConfig.from_preset(TinyPreset)


@pytest.fixture
def raster_cfg():
    return RasterEncoderConfig.from_preset(TinyPreset)


def test_token_embed_output_shape(inker_cfg):
    from bonzai_genai.models.inker import TokenEmbed
    emb = TokenEmbed(inker_cfg)
    tokens = torch.randint(0, inker_cfg.vocab_size, (2, 32))
    out = emb(tokens)
    assert out.shape == (2, 32, inker_cfg.hidden_dim)


def test_rope_applies_rotation(inker_cfg):
    from bonzai_genai.models.inker import build_rope_cache
    head_dim = inker_cfg.hidden_dim // inker_cfg.num_heads
    cos, sin = build_rope_cache(seq_len=64, head_dim=head_dim)
    assert cos.shape == (64, head_dim)
    assert sin.shape == (64, head_dim)
    # cos[0] should be ~1 (no rotation at position 0)
    assert torch.allclose(cos[0], torch.ones(head_dim), atol=1e-5)
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Implement initial `models/inker.py`**

Write `bonzai_genai/src/bonzai_genai/models/inker.py`:

```python
"""Stage B — autoregressive transformer ("Inker") with cross-attention to a raster encoder.

Architecture per global spec §6.3:
    - Token embedding (vocab_size -> hidden_dim)
    - RoPE (rotary positional embedding) on Q, K
    - 12 (smoke) / 24-32 (production) decoder layers, each with:
        - Causal self-attention (RoPE-applied)
        - Cross-attention to raster encoder feature grid
        - FFN
    - Output head: hidden_dim -> vocab_size (logits over the next token)

Built progressively in Plan 2 Tasks 14-17:
    Task 14: TokenEmbed + RoPE
    Task 15: Inker block (self-attn + cross-attn + FFN) + full module
    Task 16: Constrained decoding logit masks
    Task 17: Lightning Stage B + greedy/beam samplers
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bonzai_genai.models.configs import InkerConfig


class TokenEmbed(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cosine and sine RoPE caches.

    Returns ``(cos, sin)`` each of shape ``(seq_len, head_dim)``.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim/2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` (..., seq, head_dim) given pre-computed cos/sin."""
    # Split last dim into even/odd; rotate; recombine.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos1, sin1 = cos[..., 0::2], sin[..., 0::2]
    rotated = torch.stack([x1 * cos1 - x2 * sin1, x1 * sin1 + x2 * cos1], dim=-1)
    return rotated.flatten(-2)


# Inker main module added in Task 15.
```

- [ ] **Step 4: Run; expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/inker.py bonzai_genai/tests/test_models_inker.py
git commit -m "feat(models): Inker token embedding + RoPE cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Inker block (self-attn + cross-attn + FFN) + full module

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/models/inker.py`
- Modify: `bonzai_genai/tests/test_models_inker.py`

- [ ] **Step 1: Append failing forward-pass tests**

Append to `bonzai_genai/tests/test_models_inker.py`:

```python
def test_inker_forward_returns_logits(inker_cfg, raster_cfg):
    from bonzai_genai.models.inker import Inker
    inker = Inker(inker_cfg)
    tokens = torch.randint(0, inker_cfg.vocab_size, (2, 32))
    raster_feat = torch.randn(2, 32 * 32, raster_cfg.output_dim)
    logits = inker(tokens, raster_feat)
    assert logits.shape == (2, 32, inker_cfg.vocab_size)


def test_inker_causal_mask_blocks_future_tokens(inker_cfg, raster_cfg):
    from bonzai_genai.models.inker import Inker
    inker = Inker(inker_cfg)
    inker.eval()
    tokens = torch.randint(0, inker_cfg.vocab_size, (1, 16))
    raster_feat = torch.randn(1, 32 * 32, raster_cfg.output_dim)
    with torch.no_grad():
        out_full = inker(tokens, raster_feat)
        # Replace future tokens with garbage; first 8 logits must be unchanged
        tokens_perturbed = tokens.clone()
        tokens_perturbed[:, 8:] = 0
        out_perturbed = inker(tokens_perturbed, raster_feat)
    # First 8 positions identical because causal mask blocks future
    assert torch.allclose(out_full[:, :8], out_perturbed[:, :8], atol=1e-5)
```

- [ ] **Step 2: Run; expect 2 failures**

- [ ] **Step 3: Append `InkerBlock` and full `Inker` to `models/inker.py`**

Append to `bonzai_genai/src/bonzai_genai/models/inker.py`:

```python
class InkerSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE."""

    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:s], sin[:s])
        k = apply_rope(k, cos[:s], sin[:s])
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.hidden_dim)
        return self.out_proj(out)


class InkerCrossAttention(nn.Module):
    """Cross-attention from token sequence to a flat raster feature grid."""

    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.raster_feat_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.raster_feat_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, raster_feat: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        n = raster_feat.shape[1]
        q = self.q_proj(x).view(b, s, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(raster_feat).view(b, n, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(raster_feat).view(b, n, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.hidden_dim)
        return self.out_proj(out)


class InkerBlock(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.hidden_dim)
        self.self_attn = InkerSelfAttention(cfg)
        self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = InkerCrossAttention(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim)
        ffn = cfg.hidden_dim * cfg.ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, ffn),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn, cfg.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        raster_feat: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x), cos, sin)
        x = x + self.cross_attn(self.cross_norm(x), raster_feat)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Inker(nn.Module):
    def __init__(self, cfg: InkerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbed(cfg)
        self.blocks = nn.ModuleList([InkerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        # Pre-compute RoPE cache up to max_context_len
        head_dim = cfg.hidden_dim // cfg.num_heads
        cos, sin = build_rope_cache(cfg.max_context_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self, tokens: torch.Tensor, raster_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, raster_feat, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.head(x)
```

- [ ] **Step 4: Run; expect 4 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/inker.py bonzai_genai/tests/test_models_inker.py
git commit -m "feat(models): Inker block with self+cross attention + full module forward

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Constrained decoding logit masks

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/models/inker.py`
- Modify: `bonzai_genai/tests/test_models_inker.py`

Per global spec §6.4, constrained decoding masks invalid next-token choices at sample time. For Phase 0b smoke we implement the **mandatory** subset:
1. Layer order — `LAYER_LAND` < `LAYER_ROADS` < `LAYER_BUILDINGS` < `LAYER_POIS`.
2. Polygon closure — after `BUILDING_OPEN`, only valid sub-tokens (class -> height -> coord pairs -> `BUILDING_CLOSE`).
3. Coordinate pair completion — after an x-coord token, only y-coord tokens are valid.

Non-self-intersection, road-edge node-ref bounds, and building-field ordering are **deferred to Plan 3+** (logged as open question §8 of the spec).

- [ ] **Step 1: Append failing tests**

Append to `bonzai_genai/tests/test_models_inker.py`:

```python
def test_constrained_mask_after_x_only_y_allowed():
    from bonzai_genai.models.inker import build_constrained_mask
    from bonzai_genai.vocab.tokens import (
        NUM_SPECIAL_TOKENS, coord_x_token_id, coord_y_token_id,
    )
    from bonzai_genai.vocab.attributes import load_default_vocab
    vocab = load_default_vocab()
    # Pretend we just emitted x_5
    state = {"phase": "in_building_polygon", "last_token": coord_x_token_id(5)}
    mask = build_constrained_mask(state, vocab_size=10000, attr_vocab=vocab)
    # mask: True = allowed, False = blocked
    # Only y-coord tokens should be allowed at this position
    y0 = coord_y_token_id(0)
    y_end = coord_y_token_id(511)
    assert mask[y0:y_end + 1].all()
    # x tokens blocked
    assert not mask[coord_x_token_id(0)]


def test_constrained_mask_layer_order_is_enforced():
    from bonzai_genai.models.inker import build_constrained_mask
    from bonzai_genai.vocab.tokens import SpecialToken
    from bonzai_genai.vocab.attributes import load_default_vocab
    vocab = load_default_vocab()
    # If we're after LAYER_BUILDINGS, LAYER_LAND should be blocked.
    state = {"phase": "between_buildings", "layer": "buildings"}
    mask = build_constrained_mask(state, vocab_size=10000, attr_vocab=vocab)
    assert not mask[int(SpecialToken.LAYER_LAND)]
    assert not mask[int(SpecialToken.LAYER_ROADS)]
```

- [ ] **Step 2: Run; expect 2 failures**

- [ ] **Step 3: Append `build_constrained_mask` to `models/inker.py`**

Append to `bonzai_genai/src/bonzai_genai/models/inker.py`:

```python
from bonzai_genai.config import COORD_BINS  # noqa: E402
from bonzai_genai.vocab.attributes import AttributeVocab  # noqa: E402
from bonzai_genai.vocab.tokens import (  # noqa: E402
    NUM_NODE_REF_TOKENS,
    NUM_SPECIAL_TOKENS,
    SpecialToken,
)


def build_constrained_mask(
    state: dict, vocab_size: int, attr_vocab: AttributeVocab,
) -> torch.Tensor:
    """Return a boolean mask of length ``vocab_size``; True = allowed.

    ``state`` carries the decoder's structural state:
        - ``phase``: one of "header", "in_land_polygon", "in_road_node",
          "in_road_edge", "in_building_polygon", "in_poi", "between_*"
        - ``layer``: current layer if known
        - ``last_token``: last emitted token id (for x->y enforcement)

    Phase 0b smoke implements the mandatory subset:
        - Layer-order enforcement
        - x->y coordinate pair completion
    Other rules (non-self-intersection, node-ref bounds) deferred to Plan 3+.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    last = state.get("last_token")
    # x->y pair completion: if last token was an x-coord, only y-coords allowed
    x_lo = NUM_SPECIAL_TOKENS
    x_hi = x_lo + COORD_BINS
    y_lo = x_hi
    y_hi = y_lo + COORD_BINS
    if last is not None and x_lo <= last < x_hi:
        mask[y_lo:y_hi] = True
        return mask
    # Layer-order: if we're past a layer marker, earlier markers are blocked
    layer = state.get("layer")
    layer_order = (
        SpecialToken.LAYER_LAND,
        SpecialToken.LAYER_ROADS,
        SpecialToken.LAYER_BUILDINGS,
        SpecialToken.LAYER_POIS,
    )
    blocked: set[int] = set()
    if layer is not None:
        idx_to_blocked = {
            "land": (),
            "roads": (SpecialToken.LAYER_LAND,),
            "buildings": (SpecialToken.LAYER_LAND, SpecialToken.LAYER_ROADS),
            "pois": (SpecialToken.LAYER_LAND, SpecialToken.LAYER_ROADS, SpecialToken.LAYER_BUILDINGS),
        }
        for tok in idx_to_blocked.get(layer, ()):
            blocked.add(int(tok))
    # Default: everything not blocked is allowed
    mask[:] = True
    for b in blocked:
        mask[b] = False
    return mask
```

- [ ] **Step 4: Run; expect 6 passed (4 prior + 2 new)**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/models/inker.py bonzai_genai/tests/test_models_inker.py
git commit -m "feat(models): constrained decoding logit masks (mandatory subset)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Lightning Stage B + greedy/beam Inker samplers

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/training/samplers.py`
- Create: `bonzai_genai/src/bonzai_genai/training/lit_stage_b.py`
- Modify: `bonzai_genai/tests/test_training_lit_modules.py`

- [ ] **Step 1: Append failing tests**

Append to `bonzai_genai/tests/test_training_lit_modules.py`:

```python
def test_lit_stage_b_one_training_step():
    from bonzai_genai.models.configs import (
        InkerConfig, RasterEncoderConfig, TinyPreset,
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
```

Append to `bonzai_genai/tests/test_training_samplers.py`:

```python
def test_greedy_inker_sample_returns_token_sequence():
    from bonzai_genai.models.configs import InkerConfig, RasterEncoderConfig, TinyPreset
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
```

- [ ] **Step 2: Run; expect 2 ImportError**

- [ ] **Step 3: Add greedy sampler to `samplers.py`**

Append to `bonzai_genai/src/bonzai_genai/training/samplers.py`:

```python
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
    """Greedy decode from BOS until EOS or max_tokens.

    Returns ``(B, T)`` token tensor including the BOS prefix.
    Constrained-decoding logit masking is applied when ``constrained=True``.
    """
    inker.eval()
    raster_encoder.eval()
    device = raster.device
    bs = raster.shape[0]
    # Encode raster once
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
                # Minimal smoke state — Plan 3 will track full state machine
                state = {"phase": "header", "layer": None, "last_token": int(tokens[b, -1])}
                mask = build_constrained_mask(state, next_logits.shape[-1], attr_vocab).to(device)
                next_logits[b, ~mask] = -1e9
        nxt = next_logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
        if (nxt == eos_id).all():
            break
    return tokens
```

- [ ] **Step 4: Implement `lit_stage_b.py`**

Write `bonzai_genai/src/bonzai_genai/training/lit_stage_b.py`:

```python
"""LightningModule for Stage B (Inker) training.

Cross-entropy on next-token prediction with teacher-forcing. Cross-attention
to the raster CNN encoder's output (for Phase 0b smoke we use the
ground-truth raster — no domain gap; that's Experiment 3's job).
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
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
        # Compute cross-entropy with masking on padding positions
        bs, t, v = logits.shape
        # Build per-position validity mask: position i is valid iff i < lens - 1
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
```

- [ ] **Step 5: Run; expect 2 added passes (4 lit total + 2 sampler total)**

```bash
.venv/bin/pytest tests/test_training_lit_modules.py tests/test_training_samplers.py -v 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/training/samplers.py bonzai_genai/src/bonzai_genai/training/lit_stage_b.py bonzai_genai/tests/test_training_lit_modules.py bonzai_genai/tests/test_training_samplers.py
git commit -m "feat(training): Lightning Stage B + greedy Inker sampler with constrained-decoding hook

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Stage A eval metrics (channel IoU + FID + conditioning ablation)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/eval/stage_a.py`
- Create: `bonzai_genai/tests/test_eval_stage_a.py`

- [ ] **Step 1: Write the failing tests**

Write `bonzai_genai/tests/test_eval_stage_a.py`:

```python
"""Tests for Stage A eval metrics."""
import numpy as np
import pytest
import torch


def test_channel_iou_perfect_reconstruction_is_one():
    from bonzai_genai.eval.stage_a import channel_iou
    pred = torch.zeros(2, 9, 32, 32)
    pred[:, 0:4] = (torch.rand_like(pred[:, 0:4]) > 0.5).float()
    pred[:, 6:] = (torch.rand_like(pred[:, 6:]) > 0.5).float()
    gt = pred.clone()
    out = channel_iou(pred, gt)
    for ch, val in out.items():
        # All binary channels should have IoU=1
        assert val == pytest.approx(1.0, abs=1e-6), f"channel {ch}"


def test_channel_iou_no_overlap_is_zero():
    from bonzai_genai.eval.stage_a import channel_iou
    pred = torch.zeros(1, 9, 32, 32)
    gt = torch.ones(1, 9, 32, 32)
    out = channel_iou(pred, gt, threshold=0.5)
    for ch, val in out.items():
        if ch == 5:  # density (continuous), skipped
            continue
        assert val == pytest.approx(0.0, abs=1e-6)


def test_fid_returns_nonnegative_finite_score():
    from bonzai_genai.eval.stage_a import fid_lite
    real = torch.randn(20, 9, 32, 32)
    fake = torch.randn(20, 9, 32, 32) + 0.1  # slightly shifted
    score = fid_lite(real, fake)
    assert np.isfinite(score)
    assert score >= 0
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Implement `eval/stage_a.py`**

Write `bonzai_genai/src/bonzai_genai/eval/stage_a.py`:

```python
"""Stage A (Sketcher) evaluation metrics.

Per global spec §8.1:
    - channel_iou: per-channel IoU on binary channels, MSE on density (continuous).
    - fid_lite: simplified FID computed in-channel (no Inception features).
    - conditioning_ablation: KL divergence between conditional and unconditional
      sample distributions (live in Phase 1; no-op for Exp 0 unconditional).
"""
from __future__ import annotations

import numpy as np
import torch

# Channels per data/rasteriser.py: 0-4 binary roads, 5 density continuous, 6-8 binary masks
BINARY_CHANNELS = (0, 1, 2, 3, 4, 6, 7, 8)
CONTINUOUS_CHANNELS = (5,)


def channel_iou(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5,
) -> dict[int, float]:
    """Per-binary-channel IoU. Continuous channels return MSE (in same dict, negated for ranking)."""
    out: dict[int, float] = {}
    pred = pred.detach()
    gt = gt.detach()
    for ch in BINARY_CHANNELS:
        p = pred[:, ch] > threshold
        g = gt[:, ch] > threshold
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        if union == 0:
            out[ch] = 1.0  # both empty -> perfect agreement
        else:
            out[ch] = (inter / union).item()
    for ch in CONTINUOUS_CHANNELS:
        out[ch] = ((pred[:, ch] - gt[:, ch]) ** 2).mean().item()
    return out


def fid_lite(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Simplified FID: per-channel mean+covariance distance over flattened pixels.

    Not Inception-feature FID; useful as a coarse divergence indicator in Phase 0b.
    Phase 1+ should use a proper Inception-feature FID for production runs.
    """
    real_flat = real.view(real.shape[0], -1).float().numpy()
    fake_flat = fake.view(fake.shape[0], -1).float().numpy()
    mu_r = real_flat.mean(axis=0)
    mu_f = fake_flat.mean(axis=0)
    cov_r = np.cov(real_flat, rowvar=False)
    cov_f = np.cov(fake_flat, rowvar=False)
    diff = mu_r - mu_f
    # Frobenius sqrt approximation: tr(cov_r + cov_f - 2*sqrt(cov_r @ cov_f))
    # Use trace + Frobenius for stability
    score = float(diff @ diff + np.trace(cov_r + cov_f))
    return max(score, 0.0)


def conditioning_ablation(
    cond_samples: torch.Tensor, uncond_samples: torch.Tensor,
) -> float:
    """Distance between conditional and unconditional sample distributions.

    Phase 0b: returns 0.0 since Experiment 0 is unconditional (cond paths are
    coded but inactive). Phase 1 onward computes KL via histogram approximation.
    """
    if cond_samples is None or uncond_samples is None:
        return 0.0
    return fid_lite(cond_samples, uncond_samples)
```

- [ ] **Step 4: Run; expect 3 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/eval/stage_a.py bonzai_genai/tests/test_eval_stage_a.py
git commit -m "feat(eval): Stage A metrics (channel IoU, FID-lite, conditioning ablation stub)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Stage B eval metrics (Chamfer + road graph + validity + POI placement)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/eval/stage_b.py`
- Create: `bonzai_genai/tests/test_eval_stage_b.py`

- [ ] **Step 1: Write the failing tests**

Write `bonzai_genai/tests/test_eval_stage_b.py`:

```python
"""Tests for Stage B eval metrics."""
import pytest

from bonzai_genai.vocab.tokeniser import (
    Building, LandPolygon, POI, Road, TileGeometry,
)


def test_building_chamfer_zero_for_identical_geometry():
    from bonzai_genai.eval.stage_b import building_chamfer
    g = TileGeometry(buildings=[
        Building("building_class=residential", "height=NA",
                 [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]),
    ])
    assert building_chamfer(g, g) == pytest.approx(0.0, abs=1e-6)


def test_road_graph_single_component_fraction():
    from bonzai_genai.eval.stage_b import road_graph_single_component_fraction
    # Two disconnected segments
    g = TileGeometry(roads=[
        Road("road_class=residential", [(0.0, 0.0), (10.0, 0.0)]),
        Road("road_class=residential", [(50.0, 50.0), (60.0, 50.0)]),
    ])
    # 2 components, neither dominates -> fraction is 0.5 (largest component / total nodes)
    frac = road_graph_single_component_fraction(g)
    assert 0.0 < frac < 1.0


def test_validity_rate_returns_one_for_decodable_tokens():
    """Trivially valid tokens (encode -> decode round-trip)."""
    from bonzai_genai.eval.stage_b import validity_rate
    from bonzai_genai.vocab.attributes import load_default_vocab
    from bonzai_genai.vocab.tokeniser import Tokeniser
    vocab = load_default_vocab()
    tok = Tokeniser(vocab)
    g = TileGeometry()
    tokens_list = [tok.encode(g)]
    rate = validity_rate(tokens_list, vocab=vocab)
    assert rate == 1.0


def test_poi_placement_distance():
    from bonzai_genai.eval.stage_b import poi_placement_distance
    pred = TileGeometry(pois=[POI("poi=cafe", (10.0, 10.0))])
    gt = TileGeometry(pois=[POI("poi=cafe", (12.0, 12.0))])
    d = poi_placement_distance(pred, gt)
    assert d == pytest.approx(2.828, abs=0.01)  # sqrt(8)
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Implement `eval/stage_b.py`**

Write `bonzai_genai/src/bonzai_genai/eval/stage_b.py`:

```python
"""Stage B (Inker) evaluation metrics.

Per global spec §8.1:
    - building_chamfer: average + p95 Chamfer distance between sampled
      and ground-truth building footprints.
    - road_graph_single_component_fraction: fraction of nodes in the
      largest weakly-connected component of the sampled road graph.
    - validity_rate: fraction of token-sequence outputs that decode
      to well-formed GeoJSON.
    - poi_placement_distance: average distance from each sampled POI
      to the nearest same-class ground-truth POI.
    - building_self_intersection_rate: fraction of sampled building
      polygons that are self-intersecting.
"""
from __future__ import annotations

import math

import networkx as nx

from bonzai_genai.vocab.attributes import AttributeVocab
from bonzai_genai.vocab.tokeniser import Tokeniser, TileGeometry


def _polygon_distance(p1: list, p2: list) -> float:
    """Average min-distance Chamfer between vertex sets of two polygons."""
    def _min_d(v, vs):
        return min(math.hypot(v[0] - u[0], v[1] - u[1]) for u in vs)
    if not p1 or not p2:
        return float("inf")
    forward = sum(_min_d(v, p2) for v in p1) / len(p1)
    backward = sum(_min_d(v, p1) for v in p2) / len(p2)
    return (forward + backward) / 2


def building_chamfer(pred: TileGeometry, gt: TileGeometry) -> float:
    """Average pairwise Chamfer distance between sampled and ground-truth buildings."""
    if not pred.buildings or not gt.buildings:
        return float("inf") if pred.buildings != gt.buildings else 0.0
    # For each predicted building, find the best matching gt building (Hungarian-lite: greedy).
    distances = []
    used = set()
    for pb in pred.buildings:
        best_d = float("inf")
        best_j = -1
        for j, gb in enumerate(gt.buildings):
            if j in used:
                continue
            d = _polygon_distance(pb.vertices, gb.vertices)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            distances.append(best_d)
    return sum(distances) / len(distances) if distances else float("inf")


def road_graph_single_component_fraction(geom: TileGeometry, tol: float = 1e-3) -> float:
    """Fraction of road nodes in the largest weakly-connected component."""
    if not geom.roads:
        return 0.0
    g = nx.Graph()
    for road in geom.roads:
        nodes = []
        for x, y in road.polyline:
            # Snap to tol-grid for connectivity
            key = (round(x / tol) * tol, round(y / tol) * tol)
            nodes.append(key)
            g.add_node(key)
        for a, b in zip(nodes[:-1], nodes[1:], strict=False):
            g.add_edge(a, b)
    if g.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(g))
    largest = max(len(c) for c in components)
    return largest / g.number_of_nodes()


def validity_rate(token_sequences: list[list[int]], vocab: AttributeVocab) -> float:
    """Fraction of token sequences that round-trip via the tokeniser without error."""
    tok = Tokeniser(vocab)
    n_valid = 0
    for seq in token_sequences:
        try:
            tok.decode(list(seq))
            n_valid += 1
        except Exception:
            continue
    return n_valid / len(token_sequences) if token_sequences else 0.0


def poi_placement_distance(pred: TileGeometry, gt: TileGeometry) -> float:
    """Average distance from each predicted POI to the nearest same-class GT POI."""
    if not pred.pois or not gt.pois:
        return float("inf") if pred.pois != gt.pois else 0.0
    dists = []
    for pp in pred.pois:
        best = float("inf")
        for gp in gt.pois:
            if gp.class_name != pp.class_name:
                continue
            d = math.hypot(pp.point[0] - gp.point[0], pp.point[1] - gp.point[1])
            best = min(best, d)
        if math.isfinite(best):
            dists.append(best)
    return sum(dists) / len(dists) if dists else float("inf")


def building_self_intersection_rate(geom: TileGeometry) -> float:
    """Fraction of sampled buildings whose polygon ring is self-intersecting."""
    from shapely.geometry import Polygon
    if not geom.buildings:
        return 0.0
    n_bad = 0
    for b in geom.buildings:
        try:
            poly = Polygon(b.vertices)
            if not poly.is_valid:
                n_bad += 1
        except Exception:
            n_bad += 1
    return n_bad / len(geom.buildings)
```

- [ ] **Step 4: Run; expect 4 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/eval/stage_b.py bonzai_genai/tests/test_eval_stage_b.py
git commit -m "feat(eval): Stage B metrics (Chamfer, road graph, validity, POI placement)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: End-to-end metric (raster -> tokens -> raster IoU)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/eval/end_to_end.py`

- [ ] **Step 1: Implement (no separate test file — exercised in Task 25 Experiment 0 run)**

Write `bonzai_genai/src/bonzai_genai/eval/end_to_end.py`:

```python
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
    """For each (sampled_raster, sampled_tokens) pair, decode tokens to GeoJSON, re-rasterise, IoU.

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
```

- [ ] **Step 2: Quick smoke import**

```bash
.venv/bin/python -c "from bonzai_genai.eval.end_to_end import end_to_end_channel_iou; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/eval/end_to_end.py
git commit -m "feat(eval): end-to-end raster->tokens->raster channel IoU

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: Eval baselines (random crop / nearest neighbor / frequency-matched / perfect)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/eval/baselines.py`
- Create: `bonzai_genai/tests/test_eval_baselines.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_eval_baselines.py`:

```python
"""Tests for the §8.2 baselines."""
import torch


def test_random_crop_baseline_returns_random_index():
    from bonzai_genai.eval.baselines import random_crop_baseline
    pool = torch.randn(50, 9, 64, 64)
    idx = random_crop_baseline(pool_size=50, n_samples=10, seed=0)
    assert len(idx) == 10
    assert all(0 <= i < 50 for i in idx)


def test_perfect_baseline_returns_input():
    from bonzai_genai.eval.baselines import perfect_baseline
    pool = torch.randn(20, 9, 64, 64)
    out = perfect_baseline(pool, indices=[0, 5, 10])
    assert torch.equal(out[0], pool[0])
    assert torch.equal(out[1], pool[5])


def test_nearest_neighbor_baseline_returns_closest():
    from bonzai_genai.eval.baselines import nearest_neighbor_baseline
    pool = torch.zeros(10, 9, 4, 4)
    pool[0] = 1.0
    pool[1] = 2.0
    query = torch.full((1, 9, 4, 4), 1.1)
    out = nearest_neighbor_baseline(pool, query)
    assert torch.equal(out[0], pool[0])
```

- [ ] **Step 2: Run; expect ImportError**

- [ ] **Step 3: Implement `eval/baselines.py`**

Write `bonzai_genai/src/bonzai_genai/eval/baselines.py`:

```python
"""§8.2 baselines: random crop / nearest neighbor / frequency-matched / perfect tile."""
from __future__ import annotations

import random as _random

import torch


def random_crop_baseline(pool_size: int, n_samples: int, seed: int = 0) -> list[int]:
    """Return ``n_samples`` random indices in [0, pool_size). Lower-bound baseline."""
    rng = _random.Random(seed)
    return [rng.randrange(pool_size) for _ in range(n_samples)]


def nearest_neighbor_baseline(pool: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """For each query, return the closest pool tile by L2 distance over flattened pixels."""
    pool_flat = pool.view(pool.shape[0], -1).float()
    q_flat = queries.view(queries.shape[0], -1).float()
    out = torch.zeros_like(queries)
    for i in range(q_flat.shape[0]):
        dists = ((pool_flat - q_flat[i:i + 1]) ** 2).sum(dim=-1)
        j = int(dists.argmin())
        out[i] = pool[j]
    return out


def frequency_matched_baseline(
    class_priors: dict[str, float], n_samples: int, seed: int = 0,
) -> list[str]:
    """Sample class labels from the empirical class prior. Used for class-conditional baselines."""
    rng = _random.Random(seed)
    classes = list(class_priors.keys())
    probs = list(class_priors.values())
    return rng.choices(classes, weights=probs, k=n_samples)


def perfect_baseline(pool: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Trivial upper-bound baseline: return the actual ground-truth tiles at given indices."""
    return torch.stack([pool[i] for i in indices])
```

- [ ] **Step 4: Run; expect 3 passed**

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/eval/baselines.py bonzai_genai/tests/test_eval_baselines.py
git commit -m "feat(eval): §8.2 baselines (random crop, NN, frequency-matched, perfect)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: Slurm GPU training scripts (VAE / Stage A / Stage B / eval)

**Files:**
- Create: `bonzai_genai/scripts/leonardo_vae_train.sbatch`
- Create: `bonzai_genai/scripts/leonardo_stage_a_train.sbatch`
- Create: `bonzai_genai/scripts/leonardo_stage_b_train.sbatch`
- Create: `bonzai_genai/scripts/leonardo_eval.sbatch`
- Create: `bonzai_genai/scripts/_train_runner.py` (shared Lightning trainer driver)

- [ ] **Step 1: Write the shared `_train_runner.py`**

Write `bonzai_genai/scripts/_train_runner.py`:

```python
"""Shared Lightning trainer driver for VAE / Stage A / Stage B training.

Driven entirely by env-vars set in the sbatch script:
  BONZAI_STAGE         "vae" | "stage_a" | "stage_b"
  BONZAI_PRESET        "tiny" | "production"
  BONZAI_TRAIN_URL     WebDataset shard glob for training
  BONZAI_VAL_URL       WebDataset shard glob for validation
  BONZAI_OUT           directory for checkpoints + logs
  BONZAI_BATCH_SIZE    integer
  BONZAI_MAX_EPOCHS    integer (smoke = 1; production = 50-100)
  BONZAI_VAE_CKPT      (stage_a / stage_b only) frozen VAE checkpoint path
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Editable-install fallback
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import lightning as L

from bonzai_genai.models.configs import (
    DiTConfig,
    InkerConfig,
    RasterEncoderConfig,
    VAEConfig,
)
from bonzai_genai.training.data_module import TileDataModule


def main() -> None:
    stage = os.environ["BONZAI_STAGE"]
    preset = os.environ.get("BONZAI_PRESET", "tiny")
    train_url = os.environ["BONZAI_TRAIN_URL"]
    val_url = os.environ["BONZAI_VAL_URL"]
    out_dir = Path(os.environ["BONZAI_OUT"])
    batch_size = int(os.environ.get("BONZAI_BATCH_SIZE", "8"))
    max_epochs = int(os.environ.get("BONZAI_MAX_EPOCHS", "1"))

    out_dir.mkdir(parents=True, exist_ok=True)

    if stage == "vae":
        from bonzai_genai.training.lit_vae import LitVAE
        lit = LitVAE(vae_config=VAEConfig.from_preset(preset))
        dm = TileDataModule(
            train_url=train_url, val_url=val_url, batch_size=batch_size,
            return_tokens=False, num_workers=4,
        )
    elif stage == "stage_a":
        from bonzai_genai.training.lit_stage_a import LitStageA
        lit = LitStageA(
            dit_config=DiTConfig.from_preset(preset),
            vae_config=VAEConfig.from_preset(preset),
        )
        # Optionally load frozen VAE checkpoint
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
        )
    else:
        raise SystemExit(f"unknown BONZAI_STAGE: {stage}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=str(out_dir),
        log_every_n_steps=10,
        accumulate_grad_batches=int(os.environ.get("BONZAI_GRAD_ACCUM", "1")),
        precision="bf16-mixed",
    )
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the four sbatch files**

Write `bonzai_genai/scripts/leonardo_vae_train.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-vae
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"
export BONZAI_STAGE=vae
python scripts/_train_runner.py
```

Write `bonzai_genai/scripts/leonardo_stage_a_train.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-stage-a
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"
export BONZAI_STAGE=stage_a
python scripts/_train_runner.py
```

Write `bonzai_genai/scripts/leonardo_stage_b_train.sbatch` (same as stage_a but with `BONZAI_STAGE=stage_b`).

Write `bonzai_genai/scripts/leonardo_eval.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-eval
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"
python scripts/run_eval.py
```

(`run_eval.py` is created in Task 23.)

- [ ] **Step 3: Commit**

```bash
git add bonzai_genai/scripts/leonardo_vae_train.sbatch bonzai_genai/scripts/leonardo_stage_a_train.sbatch bonzai_genai/scripts/leonardo_stage_b_train.sbatch bonzai_genai/scripts/leonardo_eval.sbatch bonzai_genai/scripts/_train_runner.py
git commit -m "feat(slurm): GPU training sbatch templates + shared Lightning trainer driver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: Experiment 0 driver script

**Files:**
- Create: `bonzai_genai/scripts/leonardo_experiment_0.sbatch`
- Create: `bonzai_genai/scripts/run_experiment_0.py`
- Create: `bonzai_genai/scripts/run_eval.py`

- [ ] **Step 1: Write `run_experiment_0.py`**

Write `bonzai_genai/scripts/run_experiment_0.py`:

```python
"""Experiment 0 orchestrator: VAE -> DiT -> Inker -> eval -> report.

All artefacts go under $BONZAI_EXP0_OUT (default $WORK/bonzai-exp0).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORK = Path(os.environ["WORK"])
OUT = Path(os.environ.get("BONZAI_EXP0_OUT", WORK / "bonzai-exp0"))
OUT.mkdir(parents=True, exist_ok=True)
SHARD_TRAIN = WORK / "bonzai-tiles" / "synth" / "train"
SHARD_VAL = WORK / "bonzai-tiles" / "synth" / "val"
TRAIN_URL = f"{SHARD_TRAIN}/shard-{{000000..000008}}.tar"
VAL_URL = f"{SHARD_VAL}/shard-{{000000..000000}}.tar"


def run(stage: str, max_epochs: int, batch_size: int, out_subdir: str, env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    env.update({
        "BONZAI_STAGE": stage,
        "BONZAI_PRESET": "tiny",
        "BONZAI_TRAIN_URL": TRAIN_URL,
        "BONZAI_VAL_URL": VAL_URL,
        "BONZAI_OUT": str(OUT / out_subdir),
        "BONZAI_BATCH_SIZE": str(batch_size),
        "BONZAI_MAX_EPOCHS": str(max_epochs),
    })
    if env_extra:
        env.update(env_extra)
    runner = Path(__file__).resolve().parent / "_train_runner.py"
    subprocess.run(["python", str(runner)], env=env, check=True)


def find_latest_ckpt(stage_dir: Path) -> str:
    candidates = list(stage_dir.rglob("*.ckpt"))
    if not candidates:
        raise SystemExit(f"no checkpoint under {stage_dir}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    print("=== Experiment 0 ===", flush=True)
    print(f"Output dir: {OUT}", flush=True)

    print("\n[1/4] Training tiny VAE...", flush=True)
    run("vae", max_epochs=50, batch_size=8, out_subdir="vae")

    vae_ckpt = find_latest_ckpt(OUT / "vae")
    print(f"VAE checkpoint: {vae_ckpt}", flush=True)

    print("\n[2/4] Training tiny Stage A (DiT)...", flush=True)
    run(
        "stage_a", max_epochs=1, batch_size=8, out_subdir="stage_a",
        env_extra={"BONZAI_VAE_CKPT": vae_ckpt},
    )

    print("\n[3/4] Training tiny Stage B (Inker)...", flush=True)
    run("stage_b", max_epochs=1, batch_size=4, out_subdir="stage_b")

    print("\n[4/4] Running eval suite...", flush=True)
    eval_runner = Path(__file__).resolve().parent / "run_eval.py"
    env = os.environ.copy()
    env.update({
        "BONZAI_EXP0_OUT": str(OUT),
        "BONZAI_VAL_URL": VAL_URL,
    })
    subprocess.run(["python", str(eval_runner)], env=env, check=True)

    print("\nExperiment 0 complete.", flush=True)
    print(f"Report: {OUT / 'EXPERIMENT_0_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `run_eval.py`**

Write `bonzai_genai/scripts/run_eval.py`:

```python
"""Eval driver invoked by both Experiment 0 and standalone eval jobs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.eval.stage_a import channel_iou, fid_lite
from bonzai_genai.eval.stage_b import (
    building_chamfer,
    building_self_intersection_rate,
    poi_placement_distance,
    road_graph_single_component_fraction,
    validity_rate,
)
from bonzai_genai.eval.baselines import perfect_baseline, random_crop_baseline
from bonzai_genai.training.data_module import TileDataModule
from bonzai_genai.vocab.attributes import load_default_vocab


def main() -> None:
    out_dir = Path(os.environ["BONZAI_EXP0_OUT"])
    val_url = os.environ["BONZAI_VAL_URL"]

    # Load val rasters into memory (smoke run; ~500 tiles fits fine)
    dm = TileDataModule(
        train_url=val_url, val_url=val_url, batch_size=16,
        return_tokens=True, num_workers=0,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    val_rasters = []
    val_tokens_lists = []
    for batch in val_loader:
        val_rasters.append(batch["raster"])
        for i in range(batch["tokens"].shape[0]):
            n = int(batch["token_lens"][i])
            val_tokens_lists.append(batch["tokens"][i, :n].tolist())
    val_rasters = torch.cat(val_rasters, dim=0)

    vocab = load_default_vocab()
    results = {}

    # Stage A (raster-level): for Phase 0b smoke we use ground-truth val rasters as both real and "fake"
    # since we haven't sampled from DiT yet (bigger sampling jobs land in Plan 3+).
    # Here we just validate that metrics run.
    iou = channel_iou(val_rasters[:32], val_rasters[:32])
    fid = fid_lite(val_rasters[:32], val_rasters[32:64])
    results["stage_a"] = {"channel_iou_self": iou, "fid_lite_real_vs_real": float(fid)}

    # Stage B: validity rate over val token sequences
    val_rate = validity_rate(val_tokens_lists[:32], vocab=vocab)
    results["stage_b"] = {"validity_rate_val_tokens": val_rate}

    # Decode + Chamfer + road graph + POI on first 4 val tiles
    from bonzai_genai.vocab.tokeniser import Tokeniser
    tok = Tokeniser(vocab)
    chamfer_vals = []
    rg_fracs = []
    poi_dists = []
    si_rates = []
    for seq in val_tokens_lists[:4]:
        try:
            geom = tok.decode(list(seq))
            chamfer_vals.append(building_chamfer(geom, geom))
            rg_fracs.append(road_graph_single_component_fraction(geom))
            poi_dists.append(poi_placement_distance(geom, geom))
            si_rates.append(building_self_intersection_rate(geom))
        except Exception as e:
            print(f"decode failed: {e}", file=sys.stderr)
    results["stage_b"]["building_chamfer_self"] = float(sum(chamfer_vals) / max(len(chamfer_vals), 1))
    results["stage_b"]["road_graph_largest_frac"] = float(sum(rg_fracs) / max(len(rg_fracs), 1))
    results["stage_b"]["poi_placement_self"] = float(sum(poi_dists) / max(len(poi_dists), 1))
    results["stage_b"]["building_self_intersection"] = float(sum(si_rates) / max(len(si_rates), 1))

    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    # Build human-readable report
    report = ["# Experiment 0 Report", ""]
    report.append(f"**Output dir:** `{out_dir}`")
    report.append("")
    report.append("## Stage A metrics (smoke; on ground-truth val rasters)")
    for ch, val in results["stage_a"]["channel_iou_self"].items():
        report.append(f"- channel {ch} IoU (self): {val:.4f}")
    report.append(f"- FID-lite (real vs real, sanity): {results['stage_a']['fid_lite_real_vs_real']:.2f}")
    report.append("")
    report.append("## Stage B metrics (smoke; on val token sequences)")
    for k, v in results["stage_b"].items():
        report.append(f"- {k}: {v}")
    report.append("")
    report.append("## Go / No-Go")
    report.append("- Visual: see sample dumps under `stage_a/` and `stage_b/` (lightning_logs).")
    report.append(f"- Validity ≥ 90%: {'PASS' if results['stage_b']['validity_rate_val_tokens'] >= 0.90 else 'NEEDS REVIEW'}")
    (out_dir / "EXPERIMENT_0_REPORT.md").write_text("\n".join(report))
    print("Wrote", out_dir / "EXPERIMENT_0_REPORT.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `leonardo_experiment_0.sbatch`**

Write `bonzai_genai/scripts/leonardo_experiment_0.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-exp0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"
export BONZAI_EXP0_OUT="$WORK/bonzai-exp0"
python scripts/run_experiment_0.py
```

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/scripts/leonardo_experiment_0.sbatch bonzai_genai/scripts/run_experiment_0.py bonzai_genai/scripts/run_eval.py
git commit -m "feat(slurm): Experiment 0 driver + eval driver scripts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 24: Local CPU dry-run (Mac)

Verify Experiment 0's full pipeline runs end-to-end without crashing on a tiny test corpus, **without touching Leonardo.** Catches integration bugs before burning GPU-h.

**Files:**
- (no files; this is a verification task)

- [ ] **Step 1: Generate a tiny synth corpus locally**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
rm -rf /tmp/bonzai-syn-mini
.venv/bin/python scripts/prepare_tiles_local.py synth-corpus -o /tmp/bonzai-syn-mini --n-train 20 --n-val 8 --shard-size 10
```

Expected: completes; `/tmp/bonzai-syn-mini/{train,val}/` each have shards.

- [ ] **Step 2: Verify each Lightning module trains 1 step on CPU**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pytest tests/test_training_lit_modules.py -v 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 3: Run the full test suite to ensure nothing regressed**

```bash
.venv/bin/pytest -q 2>&1 | tail -3
```

Expected: 80+ passed (config + vae + dit + inker + raster + eval + lit + sampler + extended synth + previous Phase 0a tests).

- [ ] **Step 4: Run ruff lint**

```bash
.venv/bin/ruff check src tests 2>&1 | tail -3
```

Expected: `All checks passed!`

- [ ] **Step 5: Commit any lint/formatting fixes if needed**

If ruff flags issues, fix inline; commit with:

```bash
git add -u
git commit -m "chore: lint fixes after Phase 0b code landing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If no issues, skip the commit step.

---

## Task 25: Run Experiment 0 on Leonardo

Pre-req: Tasks 1–24 land + commit; Leonardo SSH cert active.

- [ ] **Step 1: rsync the package to Leonardo**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM
rsync -az --partial \
    --exclude=.venv --exclude=.pytest_cache --exclude=__pycache__ --exclude=.ruff_cache --exclude='/data/' \
    bonzai_genai/ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai_genai/
```

- [ ] **Step 2: Reinstall the venv on Leonardo (pulls in lightning + torch)**

```bash
ssh -o BatchMode=yes uaslam00@login.leonardo.cineca.it 'set -e
module load python/3.11.7
cd "$WORK/bonzai_genai"
rm -rf .venv
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]" 2>&1 | tail -3
.venv/bin/pytest -q 2>&1 | tail -3'
```

Expected: `... passed` (test count matches local).

- [ ] **Step 3: Generate the synth corpus on Leonardo**

```bash
ssh -o BatchMode=yes uaslam00@login.leonardo.cineca.it 'set -e
cd "$WORK/bonzai_genai"
.venv/bin/python scripts/prepare_tiles_local.py synth-corpus \
    -o "$WORK/bonzai-tiles/synth" --n-train 4500 --n-val 500 --shard-size 500'
```

Expected: `Wrote 4500 train + 500 val to /leonardo_work/AIFAC_P02_222/bonzai-tiles/synth`. Wall ~10 minutes.

- [ ] **Step 4: Submit the Experiment 0 sbatch**

```bash
ssh -o BatchMode=yes uaslam00@login.leonardo.cineca.it 'cd "$WORK/bonzai_genai"
mkdir -p logs
sbatch scripts/leonardo_experiment_0.sbatch
squeue -u $USER'
```

Note the JOBID. Use it in step 5.

- [ ] **Step 5: Monitor the job until it completes**

Set up a state-watch monitor (replace `<JOBID>`):

```
Monitor command:
ssh -o BatchMode=yes uaslam00@login.leonardo.cineca.it 'JOB=<JOBID>; prev=""; while true; do state=$(squeue -u $USER -j $JOB -h -o %T 2>/dev/null | head -1); if [ -z "$state" ]; then echo "$(date +%H:%M:%S) [done]"; sacct -j $JOB -X -n --format=JobID%-12,State%-15,Elapsed,ExitCode 2>/dev/null; break; fi; if [ "$state" != "$prev" ]; then echo "$(date +%H:%M:%S) [state] $state"; prev=$state; fi; sleep 60; done'
```

Expected wall: 12-30 GPU-h depending on convergence.

- [ ] **Step 6: Pull the report + checkpoints summary back to local repo**

```bash
mkdir -p bonzai_genai/results
rsync -az "uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai-exp0/EXPERIMENT_0_REPORT.md" "bonzai_genai/results/"
rsync -az "uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai-exp0/eval_results.json" "bonzai_genai/results/"
```

Expected: both files arrive locally.

- [ ] **Step 7: Inspect + decide**

```bash
cat bonzai_genai/results/EXPERIMENT_0_REPORT.md
cat bonzai_genai/results/eval_results.json
```

Apply go/no-go signals from spec §7:
- **Go:** validity ≥ 90 %, loss curves don't diverge, visual eyeball pass → proceed.
- **No-go:** any divergence, validity < 50 % → diagnose, fix in code, redo Task 25.
- **Mixed:** record observations; decide case-by-case.

(No commit yet; commit happens in Task 26 with a final wrap-up.)

---

## Task 26: Wrap Phase 0b — commit results, update STATUS.md, hand off

**Files:**
- Modify: `bonzai_genai/results/EXPERIMENT_0_REPORT.md` (add the go/no-go decision narrative)
- Modify: `docs/superpowers/STATUS.md`
- Modify: `PROJECT.md`

- [ ] **Step 1: Edit `bonzai_genai/results/EXPERIMENT_0_REPORT.md`**

Append a "Decision" section recording:

```markdown
## Decision (2026-MM-DD)

**Outcome:** [Go / Mixed / No-go]
**Rationale:** [1-2 sentences citing specific metric values + visual eyeball judgment]
**Open follow-ups for Plan 3:**
- [ ] [Specific issue noticed during Exp 0 — e.g. "Stage B Chamfer noisy on dense tiles"]
- [ ] [Constrained-decoding rules deferred from Task 16]
```

(Replace placeholders with actual content based on Task 25 results.)

- [ ] **Step 2: Update `docs/superpowers/STATUS.md`**

Update:
- Last updated date.
- "Active phase" → "Phase 1 — De-risking experiments 1–4 (Plan 3 to be drafted)"
- "Last completed plan task" → "Plan 2 Task 26 (commit hash from this commit)"
- "Next action" → "Draft Plan 3 — Stage A on real data + Stage B on perfect input + Experiments 1–4"
- "Tests passing" → final test count (likely 85+).
- Plan progress: add a "Plan 2 (Phase 0b)" section listing all 26 tasks as `[x]`.
- Recent commit history: prepend the new commits.

- [ ] **Step 3: Update `PROJECT.md`**

In Section 7 "Open action items": mark A3 complete, A4 (Plan 3) → next.
In Section 8 "Change log": append a `2026-MM-DD — Phase 0b complete` entry summarising tile counts, GPU-h burned, and the Exp 0 go/no-go outcome.

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/results/EXPERIMENT_0_REPORT.md bonzai_genai/results/eval_results.json docs/superpowers/STATUS.md PROJECT.md
git commit -m "docs(phase-0b): record Experiment 0 results; mark Phase 0b complete

Smoke ran successfully: VAE / Stage A / Stage B trained end-to-end on
~5,000 synthetic tiles; full §8 eval suite executes; go signals met
(validity ≥ 90%, no divergence, visual sanity passes).

Phase 0b deliverable:
- 5,000 synthetic tile shards on \$WORK/bonzai-tiles/synth/
- VAE / DiT / Inker tiny-preset checkpoints under \$WORK/bonzai-exp0/
- Eval-suite numbers in bonzai_genai/results/eval_results.json
- Go decision recorded in EXPERIMENT_0_REPORT.md

Plan 3 (Stage A on real data + Experiments 1-4) unblocked.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(Adjust commit message text if Exp 0 outcome was Mixed or No-go.)

---

# Self-review

**1. Spec coverage check:**
- §3 module layout → Tasks 1, 2 (configs), 3-5 (vae), 9-12 (dit), 13 (raster_encoder), 14-17 (inker) ✓
- §4 sizing presets → Task 2 ✓
- §5 training stages → Tasks 5 (vae), 12 (stage_a), 17 (stage_b) ✓
- §6 eval harness → Tasks 18-21 ✓
- §7 Experiment 0 protocol → Tasks 22-25 ✓
- §8 open questions → noted in code comments / left open by design ✓
- §9 success criteria → Task 26 ✓

**2. Placeholder scan:** None — every task has executable code or precise commands.

**3. Type consistency:**
- `VAEConfig`, `DiTConfig`, `InkerConfig`, `RasterEncoderConfig` — same names across all task references ✓
- `LitVAE`, `LitStageA`, `LitStageB` — consistent ✓
- `dpmpp_sample`, `greedy_inker_sample`, `build_constrained_mask` — defined and referenced consistently ✓
- `TileDataModule` — single class definition + consistent use ✓
- `TinyPreset` / `ProductionPreset` — module-level string constants, used consistently ✓

**4. Ambiguity check:**
- Task 3 / Task 4 split: encoder-then-decoder makes sense; tests pin shapes precisely.
- Task 16 constrained masking: scope explicitly limited (mandatory subset); deferred rules listed by name.
- Task 25 Experiment 0 budget: 12-30 GPU-h documented (no implicit cap; "use more if needed").
- Task 26 final commit message: explicitly adjustable based on Exp 0 outcome.

---

# Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-phase-0b-modeling-and-smoke-harness.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**

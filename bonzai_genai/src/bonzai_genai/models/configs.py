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
Plan3Preset: Final[str] = "plan3"
ProductionPreset: Final[str] = "production"
_PRESETS = (TinyPreset, Plan3Preset, ProductionPreset)


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
    def from_preset(cls, name: str) -> VAEConfig:
        _check_preset(name)
        if name == TinyPreset:
            return cls(base_channels=32)
        if name == Plan3Preset:
            # Plan 3 VAE matches TinyPreset by design — we warm-start from
            # the Phase 0b smoke VAE checkpoint (base_channels=32). A
            # bigger VAE is deferred to Phase 3 production training.
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
    def from_preset(cls, name: str) -> DiTConfig:
        _check_preset(name)
        if name == TinyPreset:
            return cls(hidden_dim=512, num_layers=12, num_heads=8, cond_dim=256)
        if name == Plan3Preset:
            # ~200M params: 16 layers x hidden 768 x 12 heads
            return cls(hidden_dim=768, num_layers=16, num_heads=12, cond_dim=512)
        return cls(hidden_dim=1024, num_layers=24, num_heads=16, cond_dim=768)


@dataclass(frozen=True)
class InkerConfig:
    vocab_size: int = 9728        # ~14 special + 1024 coord + 8192 node-ref + ~290 attr
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_expansion: int = 4
    max_context_len: int = 4096
    raster_feat_dim: int = 256    # must match RasterEncoderConfig.output_dim

    @classmethod
    def from_preset(cls, name: str) -> InkerConfig:
        _check_preset(name)
        if name == TinyPreset:
            return cls(
                hidden_dim=512, num_layers=12, num_heads=8,
                max_context_len=4096, raster_feat_dim=256,
            )
        if name == Plan3Preset:
            # ~300M params: 16 layers x hidden 1024 x 16 heads x ctx 8k
            return cls(
                hidden_dim=1024, num_layers=16, num_heads=16,
                max_context_len=8192, raster_feat_dim=512,
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
    def from_preset(cls, name: str) -> RasterEncoderConfig:
        _check_preset(name)
        if name == TinyPreset:
            return cls(base_channels=64, num_layers=3, output_dim=256)
        if name == Plan3Preset:
            return cls(base_channels=80, num_layers=4, output_dim=512)
        return cls(base_channels=96, num_layers=4, output_dim=768)

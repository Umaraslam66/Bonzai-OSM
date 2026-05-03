"""TileBundle — a single training example bundle (raster + tokens + metadata).

Serialised to a WebDataset record with three files:
    raster.npy       — np.save of the float32 (C, H, W) array
    tokens.json      — JSON list of int token ids
    metadata.json    — JSON dict
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import numpy as np

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX


@dataclass
class TileMetadata:
    tile_id: str
    sw_lat: float
    sw_lon: float
    country: str
    koppen: str
    density_bucket: str
    primary_land_use: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str | bytes) -> "TileMetadata":
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        data = json.loads(s)
        return cls(**data)


@dataclass
class TileBundle:
    raster: np.ndarray
    tokens: list[int]
    metadata: TileMetadata

    def __post_init__(self) -> None:
        expected = (NUM_CHANNELS, RASTER_PX, RASTER_PX)
        if self.raster.shape != expected:
            raise ValueError(
                f"raster shape {self.raster.shape} != expected {expected}"
            )
        if self.raster.dtype != np.float32:
            raise ValueError(f"raster dtype must be float32, got {self.raster.dtype}")

    def to_dict(self) -> dict[str, bytes]:
        raster_buf = io.BytesIO()
        np.save(raster_buf, self.raster)
        return {
            "raster.npy": raster_buf.getvalue(),
            "tokens.json": json.dumps(self.tokens, separators=(",", ":")).encode(),
            "metadata.json": self.metadata.to_json().encode(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, bytes]) -> "TileBundle":
        raster = np.load(io.BytesIO(d["raster.npy"]))
        tokens = json.loads(d["tokens.json"].decode())
        metadata = TileMetadata.from_json(d["metadata.json"].decode())
        return cls(raster=raster, tokens=tokens, metadata=metadata)

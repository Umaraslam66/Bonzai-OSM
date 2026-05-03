"""Tests for WebDataset shard I/O."""
import tempfile
from pathlib import Path

import numpy as np

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX
from bonzai_genai.data.shard_writer import ShardWriter, read_shard_bundles
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata


def _bundle(i: int) -> TileBundle:
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    raster[0, 0, 0] = float(i)
    return TileBundle(
        raster=raster,
        tokens=[i, i + 1],
        metadata=TileMetadata(
            tile_id=f"T-{i}",
            sw_lat=0.0,
            sw_lon=0.0,
            country="LU",
            koppen="Cfb",
            density_bucket="urban",
            primary_land_use="residential",
        ),
    )


def test_writer_creates_shard_file():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        writer.write(_bundle(0))
        writer.close()
        shard_files = list(Path(tmp).glob("shard-*.tar"))
        assert len(shard_files) == 1


def test_writer_rolls_over_at_shard_size():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=3)
        for i in range(7):
            writer.write(_bundle(i))
        writer.close()
        shard_files = sorted(Path(tmp).glob("shard-*.tar"))
        # 3, 3, 1 -> 3 shards
        assert len(shard_files) == 3


def test_read_shard_bundles_matches_written():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        originals = [_bundle(i) for i in range(5)]
        for b in originals:
            writer.write(b)
        writer.close()
        readback = list(read_shard_bundles(Path(tmp)))
        assert len(readback) == 5
        ids_in = sorted(b.metadata.tile_id for b in originals)
        ids_out = sorted(b.metadata.tile_id for b in readback)
        assert ids_in == ids_out


def test_writer_writes_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=5)
        for i in range(3):
            writer.write(_bundle(i))
        writer.close()
        manifest = Path(tmp) / "manifest.json"
        assert manifest.exists()

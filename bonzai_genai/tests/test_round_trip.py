"""End-to-end pipeline round-trip on synthetic data:
synthetic geometry → raster + tokens → bundle → shard → bundle → tokens decoded → IoU sanity.
"""
import tempfile
from pathlib import Path

import numpy as np

from bonzai_genai.data.rasteriser import CH_ALL_ROADS, CH_BUILDINGS, rasterise
from bonzai_genai.data.shard_writer import ShardWriter, read_shard_bundles
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata
from bonzai_genai.synth.procedural import generate_synthetic_tile
from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokeniser import Tokeniser


def test_full_pipeline_synthetic_round_trip():
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)

    # 1. Generate synthetic geometry
    geom = generate_synthetic_tile(seed=7)
    assert len(geom.roads) > 0
    assert len(geom.buildings) > 0

    # 2. Rasterise
    raster = rasterise(geom)
    assert raster[CH_BUILDINGS].sum() > 0
    assert raster[CH_ALL_ROADS].sum() > 0

    # 3. Tokenise
    tokens = tokeniser.encode(geom)
    assert len(tokens) > 0

    # 4. Bundle + write to shard
    metadata = TileMetadata(
        tile_id="SYN-0",
        sw_lat=0.0, sw_lon=0.0,
        country="SYN", koppen="N/A",
        density_bucket="urban", primary_land_use="residential",
    )
    bundle = TileBundle(raster=raster, tokens=tokens, metadata=metadata)

    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        writer.write(bundle)
        writer.close()

        # 5. Read back from shard
        readback = list(read_shard_bundles(Path(tmp)))
        assert len(readback) == 1
        rb = readback[0]

        # 6. Round-trip raster
        np.testing.assert_array_equal(rb.raster, raster)

        # 7. Round-trip tokens decode → re-rasterise → IoU check
        decoded = tokeniser.decode(rb.tokens)
        decoded_raster = rasterise(decoded)
        b_orig = raster[CH_BUILDINGS] > 0
        b_dec = decoded_raster[CH_BUILDINGS] > 0
        intersection = np.logical_and(b_orig, b_dec).sum()
        union = np.logical_or(b_orig, b_dec).sum()
        iou = intersection / max(union, 1)
        # 4-metre quantisation on 30+ metre buildings should give IoU > 0.85
        assert iou > 0.85, f"building IoU = {iou:.3f}"

"""Tests for the TileBundle dataclass and (de)serialisation."""
import numpy as np
import pytest

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata


def _example_bundle() -> TileBundle:
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    raster[0, 100, 100] = 1.0
    tokens = [0, 4, 1]
    meta = TileMetadata(
        tile_id="LU-12-34",
        sw_lat=49.5,
        sw_lon=6.0,
        country="LU",
        koppen="Cfb",
        density_bucket="urban",
        primary_land_use="residential",
    )
    return TileBundle(raster=raster, tokens=tokens, metadata=meta)


def test_bundle_construction_validates_shape():
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    bundle = TileBundle(
        raster=raster,
        tokens=[1, 2],
        metadata=TileMetadata(
            tile_id="x", sw_lat=0.0, sw_lon=0.0,
            country="LU", koppen="Cfb", density_bucket="urban",
            primary_land_use="residential",
        ),
    )
    assert bundle.raster.shape == (NUM_CHANNELS, RASTER_PX, RASTER_PX)

    with pytest.raises(ValueError):
        TileBundle(raster=np.zeros((1, 1, 1), dtype=np.float32), tokens=[], metadata=bundle.metadata)


def test_metadata_to_json_roundtrip():
    meta = _example_bundle().metadata
    s = meta.to_json()
    parsed = TileMetadata.from_json(s)
    assert parsed == meta


def test_bundle_to_dict_keys():
    bundle = _example_bundle()
    d = bundle.to_dict()
    assert set(d.keys()) == {"raster.npy", "tokens.json", "metadata.json"}


def test_bundle_to_dict_values_are_bytes():
    bundle = _example_bundle()
    d = bundle.to_dict()
    for v in d.values():
        assert isinstance(v, bytes)


def test_bundle_from_dict_roundtrip():
    bundle = _example_bundle()
    d = bundle.to_dict()
    restored = TileBundle.from_dict(d)
    np.testing.assert_array_equal(restored.raster, bundle.raster)
    assert restored.tokens == bundle.tokens
    assert restored.metadata == bundle.metadata

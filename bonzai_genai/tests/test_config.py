"""Tests for the global config constants."""
from bonzai_genai.config import (
    TILE_SIDE_M,
    RASTER_PX,
    METRES_PER_PX,
    COORD_BINS,
    NUM_CHANNELS,
    CHANNEL_NAMES,
)


def test_tile_dimensions_are_consistent():
    assert TILE_SIDE_M == 2048
    assert RASTER_PX == 512
    assert METRES_PER_PX == TILE_SIDE_M / RASTER_PX  # 4
    assert METRES_PER_PX == 4.0


def test_coord_bins_match_raster_resolution():
    assert COORD_BINS == 512
    assert COORD_BINS == RASTER_PX  # one bin per raster pixel


def test_channel_layout_has_nine_channels():
    assert NUM_CHANNELS == 9
    assert len(CHANNEL_NAMES) == 9


def test_channel_names_are_unique_strings():
    assert len(set(CHANNEL_NAMES)) == 9
    assert all(isinstance(name, str) for name in CHANNEL_NAMES)


def test_channel_names_match_spec_order():
    expected = [
        "all_roads",
        "major_roads",
        "mid_roads",
        "minor_roads",
        "buildings",
        "building_density",
        "water",
        "green",
        "urban",
    ]
    assert list(CHANNEL_NAMES) == expected

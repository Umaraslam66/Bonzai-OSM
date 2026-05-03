"""Tests for tile sampling on real OSM data."""
from pathlib import Path

import pytest

from bonzai_genai.data.sampling import (
    extract_tile_geometry_from_osm,
    iter_tile_centres,
)

SG_PBF = Path(__file__).resolve().parents[1] / "data" / "malaysia-singapore-brunei-latest.osm.pbf"
SKIP_REAL = pytest.mark.skipif(
    not SG_PBF.exists(),
    reason="Singapore-area PBF not downloaded; see Task 14 step 1",
)


def test_iter_tile_centres_returns_grid_inside_bbox():
    """Sample on Singapore's bbox; count should match expected grid size."""
    centres = list(iter_tile_centres(
        sw_lat=1.20, sw_lon=103.60, ne_lat=1.48, ne_lon=104.05,
    ))
    # bbox ≈ 0.28° × 0.45° at 1.3° lat → ~31 km × 50 km / (2 km)² ≈ 400 tiles
    assert 300 <= len(centres) <= 500
    for lat, lon in centres:
        assert 1.20 <= lat <= 1.48
        assert 103.60 <= lon <= 104.05


@SKIP_REAL
def test_extract_tile_geometry_from_sg_pbf_returns_some_features():
    """One real tile from central Singapore should have buildings + roads."""
    # Marina Bay area, ~1.28 N, 103.85 E
    geom = extract_tile_geometry_from_osm(SG_PBF, sw_lat=1.275, sw_lon=103.845)
    assert len(geom.roads) > 0
    assert len(geom.buildings) > 0

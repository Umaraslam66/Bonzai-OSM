"""Tests for vector → 9-channel raster."""
import numpy as np

from bonzai_genai.config import METRES_PER_PX, NUM_CHANNELS, RASTER_PX
from bonzai_genai.data.rasteriser import rasterise
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
)


def test_empty_tile_yields_zero_raster():
    raster = rasterise(TileGeometry())
    assert raster.shape == (NUM_CHANNELS, RASTER_PX, RASTER_PX)
    assert raster.dtype == np.float32
    assert raster.sum() == 0.0


def test_road_paints_into_road_channels():
    geom = TileGeometry(roads=[
        Road("road_class=motorway", [(0.0, 1024.0), (2048.0 - METRES_PER_PX, 1024.0)]),
    ])
    raster = rasterise(geom)
    assert raster[0].sum() > 0   # all_roads
    assert raster[1].sum() > 0   # major_roads
    assert raster[2].sum() == 0  # mid_roads
    assert raster[3].sum() == 0  # minor_roads


def test_residential_paints_only_into_minor_roads():
    geom = TileGeometry(roads=[
        Road("road_class=residential", [(0.0, 1024.0), (2044.0, 1024.0)]),
    ])
    raster = rasterise(geom)
    assert raster[0].sum() > 0   # all_roads
    assert raster[3].sum() > 0   # minor_roads
    assert raster[1].sum() == 0
    assert raster[2].sum() == 0


def test_building_paints_into_building_and_density_channels():
    geom = TileGeometry(buildings=[
        Building("building_class=residential", "height=10m",
                 [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]),
    ])
    raster = rasterise(geom)
    assert raster[4].sum() > 0
    assert raster[5].sum() > raster[4].sum()  # blur spreads area


def test_water_paints_into_water_channel():
    geom = TileGeometry(land=[
        LandPolygon("water_class=lake", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[6].sum() > 0
    assert raster[7].sum() == 0
    assert raster[8].sum() == 0


def test_park_paints_into_green_channel():
    geom = TileGeometry(land=[
        LandPolygon("land_class=park", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[7].sum() > 0


def test_residential_landuse_paints_into_urban_channel():
    geom = TileGeometry(land=[
        LandPolygon("land_class=residential", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[8].sum() > 0


def test_pois_do_not_paint_raster():
    geom = TileGeometry(pois=[POI("poi=cafe", (1024.0, 1024.0))])
    raster = rasterise(geom)
    assert raster.sum() == 0  # POIs live only in vector tokens

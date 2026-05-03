"""Vector geometry → 9-channel raster.

Uses Pillow ImageDraw for line/polygon rasterisation (vectorised C
backend), then SciPy for the density blur on channel 5.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from bonzai_genai.config import (
    BUILDING_DENSITY_SIGMA_PX,
    METRES_PER_PX,
    NUM_CHANNELS,
    RASTER_PX,
)
from bonzai_genai.vocab.tokeniser import LandPolygon, Road, TileGeometry

# Channel index constants (mirrors config.CHANNEL_NAMES)
CH_ALL_ROADS = 0
CH_MAJOR_ROADS = 1
CH_MID_ROADS = 2
CH_MINOR_ROADS = 3
CH_BUILDINGS = 4
CH_BUILDING_DENSITY = 5
CH_WATER = 6
CH_GREEN = 7
CH_URBAN = 8

MAJOR_CLASSES = {"motorway", "trunk", "primary"}
MID_CLASSES = {"secondary", "tertiary"}
MINOR_CLASSES = {
    "residential", "service", "living_street", "pedestrian", "cycleway",
    "footway", "path", "track", "unclassified", "busway", "bus_guideway",
    "construction", "escape", "raceway", "rest_area", "road", "steps",
    "emergency_bay", "bridleway",
}

WATER_PREFIX = "water_class="
GREEN_LAND_CLASSES = {
    "park", "forest", "meadow", "wood", "grass", "farmland", "farmyard",
    "orchard", "vineyard", "cemetery", "allotments", "recreation_ground",
    "playground", "golf_course", "garden", "village_green",
    "greenhouse_horticulture", "heath", "scrub",
}
URBAN_LAND_CLASSES = {
    "residential", "commercial", "industrial", "retail", "construction",
    "brownfield", "greenfield", "military", "education", "religious",
    "leisure",
}


def _metres_to_px(coord_m: float) -> float:
    return coord_m / METRES_PER_PX


def _polyline_px(polyline: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(_metres_to_px(x), _metres_to_px(y)) for x, y in polyline]


def _classify_road(road: Road) -> int | None:
    if not road.class_name.startswith("road_class="):
        return None
    name = road.class_name.split("=", 1)[1]
    if name in MAJOR_CLASSES:
        return CH_MAJOR_ROADS
    if name in MID_CLASSES:
        return CH_MID_ROADS
    if name in MINOR_CLASSES:
        return CH_MINOR_ROADS
    return None


def _land_target_channel(poly: LandPolygon) -> int | None:
    if poly.class_name.startswith(WATER_PREFIX):
        return CH_WATER
    if not poly.class_name.startswith("land_class="):
        return None
    name = poly.class_name.split("=", 1)[1]
    if name in GREEN_LAND_CLASSES:
        return CH_GREEN
    if name in URBAN_LAND_CLASSES:
        return CH_URBAN
    return None


def _draw_line(channel: np.ndarray, polyline_px: list[tuple[float, float]], width: int) -> None:
    if len(polyline_px) < 2:
        return
    img = Image.fromarray((channel * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    draw.line(polyline_px, fill=255, width=width, joint="curve")
    channel[:] = (np.array(img) > 0).astype(np.float32)


def _draw_polygon(channel: np.ndarray, vertices_px: list[tuple[float, float]]) -> None:
    if len(vertices_px) < 3:
        return
    img = Image.fromarray((channel * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices_px, fill=255)
    channel[:] = (np.array(img) > 0).astype(np.float32)


def rasterise(geom: TileGeometry) -> np.ndarray:
    """Convert TileGeometry to a (NUM_CHANNELS, RASTER_PX, RASTER_PX) float32 array.

    All channels are binary [0, 1] except channel 5 (building_density)
    which is the Gaussian-blurred building footprint mask in [0, 1].
    """
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)

    # Roads
    for road in geom.roads:
        line_px = _polyline_px(road.polyline)
        _draw_line(raster[CH_ALL_ROADS], line_px, width=1)
        target = _classify_road(road)
        if target is not None:
            line_width = 2 if target == CH_MAJOR_ROADS else 1
            _draw_line(raster[target], line_px, width=line_width)

    # Buildings (binary mask)
    for b in geom.buildings:
        poly_px = _polyline_px(b.vertices)
        _draw_polygon(raster[CH_BUILDINGS], poly_px)

    # Building density: blur the binary mask
    if raster[CH_BUILDINGS].sum() > 0:
        blurred = gaussian_filter(raster[CH_BUILDINGS], sigma=BUILDING_DENSITY_SIGMA_PX)
        max_val = blurred.max()
        if max_val > 0:
            blurred /= max_val
        raster[CH_BUILDING_DENSITY] = blurred.astype(np.float32)

    # Land polygons
    for poly in geom.land:
        target = _land_target_channel(poly)
        if target is None:
            continue
        _draw_polygon(raster[target], _polyline_px(poly.vertices))

    # POIs deliberately not rasterised — they live in vector tokens.

    return raster

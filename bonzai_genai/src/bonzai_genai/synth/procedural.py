"""Procedural city generator — grid roads, axis-aligned rectangle buildings,
random POIs. Used for the Experiment 0 smoke test in Plan 6.
"""
from __future__ import annotations

import random

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    Building,
    POI,
    Road,
    TileGeometry,
)

# Fixed grid: 8 horizontal + 8 vertical roads, evenly spaced.
GRID_LINES = 8


def _grid_positions() -> list[float]:
    return [TILE_SIDE_M * (i + 1) / (GRID_LINES + 1) for i in range(GRID_LINES)]


def generate_synthetic_tile(seed: int = 0) -> TileGeometry:
    """Return a deterministic synthetic TileGeometry for the given seed."""
    rng = random.Random(seed)

    roads: list[Road] = []
    pos = _grid_positions()
    for y in pos:
        roads.append(Road(
            class_name="road_class=residential",
            polyline=[(0.0, y), (TILE_SIDE_M - 1.0, y)],
        ))
    for x in pos:
        roads.append(Road(
            class_name="road_class=residential",
            polyline=[(x, 0.0), (x, TILE_SIDE_M - 1.0)],
        ))

    # Buildings: place a rectangle in each interior grid cell, with jitter.
    buildings: list[Building] = []
    for i in range(GRID_LINES + 1):
        for j in range(GRID_LINES + 1):
            x_lo = TILE_SIDE_M * i / (GRID_LINES + 1)
            x_hi = TILE_SIDE_M * (i + 1) / (GRID_LINES + 1)
            y_lo = TILE_SIDE_M * j / (GRID_LINES + 1)
            y_hi = TILE_SIDE_M * (j + 1) / (GRID_LINES + 1)
            margin = 30.0
            x0 = x_lo + margin + rng.uniform(0, 20)
            x1 = x_hi - margin - rng.uniform(0, 20)
            y0 = y_lo + margin + rng.uniform(0, 20)
            y1 = y_hi - margin - rng.uniform(0, 20)
            if x1 <= x0 or y1 <= y0:
                continue
            buildings.append(Building(
                class_name="building_class=residential",
                height_name="height=10m",
                vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            ))

    # POIs: one per ~5 buildings.
    pois: list[POI] = []
    for b in rng.sample(buildings, k=max(1, len(buildings) // 5)):
        cx = sum(v[0] for v in b.vertices) / 4
        cy = sum(v[1] for v in b.vertices) / 4
        pois.append(POI(class_name="poi=cafe", point=(cx, cy)))

    return TileGeometry(land=[], roads=roads, buildings=buildings, pois=pois)

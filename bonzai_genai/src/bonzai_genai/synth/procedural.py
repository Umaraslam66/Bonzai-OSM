"""Procedural city generator — grid roads, axis-aligned rectangle buildings,
random POIs, plus varied diagonal roads in the "dense" mode and a sparser
"sparse" mode for the Phase 0b smoke harness.
"""
from __future__ import annotations

import math
import random

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    POI,
    Building,
    LandPolygon,
    Road,
    TileGeometry,
)

# Fixed grid: 8 horizontal + 8 vertical roads, evenly spaced.
GRID_LINES = 8


def _grid_positions() -> list[float]:
    return [TILE_SIDE_M * (i + 1) / (GRID_LINES + 1) for i in range(GRID_LINES)]


def _clamp(v: float) -> float:
    return max(0.0, min(TILE_SIDE_M - 1.0, v))


def _diag_road(rng: random.Random) -> Road:
    """A diagonal road at a random angle, fully inside the tile."""
    angle = rng.uniform(0.2, math.pi - 0.2)
    if abs(angle - math.pi / 2) < 0.1:
        angle += 0.3  # avoid near-vertical (already covered by grid)
    cx = rng.uniform(TILE_SIDE_M * 0.3, TILE_SIDE_M * 0.7)
    cy = rng.uniform(TILE_SIDE_M * 0.3, TILE_SIDE_M * 0.7)
    length = rng.uniform(400, 1500)
    dx = math.cos(angle) * length / 2
    dy = math.sin(angle) * length / 2
    cls = rng.choice(["secondary", "tertiary", "primary"])
    return Road(
        class_name=f"road_class={cls}",
        polyline=[(_clamp(cx - dx), _clamp(cy - dy)), (_clamp(cx + dx), _clamp(cy + dy))],
    )


def _land_polygon(rng: random.Random) -> LandPolygon:
    """Octagonal land polygon at a random interior position."""
    cx = rng.uniform(150.0, TILE_SIDE_M - 150.0)
    cy = rng.uniform(150.0, TILE_SIDE_M - 150.0)
    r = rng.uniform(80.0, 150.0)
    verts: list[tuple[float, float]] = []
    for i in range(8):
        theta = i * math.pi / 4
        verts.append((_clamp(cx + r * math.cos(theta)), _clamp(cy + r * math.sin(theta))))
    cls = rng.choice(["land_class=park", "water_class=lake"])
    return LandPolygon(class_name=cls, vertices=verts)


def generate_synthetic_tile(seed: int = 0, density: str = "dense") -> TileGeometry:
    """Return a deterministic synthetic TileGeometry for the given seed.

    ``density``:
      - ``"dense"``  — full 8x8 grid + 1-3 diagonal roads + ~64 buildings + 1-3 land polys + POIs
      - ``"sparse"`` — 2-3 short roads + ~10 buildings + 0 land polys
    """
    if density not in ("sparse", "dense"):
        raise ValueError(f"density must be 'sparse' or 'dense', got {density!r}")
    rng = random.Random(seed)

    roads: list[Road] = []
    buildings: list[Building] = []
    land: list[LandPolygon] = []
    pois: list[POI] = []

    if density == "dense":
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
        # 1-3 diagonal "highways" at varied angles
        for _ in range(rng.randint(1, 3)):
            roads.append(_diag_road(rng))
        # Buildings: place a rectangle in each interior grid cell, with jitter
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
                cls = rng.choice(["residential", "commercial", "office"])
                buildings.append(Building(
                    class_name=f"building_class={cls}",
                    height_name="height=10m",
                    vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                ))
        # 1-3 land polygons
        for _ in range(rng.randint(1, 3)):
            land.append(_land_polygon(rng))
        # POIs: one per ~5 buildings, at the building centroid
        for b in rng.sample(buildings, k=max(1, len(buildings) // 5)):
            cx = sum(v[0] for v in b.vertices) / 4
            cy = sum(v[1] for v in b.vertices) / 4
            cls_p = rng.choice(["cafe", "restaurant", "bank", "school"])
            pois.append(POI(class_name=f"poi={cls_p}", point=(cx, cy)))
    else:
        # Sparse: 2-3 short roads + ~10 buildings, no land
        n_roads = rng.randint(2, 3)
        for _ in range(n_roads):
            angle = rng.uniform(0, math.pi)
            x0 = rng.uniform(0.0, TILE_SIDE_M * 0.6)
            y0 = rng.uniform(0.0, TILE_SIDE_M * 0.6)
            length = rng.uniform(300.0, 800.0)
            x1 = _clamp(x0 + math.cos(angle) * length)
            y1 = _clamp(y0 + math.sin(angle) * length)
            roads.append(Road(
                class_name="road_class=residential",
                polyline=[(x0, y0), (x1, y1)],
            ))
        for _ in range(rng.randint(8, 15)):
            cx = rng.uniform(50.0, TILE_SIDE_M - 50.0)
            cy = rng.uniform(50.0, TILE_SIDE_M - 50.0)
            w = rng.uniform(15.0, 40.0)
            h = rng.uniform(15.0, 40.0)
            x0 = max(0.0, cx - w / 2)
            x1 = min(TILE_SIDE_M - 1.0, cx + w / 2)
            y0 = max(0.0, cy - h / 2)
            y1 = min(TILE_SIDE_M - 1.0, cy + h / 2)
            buildings.append(Building(
                class_name="building_class=residential",
                height_name="height=NA",
                vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            ))
        for _ in range(rng.randint(0, 2)):
            if not buildings:
                break
            b = rng.choice(buildings)
            x, y = b.vertices[0]
            pois.append(POI(class_name="poi=cafe", point=(x, y)))

    return TileGeometry(land=land, roads=roads, buildings=buildings, pois=pois)

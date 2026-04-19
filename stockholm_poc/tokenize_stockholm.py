"""
tokenize_stockholm.py
=====================

Stockholm PoC — convert a BBBike `.osm.pbf` extract into a 1D stream of
"spatial tokens" that an autoregressive Transformer can ingest.

Pipeline
--------
1. Parse PBF with `pyosmium`. Keep:
     * `building=*`    (multipolygons)  with optional `building:levels`
     * `highway=*`     (lines)          with optional `maxspeed`, `surface`
     * `amenity=*` / `shop=*`    on nodes     -> POI points
     * `waterway=*`    (lines)
     * `railway=*`     (lines)
     * `landuse=*` and `natural=water` (multipolygons) -> LANDUSE
2. Simplify polylines/polygons with Ramer-Douglas-Peucker
   (`shapely.simplify`, preserve_topology=False).
3. Compute the chunk bounding box from all surviving geometries, then
   for every geometry:
      anchor  = (ix, iy) on a chunk-local GRID_SIZE x GRID_SIZE grid,
                encoded as two tokens `<X_ix>` and `<Y_iy>`.
      moves   = sequence of (direction_bucket, distance_bucket_m) tokens
                between consecutive simplified vertices, projected to
                local meters via an equirectangular approximation.
      extras  = optional attribute tokens injected between the class tag
                and the anchor — LEVELS (buildings), SPEED+SURFACE (roads).
      POIs    = (X, Y) anchor only; no moves, no extras.
4. Assign a Hilbert-curve key to the centroid of every object using the
   *same* bounding box as the anchor grid, sort globally, flatten the
   sorted objects to one big list of string tokens.
5. Write Apache Parquet, chunked, ready for `datasets` streaming.
6. Print a vocabulary audit and the final total vocab size.

"Semantic compression" bucketing
--------------------------------
All free-form tag values are mapped onto small, fixed vocabularies
before token generation so global vocabulary stays bounded:
  * POIs     : ~20 categories across amenity + shop values
  * Landuse  : 9 categories (park, water, residential, …)
  * Waterway : 5 (river, stream, canal, drain, other)
  * Railway  : 5 (rail, tram, subway, narrow_gauge, other)
  * Levels   : 4 buckets (1-2, 3-5, 6-10, 11+)
  * Speed    : 3 buckets (low <40, mid 40-70, high >70 kph)
  * Surface  : 2 buckets (paved, unpaved)

Design notes
------------
- We intentionally avoid geopandas for portability. Geometries come out
  of pyosmium as WKB hex; we hydrate them with shapely directly.
- Long edges are split across multiple MOVE tokens: we snap the distance
  to the largest-fitting bucket in a log-ish ladder and emit N tokens
  for anything that overshoots. This bounds the per-token distance so
  the model sees a stable unit of motion.
- The Hilbert key uses a 21-bit-per-axis encoding (~1 m resolution at
  Stockholm latitudes). Hilbert preserves 2D locality better than the
  older Z-order key — adjacent Hilbert positions are always spatial
  neighbours, which is what we want for the autoregressive stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb as shp_wkb
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

try:
    import osmium
    import osmium.geom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyosmium is required (pip install 'osmium>=3.7,<4')") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_SIZE = 256

SIMPLIFY_TOLERANCE_DEG = 1e-5

COMPASS_BUCKETS = [
    (0.0, "E"),
    (45.0, "NE"),
    (90.0, "N"),
    (135.0, "NW"),
    (180.0, "W"),
    (-135.0, "SW"),
    (-90.0, "S"),
    (-45.0, "SE"),
]

DISTANCE_BUCKETS_M = [1000, 500, 250, 100, 50, 25, 15, 10, 5]

CHUNK_SIZE = 10_000

logger = logging.getLogger("tokenize_stockholm")


# ---------------------------------------------------------------------------
# Tag / attribute bucketing
# ---------------------------------------------------------------------------

HIGHWAY_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "living_street",
    "pedestrian", "track", "path", "footway", "cycleway", "steps",
}
BUILDING_CLASSES = {
    "yes", "residential", "house", "apartments", "detached",
    "commercial", "retail", "industrial", "office", "warehouse",
    "school", "church", "hospital", "garage", "hut", "shed",
    "public", "civic",
}

# POI categories — ~20 broad buckets over amenity=* / shop=* values.
# Every unknown amenity falls through to OTHER_AMENITY; every unknown
# shop falls through to RETAIL (since shop=* is already retail by
# definition). `name=*` is explicitly *not* read.
AMENITY_TO_CATEGORY: Dict[str, str] = {
    # FOOD_BEVERAGE
    "cafe": "FOOD_BEVERAGE", "restaurant": "FOOD_BEVERAGE",
    "fast_food": "FOOD_BEVERAGE", "bar": "FOOD_BEVERAGE",
    "pub": "FOOD_BEVERAGE", "ice_cream": "FOOD_BEVERAGE",
    "biergarten": "FOOD_BEVERAGE", "food_court": "FOOD_BEVERAGE",
    # EDUCATION
    "school": "EDUCATION", "kindergarten": "EDUCATION",
    "university": "EDUCATION", "college": "EDUCATION",
    "library": "EDUCATION", "language_school": "EDUCATION",
    "music_school": "EDUCATION", "driving_school": "EDUCATION",
    # HEALTHCARE
    "hospital": "HEALTHCARE", "clinic": "HEALTHCARE",
    "pharmacy": "HEALTHCARE", "doctors": "HEALTHCARE",
    "dentist": "HEALTHCARE", "veterinary": "HEALTHCARE",
    # FINANCIAL
    "bank": "FINANCIAL", "atm": "FINANCIAL", "bureau_de_change": "FINANCIAL",
    # CIVIC
    "townhall": "CIVIC", "courthouse": "CIVIC", "post_office": "CIVIC",
    "embassy": "CIVIC", "public_building": "CIVIC",
    # SAFETY
    "police": "SAFETY", "fire_station": "SAFETY", "prison": "SAFETY",
    # ENTERTAINMENT
    "cinema": "ENTERTAINMENT", "theatre": "ENTERTAINMENT",
    "nightclub": "ENTERTAINMENT", "arts_centre": "ENTERTAINMENT",
    "casino": "ENTERTAINMENT", "community_centre": "ENTERTAINMENT",
    "events_venue": "ENTERTAINMENT",
    # WORSHIP
    "place_of_worship": "WORSHIP", "grave_yard": "WORSHIP",
    # ACCOMMODATION (amenity-side; tourism=hotel lives under tourism=*)
    "hotel": "ACCOMMODATION", "hostel": "ACCOMMODATION",
    "guest_house": "ACCOMMODATION", "motel": "ACCOMMODATION",
    # TRANSPORT (non-parking)
    "bus_station": "TRANSPORT", "taxi": "TRANSPORT",
    "ferry_terminal": "TRANSPORT", "bicycle_rental": "TRANSPORT",
    # PARKING
    "parking": "PARKING", "bicycle_parking": "PARKING",
    "motorcycle_parking": "PARKING", "parking_entrance": "PARKING",
    "parking_space": "PARKING",
    # FUEL
    "fuel": "FUEL", "charging_station": "FUEL",
    # AUTOMOTIVE
    "car_rental": "AUTOMOTIVE", "car_wash": "AUTOMOTIVE",
    "car_sharing": "AUTOMOTIVE",
    # PUBLIC_AMENITY (benches, bins, info, etc. — low-signal but common)
    "bench": "PUBLIC_AMENITY", "drinking_water": "PUBLIC_AMENITY",
    "toilets": "PUBLIC_AMENITY", "waste_basket": "PUBLIC_AMENITY",
    "waste_disposal": "PUBLIC_AMENITY", "recycling": "PUBLIC_AMENITY",
    "shelter": "PUBLIC_AMENITY", "post_box": "PUBLIC_AMENITY",
    "telephone": "PUBLIC_AMENITY", "clock": "PUBLIC_AMENITY",
    "bbq": "PUBLIC_AMENITY", "fountain": "PUBLIC_AMENITY",
    "hunting_stand": "PUBLIC_AMENITY", "vending_machine": "PUBLIC_AMENITY",
    # RECREATION
    "playground": "RECREATION", "swimming_pool": "RECREATION",
    "sports_centre": "RECREATION", "fitness_centre": "RECREATION",
    "gym": "RECREATION",
}

# Shop values: almost everything is retail, but food-adjacent shops get
# their own RETAIL_GROCERY bucket so the model can distinguish daily-
# errand land-use patterns.
SHOP_TO_CATEGORY: Dict[str, str] = {
    "supermarket": "RETAIL_GROCERY", "convenience": "RETAIL_GROCERY",
    "bakery": "RETAIL_GROCERY", "butcher": "RETAIL_GROCERY",
    "greengrocer": "RETAIL_GROCERY", "alcohol": "RETAIL_GROCERY",
    "wine": "RETAIL_GROCERY", "seafood": "RETAIL_GROCERY",
    "cheese": "RETAIL_GROCERY", "deli": "RETAIL_GROCERY",
    "farm": "RETAIL_GROCERY",
}

# Landuse + natural=water classes.
LANDUSE_TO_CATEGORY: Dict[str, str] = {
    "park": "PARK", "grass": "PARK", "meadow": "PARK",
    "forest": "PARK", "orchard": "PARK", "recreation_ground": "PARK",
    "cemetery": "PARK", "village_green": "PARK", "allotments": "PARK",
    "garden": "PARK",
    "residential": "RESIDENTIAL",
    "commercial": "COMMERCIAL", "retail": "COMMERCIAL",
    "industrial": "INDUSTRIAL", "quarry": "INDUSTRIAL",
    "landfill": "INDUSTRIAL",
    "farmland": "FARMLAND", "farmyard": "FARMLAND", "vineyard": "FARMLAND",
    "construction": "CONSTRUCTION", "brownfield": "CONSTRUCTION",
    "greenfield": "CONSTRUCTION",
    "education": "INSTITUTIONAL", "religious": "INSTITUTIONAL",
    "military": "INSTITUTIONAL",
    "basin": "WATER", "reservoir": "WATER", "pond": "WATER",
}
# We also accept natural=wood/scrub as PARK, and natural=water as WATER.
NATURAL_TO_CATEGORY: Dict[str, str] = {
    "water": "WATER", "wood": "PARK", "scrub": "PARK",
    "wetland": "PARK", "heath": "PARK",
}

WATERWAY_TO_CATEGORY: Dict[str, str] = {
    "river": "RIVER", "riverbank": "RIVER",
    "stream": "STREAM", "tidal_channel": "STREAM",
    "canal": "CANAL",
    "drain": "DRAIN", "ditch": "DRAIN",
}
RAILWAY_TO_CATEGORY: Dict[str, str] = {
    "rail": "RAIL",
    "tram": "TRAM",
    "subway": "SUBWAY", "light_rail": "SUBWAY",
    "narrow_gauge": "NARROW_GAUGE", "monorail": "NARROW_GAUGE",
    "funicular": "NARROW_GAUGE",
}

PAVED_SURFACES = {
    "paved", "asphalt", "concrete", "concrete:plates", "concrete:lanes",
    "paving_stones", "sett", "cobblestone", "metal", "wood_planks",
}
UNPAVED_SURFACES = {
    "unpaved", "gravel", "fine_gravel", "pebblestone", "compacted",
    "dirt", "ground", "sand", "earth", "grass", "mud", "wood",
}

_INT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _bucket_highway(tag: Optional[str]) -> str:
    if not tag:
        return "OTHER"
    tag = tag.lower()
    return tag.upper() if tag in HIGHWAY_CLASSES else "OTHER"


def _bucket_building(tag: Optional[str]) -> str:
    if not tag:
        return "OTHER"
    tag = tag.lower()
    return tag.upper() if tag in BUILDING_CLASSES else "OTHER"


def _bucket_poi(amenity: Optional[str], shop: Optional[str]) -> Optional[str]:
    """Return the POI category token body, or None if neither tag is set."""
    if amenity:
        key = amenity.lower()
        return AMENITY_TO_CATEGORY.get(key, "OTHER_AMENITY")
    if shop:
        key = shop.lower()
        return SHOP_TO_CATEGORY.get(key, "RETAIL")
    return None


def _bucket_landuse(landuse: Optional[str], natural: Optional[str]) -> Optional[str]:
    if landuse:
        return LANDUSE_TO_CATEGORY.get(landuse.lower(), "OTHER")
    if natural:
        return NATURAL_TO_CATEGORY.get(natural.lower())
    return None


def _bucket_waterway(tag: Optional[str]) -> str:
    if not tag:
        return "OTHER"
    return WATERWAY_TO_CATEGORY.get(tag.lower(), "OTHER")


def _bucket_railway(tag: Optional[str]) -> str:
    if not tag:
        return "OTHER"
    return RAILWAY_TO_CATEGORY.get(tag.lower(), "OTHER")


def _bucket_levels(raw: Optional[str]) -> Optional[str]:
    """Bucket building:levels into 4 tokens."""
    if not raw:
        return None
    match = _INT_RE.search(str(raw))
    if match is None:
        return None
    try:
        n = int(float(match.group(0)))
    except ValueError:
        return None
    if n < 1:
        return None
    if n <= 2:
        return "<LEVELS_1_2>"
    if n <= 5:
        return "<LEVELS_3_5>"
    if n <= 10:
        return "<LEVELS_6_10>"
    return "<LEVELS_11_PLUS>"


def _bucket_speed(raw: Optional[str]) -> Optional[str]:
    """Bucket maxspeed into <SPEED_LOW|MID|HIGH>, normalising units."""
    if not raw:
        return None
    raw_str = str(raw).lower()
    match = _INT_RE.search(raw_str)
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    # Convert mph to kph if the unit is explicit.
    if "mph" in raw_str:
        value *= 1.609
    kph = int(round(value))
    if kph < 40:
        return "<SPEED_LOW>"
    if kph <= 70:
        return "<SPEED_MID>"
    return "<SPEED_HIGH>"


def _bucket_surface(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = str(raw).lower()
    if key in PAVED_SURFACES:
        return "<SURFACE_PAVED>"
    if key in UNPAVED_SURFACES:
        return "<SURFACE_UNPAVED>"
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _equirectangular_meters(
    lon_ref: float, lat_ref: float, lon: float, lat: float
) -> Tuple[float, float]:
    """Project (lon, lat) to local meters relative to (lon_ref, lat_ref)."""
    lat_rad = math.radians(lat_ref)
    dx = (lon - lon_ref) * 111_319.49 * math.cos(lat_rad)
    dy = (lat - lat_ref) * 111_319.49
    return dx, dy


def _bearing_bucket(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(dy, dx))
    best = None
    best_delta = 360.0
    for center, name in COMPASS_BUCKETS:
        delta = abs(((angle - center) + 180.0) % 360.0 - 180.0)
        if delta < best_delta:
            best_delta = delta
            best = name
    assert best is not None
    return best


def _split_distance(meters: float) -> List[int]:
    out: List[int] = []
    remaining = meters
    smallest = DISTANCE_BUCKETS_M[-1]
    while remaining >= smallest:
        for bucket in DISTANCE_BUCKETS_M:
            if remaining >= bucket:
                out.append(bucket)
                remaining -= bucket
                break
        else:  # pragma: no cover
            break
    return out


def _iter_rings(geom: BaseGeometry) -> Iterator[Sequence[Tuple[float, float]]]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield list(geom.coords)
    elif isinstance(geom, Polygon):
        yield list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            yield list(part.exterior.coords)
    else:
        geoms = getattr(geom, "geoms", None)
        if geoms is None:
            return
        for child in geoms:
            yield from _iter_rings(child)


# ---------------------------------------------------------------------------
# Hilbert curve encoding for 2D centroid sort
# ---------------------------------------------------------------------------


_HILBERT_BITS = 21  # 2^21 -> ~2M cells per axis, ~1 m at Stockholm latitudes


def _hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    """Convert (x, y) integer coordinates to a Hilbert curve distance.

    `n` must be a power of two. Adapted from the standard reference
    implementation (Skilling 2004 / Wikipedia's Hilbert curve article).
    Hilbert preserves locality better than Morton/Z-order: adjacent
    positions along the curve are always spatial neighbours.
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_key(
    lon: float,
    lat: float,
    bounds: Tuple[float, float, float, float],
) -> int:
    min_lon, min_lat, max_lon, max_lat = bounds
    nx = (lon - min_lon) / max(max_lon - min_lon, 1e-12)
    ny = (lat - min_lat) / max(max_lat - min_lat, 1e-12)
    n = 1 << _HILBERT_BITS
    ix = max(0, min(n - 1, int(nx * n)))
    iy = max(0, min(n - 1, int(ny * n)))
    return _hilbert_xy_to_d(n, ix, iy)


# ---------------------------------------------------------------------------
# Core per-object tokenization
# ---------------------------------------------------------------------------


@dataclass
class _RawObject:
    """Simplified geometry + classification + attribute tokens.

    `extra_tokens` are injected between the <KIND_START>+<TAG_*> header
    and the first <X_*><Y_*> anchor — so e.g. a building might read
    [<BUILDING_START>, <TAG_RESIDENTIAL>, <LEVELS_3_5>, <X_42>, <Y_7>, ...].
    """
    geom: BaseGeometry
    kind: str          # BUILDING | ROAD | POI | LANDUSE | WATERWAY | RAILWAY
    tag_token: str
    extra_tokens: List[str] = field(default_factory=list)


@dataclass
class TokenizedObject:
    kind: str
    tag_token: str
    tokens: List[str]
    centroid_lon: float
    centroid_lat: float


def anchor_tokens(
    lon: float,
    lat: float,
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """Map an absolute (lon, lat) onto the chunk-local 256x256 grid and
    return the two-token form `[<X_ix>, <Y_iy>]`.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    nx = (lon - min_lon) / max(max_lon - min_lon, 1e-12)
    ny = (lat - min_lat) / max(max_lat - min_lat, 1e-12)
    ix = max(0, min(GRID_SIZE - 1, int(nx * GRID_SIZE)))
    iy = max(0, min(GRID_SIZE - 1, int(ny * GRID_SIZE)))
    return [f"<X_{ix}>", f"<Y_{iy}>"]


def tokenize_ring(
    coords: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], List[str]]:
    if not coords:
        return ((0.0, 0.0), [])

    dedup: List[Tuple[float, float]] = [coords[0]]
    for pt in coords[1:]:
        if pt != dedup[-1]:
            dedup.append(pt)

    lon0, lat0 = dedup[0]

    move_tokens: List[str] = []
    prev_lon, prev_lat = lon0, lat0
    for lon, lat in dedup[1:]:
        dx, dy = _equirectangular_meters(prev_lon, prev_lat, lon, lat)
        dist = math.hypot(dx, dy)
        if dist < DISTANCE_BUCKETS_M[-1]:
            prev_lon, prev_lat = lon, lat
            continue
        bearing = _bearing_bucket(dx, dy)
        for bucket in _split_distance(dist):
            move_tokens.append(f"<MOVE_{bearing}_{bucket}M>")
        prev_lon, prev_lat = lon, lat

    return ((lon0, lat0), move_tokens)


def tokenize_row(
    raw: _RawObject,
    bounds: Tuple[float, float, float, float],
) -> Optional[TokenizedObject]:
    """Produce a TokenizedObject from one _RawObject.

    POIs are point features — no rings, no moves, just anchor.
    Everything else walks rings and emits MOVE tokens per edge.
    """
    if raw.kind == "POI":
        if raw.geom is None or raw.geom.is_empty:
            return None
        tokens: List[str] = [f"<{raw.kind}_START>", raw.tag_token]
        tokens.extend(raw.extra_tokens)
        tokens.extend(anchor_tokens(raw.geom.x, raw.geom.y, bounds))
        tokens.append(f"<{raw.kind}_END>")
        return TokenizedObject(
            kind=raw.kind,
            tag_token=raw.tag_token,
            tokens=tokens,
            centroid_lon=float(raw.geom.x),
            centroid_lat=float(raw.geom.y),
        )

    rings = list(_iter_rings(raw.geom))
    if not rings:
        return None

    tokens = [f"<{raw.kind}_START>", raw.tag_token]
    tokens.extend(raw.extra_tokens)
    for i, ring in enumerate(rings):
        (lon0, lat0), moves = tokenize_ring(ring)
        if i > 0:
            tokens.append("<PART_SEP>")
        tokens.extend(anchor_tokens(lon0, lat0, bounds))
        tokens.extend(moves)
    tokens.append(f"<{raw.kind}_END>")

    centroid = raw.geom.centroid
    return TokenizedObject(
        kind=raw.kind,
        tag_token=raw.tag_token,
        tokens=tokens,
        centroid_lon=float(centroid.x),
        centroid_lat=float(centroid.y),
    )


# ---------------------------------------------------------------------------
# PBF parsing
# ---------------------------------------------------------------------------


class _OSMCollector(osmium.SimpleHandler):
    """pyosmium handler that accumulates every target feature class with
    its tag attributes. Geometries come out as shapely objects; numeric
    attributes (levels/speed/surface) are retained as raw strings and
    bucketed downstream.

    Buckets:
        * nodes (amenity / shop)           -> pois
        * ways  (highway)                  -> roads
        * ways  (waterway)                 -> waterways
        * ways  (railway)                  -> railways
        * areas (building)                 -> buildings
        * areas (landuse | natural=water)  -> landuses
    """

    def __init__(self) -> None:
        super().__init__()
        self._wkbf = osmium.geom.WKBFactory()
        self.pois: List[Tuple[float, float, str]] = []
        # (geom, highway_tag, maxspeed_raw, surface_raw)
        self.roads: List[Tuple[BaseGeometry, str, Optional[str], Optional[str]]] = []
        self.waterways: List[Tuple[BaseGeometry, str]] = []
        self.railways: List[Tuple[BaseGeometry, str]] = []
        # (geom, building_tag, building_levels_raw)
        self.buildings: List[Tuple[BaseGeometry, str, Optional[str]]] = []
        # (geom, landuse_bucket_key)
        self.landuses: List[Tuple[BaseGeometry, str]] = []

        self._n_node_fail = 0
        self._n_way_fail = 0
        self._n_area_fail = 0

    # -- node ---------------------------------------------------------------

    def node(self, n) -> None:
        tags = n.tags
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        if amenity is None and shop is None:
            return
        category = _bucket_poi(amenity, shop)
        if category is None:
            return
        try:
            loc = n.location
            lon, lat = loc.lon, loc.lat
        except Exception:
            self._n_node_fail += 1
            return
        self.pois.append((lon, lat, category))

    # -- way ----------------------------------------------------------------

    def way(self, w) -> None:
        tags = w.tags
        hw = tags.get("highway")
        ww = tags.get("waterway")
        rw = tags.get("railway")
        if hw is None and ww is None and rw is None:
            return
        try:
            wkb_hex = self._wkbf.create_linestring(w)
        except Exception:
            self._n_way_fail += 1
            return
        geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
        # Highway wins if both keys somehow set (rare).
        if hw is not None:
            self.roads.append(
                (geom, hw, tags.get("maxspeed"), tags.get("surface"))
            )
        elif ww is not None:
            self.waterways.append((geom, ww))
        elif rw is not None:
            self.railways.append((geom, rw))

    # -- area ---------------------------------------------------------------

    def area(self, a) -> None:
        tags = a.tags
        building = tags.get("building")
        landuse = tags.get("landuse")
        natural = tags.get("natural")

        want_building = building is not None
        want_landuse = (landuse is not None) or (natural == "water") or (natural in NATURAL_TO_CATEGORY)

        if not want_building and not want_landuse:
            return

        try:
            wkb_hex = self._wkbf.create_multipolygon(a)
        except Exception:
            self._n_area_fail += 1
            return
        geom = shp_wkb.loads(bytes.fromhex(wkb_hex))

        if want_building:
            self.buildings.append((geom, building, tags.get("building:levels")))
            return

        bucket = _bucket_landuse(landuse, natural)
        if bucket is None:
            return
        self.landuses.append((geom, bucket))


def load_pbf(pbf_path: str) -> _OSMCollector:
    """Parse the PBF via pyosmium and return the populated collector."""
    logger.info("loading PBF via pyosmium: %s", pbf_path)
    handler = _OSMCollector()
    handler.apply_file(pbf_path, locations=True, idx="flex_mem")
    logger.info(
        "loaded: %d buildings, %d roads, %d pois, %d landuses, %d waterways, %d railways",
        len(handler.buildings),
        len(handler.roads),
        len(handler.pois),
        len(handler.landuses),
        len(handler.waterways),
        len(handler.railways),
    )
    logger.info(
        "skipped (bad geometry): %d nodes, %d ways, %d areas",
        handler._n_node_fail, handler._n_way_fail, handler._n_area_fail,
    )
    return handler


# ---------------------------------------------------------------------------
# Simplification + raw-object assembly
# ---------------------------------------------------------------------------


def _simplify(geom: BaseGeometry) -> Optional[BaseGeometry]:
    """Ramer-Douglas-Peucker simplification, skipping points."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Point):
        return geom
    simplified = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
    if simplified.is_empty:
        return None
    return simplified


def build_raw_objects(collector: _OSMCollector) -> List[_RawObject]:
    """Apply simplification + tag bucketing to every collected feature
    and return a single flat list of _RawObject ready for tokenization.
    """
    out: List[_RawObject] = []

    logger.info("simplifying %d buildings", len(collector.buildings))
    for geom, tag, levels_raw in collector.buildings:
        g = _simplify(geom)
        if g is None:
            continue
        tag_token = f"<TAG_{_bucket_building(tag)}>"
        extras: List[str] = []
        level_tok = _bucket_levels(levels_raw)
        if level_tok:
            extras.append(level_tok)
        out.append(_RawObject(geom=g, kind="BUILDING", tag_token=tag_token, extra_tokens=extras))

    logger.info("simplifying %d roads", len(collector.roads))
    for geom, tag, maxspeed_raw, surface_raw in collector.roads:
        g = _simplify(geom)
        if g is None:
            continue
        tag_token = f"<TAG_{_bucket_highway(tag)}>"
        extras = []
        speed_tok = _bucket_speed(maxspeed_raw)
        if speed_tok:
            extras.append(speed_tok)
        surface_tok = _bucket_surface(surface_raw)
        if surface_tok:
            extras.append(surface_tok)
        out.append(_RawObject(geom=g, kind="ROAD", tag_token=tag_token, extra_tokens=extras))

    logger.info("packaging %d POIs", len(collector.pois))
    for lon, lat, category in collector.pois:
        out.append(_RawObject(
            geom=Point(lon, lat),
            kind="POI",
            tag_token=f"<TAG_{category}>",
        ))

    logger.info("simplifying %d landuse polygons", len(collector.landuses))
    for geom, bucket in collector.landuses:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g,
            kind="LANDUSE",
            tag_token=f"<TAG_{bucket}>",
        ))

    logger.info("simplifying %d waterways", len(collector.waterways))
    for geom, tag in collector.waterways:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g,
            kind="WATERWAY",
            tag_token=f"<TAG_{_bucket_waterway(tag)}>",
        ))

    logger.info("simplifying %d railways", len(collector.railways))
    for geom, tag in collector.railways:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g,
            kind="RAILWAY",
            tag_token=f"<TAG_{_bucket_railway(tag)}>",
        ))

    logger.info("total raw objects: %d", len(out))
    return out


# ---------------------------------------------------------------------------
# Pipeline stages: bbox, tokenize, sort
# ---------------------------------------------------------------------------


def compute_bounds(
    raws: Sequence[_RawObject],
) -> Tuple[float, float, float, float]:
    if not raws:
        raise ValueError("no geometries to bound")
    min_lon = min_lat = math.inf
    max_lon = max_lat = -math.inf
    for r in raws:
        mnx, mny, mxx, mxy = r.geom.bounds
        if mnx < min_lon:
            min_lon = mnx
        if mny < min_lat:
            min_lat = mny
        if mxx > max_lon:
            max_lon = mxx
        if mxy > max_lat:
            max_lat = mxy
    bounds = (min_lon, min_lat, max_lon, max_lat)
    logger.info(
        "chunk bbox: lon [%.5f, %.5f], lat [%.5f, %.5f]",
        bounds[0], bounds[2], bounds[1], bounds[3],
    )
    return bounds


def tokenize_objects(
    raws: Sequence[_RawObject],
    bounds: Tuple[float, float, float, float],
) -> List[TokenizedObject]:
    out: List[TokenizedObject] = []
    kind_counts: Counter[str] = Counter()
    for r in raws:
        obj = tokenize_row(r, bounds=bounds)
        if obj is None:
            continue
        out.append(obj)
        kind_counts[obj.kind] += 1
    logger.info("total tokenized objects: %d", len(out))
    logger.info("  by kind: %s", dict(kind_counts.most_common()))
    return out


def sort_objects_hilbert(
    objects: List[TokenizedObject],
    bounds: Tuple[float, float, float, float],
) -> List[TokenizedObject]:
    """Global Hilbert-curve sort, reusing the anchor-grid bbox."""
    if not objects:
        return objects
    keyed = [
        (hilbert_key(o.centroid_lon, o.centroid_lat, bounds), o)
        for o in objects
    ]
    keyed.sort(key=lambda kv: kv[0])
    return [o for _, o in keyed]


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------


def _object_batches(
    objects: Iterable[TokenizedObject], chunk_size: int
) -> Iterator[pa.RecordBatch]:
    schema = pa.schema([
        pa.field("kind", pa.string()),
        pa.field("tag_token", pa.string()),
        pa.field("centroid_lon", pa.float64()),
        pa.field("centroid_lat", pa.float64()),
        pa.field("n_tokens", pa.int32()),
        pa.field("tokens", pa.list_(pa.string())),
    ])

    buf: List[TokenizedObject] = []
    for obj in objects:
        buf.append(obj)
        if len(buf) >= chunk_size:
            yield _batch_from_buf(buf, schema)
            buf = []
    if buf:
        yield _batch_from_buf(buf, schema)


def _batch_from_buf(buf: List[TokenizedObject], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {
            "kind": [o.kind for o in buf],
            "tag_token": [o.tag_token for o in buf],
            "centroid_lon": [o.centroid_lon for o in buf],
            "centroid_lat": [o.centroid_lat for o in buf],
            "n_tokens": [len(o.tokens) for o in buf],
            "tokens": [o.tokens for o in buf],
        },
        schema=schema,
    )


def write_parquet(
    objects: List[TokenizedObject],
    out_path: str,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    logger.info("writing parquet to %s (chunk_size=%d)", out_path, chunk_size)
    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    total_tokens = 0
    try:
        for batch in _object_batches(objects, chunk_size):
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
            total_rows += batch.num_rows
            total_tokens += int(
                sum(len(t) for t in batch.column("tokens").to_pylist())
            )
    finally:
        if writer is not None:
            writer.close()
    logger.info("parquet written: %d rows, %d total tokens", total_rows, total_tokens)


# ---------------------------------------------------------------------------
# Vocab dump + audit
# ---------------------------------------------------------------------------


def _family_of(token: str) -> str:
    body = token.lstrip("<").rstrip(">")
    return body.split("_", 1)[0] if "_" in body else body


def dump_vocab(objects: Iterable[TokenizedObject], out_path: str) -> int:
    """Write a sorted, deduplicated vocabulary JSON from the corpus.
    Logs a family-level audit and returns the total unique token count.
    """
    vocab: set[str] = set()
    for obj in objects:
        vocab.update(obj.tokens)
    payload = {
        "size": len(vocab),
        "tokens": sorted(vocab),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("vocab dumped: %d unique tokens -> %s", len(vocab), out_path)

    families = Counter(_family_of(t) for t in vocab)
    logger.info("vocab audit (unique tokens per family):")
    for family, count in families.most_common():
        logger.info("  %-14s %d", family, count)

    return len(vocab)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the .osm.pbf extract")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directory to write tokens + vocab into")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Objects per Parquet row-group (default: {CHUNK_SIZE})")
    parser.add_argument("--vocab-name", default="stockholm_vocab.json")
    parser.add_argument("--output-name", default="stockholm_tokens.parquet")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--vocab-limit", type=int, default=1000,
                        help="Hard ceiling on unique tokens (default: 1000)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.isfile(args.input):
        logger.error("input PBF not found: %s", args.input)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: parse all target feature classes from the PBF.
    collector = load_pbf(args.input)

    # Stage 2: simplify + bucket + assemble _RawObjects.
    raws = build_raw_objects(collector)
    if not raws:
        logger.error("no geometries survived simplification; aborting")
        return 3

    # Stage 3: chunk-level bounding box (used for grid + Hilbert).
    bounds = compute_bounds(raws)

    # Stage 4: tokenize.
    objects = tokenize_objects(raws, bounds)

    # Stage 5: Hilbert-curve sort by centroid.
    objects = sort_objects_hilbert(objects, bounds)

    # Stage 6: persist parquet + vocab.
    parquet_path = os.path.join(args.output_dir, args.output_name)
    vocab_path = os.path.join(args.output_dir, args.vocab_name)
    write_parquet(objects, parquet_path, chunk_size=args.chunk_size)
    vocab_size = dump_vocab(objects, vocab_path)

    # Vocab ceiling check — the whole point of the bucketing pivot.
    status = "OK" if vocab_size <= args.vocab_limit else "OVER LIMIT"
    logger.info(
        "TOTAL VOCAB SIZE: %d (limit %d) -> %s",
        vocab_size, args.vocab_limit, status,
    )
    if vocab_size > args.vocab_limit:
        logger.error("vocabulary exceeds --vocab-limit; failing")
        return 4

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

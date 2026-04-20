"""
tokenize_stockholm.py
=====================

Stockholm PoC / global pipeline — convert a `.osm.pbf` extract into a
1D stream of Spatial-Foundation-Model tokens.

Major upgrades vs V2.1:
  * Geometry uses 1-meter Delta X/Y tokens (`<dx_*>`, `<dy_*>`) in
    [-32, +32] range, replacing the chunky `<MOVE_DIR_DIST>` pairs.
  * Tags now carry the OSM key in the token: `<TAG_AMENITY_HOSPITAL>`,
    `<TAG_SHOP_SUPERMARKET>`, `<TAG_BUILDING_HOUSE>`, ...
  * Richer attributes: `<SURFACE_ASPHALT>`, `<LIT_YES>`, `<LANES_2>`,
    `<ACCESS_PRIVATE>`, `<ONEWAY_YES>`, `<BRIDGE_YES>`, `<TUNNEL_YES>`,
    `<LEVELS_*>`, `<SPEED_*>`, `<HEIGHT_*>`.
  * New POI / LANDUSE sources: `leisure=*`, `tourism=*`,
    `public_transport=*`, `historic=*` alongside amenity/shop/natural.
  * New kind: `NATURAL_LINE` for coastlines/cliffs/tree_rows/ridges.
  * Macro-context prefix row (`<CONTEXT_START> <REGION_*> <CLIMATE_*>
    <DENSITY_*> <CONTEXT_END>`) injected at the start of the corpus so
    the model knows where / what it's drawing.

All vocabulary definitions live in `vocab_spec.py`; this file only
implements the *emission* logic. Run `generate_vocab_dict.py` to
produce `vocab_dictionary.md` from the spec.
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

import vocab_spec as VS

try:
    import osmium
    import osmium.geom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyosmium is required (pip install 'osmium>=3.7,<4')") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIMPLIFY_TOLERANCE_DEG = 1e-5      # ~1.1 m at Stockholm latitudes
CHUNK_SIZE = 10_000                # parquet row-group size
BBOX_PERCENTILE_LOW = 0.5
BBOX_PERCENTILE_HIGH = 99.5

logger = logging.getLogger("tokenize_stockholm")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]+")


def _sanitize(value: str) -> str:
    v = _SANITIZE_RE.sub("_", value.strip()).strip("_")
    return v.upper() if v else "OTHER"


def _to_lower(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    v = str(v).strip()
    return v.lower() if v else None


# ---------------------------------------------------------------------------
# Tag-token helpers (TAG_KEY_VALUE with bucketing)
# ---------------------------------------------------------------------------


# Pre-flattened sets for O(1) lookups per key.
_TAG_ALLOWED_SETS: Dict[str, set] = {
    k: set(vs) for k, vs in VS.TAG_ALLOWED_VALUES.items()
}


def tag_token(key: str, value: Optional[str]) -> str:
    """Map (key, raw_value) to `<TAG_KEY_VALUE>` or `<TAG_KEY_OTHER>`.

    Unknown values fall through to OTHER. Raw value is lower-cased
    before lookup so `Hospital` and `hospital` collide correctly.
    """
    key_upper = key.upper()
    v = _to_lower(value)
    if v is None:
        return f"<TAG_{key_upper}_OTHER>"
    if v in _TAG_ALLOWED_SETS.get(key, set()):
        return f"<TAG_{key_upper}_{_sanitize(v)}>"
    return f"<TAG_{key_upper}_OTHER>"


# ---------------------------------------------------------------------------
# Attribute bucketing
# ---------------------------------------------------------------------------


_INT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _surface_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    if v in VS.SURFACE_VALUES:
        return f"<SURFACE_{v.upper()}>"
    # Heuristic fold for common synonyms.
    if v in {"concrete:plates", "concrete:lanes"}:
        return "<SURFACE_CONCRETE>"
    if v in {"pebblestone", "fine_gravel"}:
        return "<SURFACE_GRAVEL>"
    if v in {"wood_planks"}:
        return "<SURFACE_WOOD>"
    return "<SURFACE_OTHER>"


def _lit_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    if v in {"yes", "true", "1", "lit"}:
        return "<LIT_YES>"
    if v in {"no", "false", "0"}:
        return "<LIT_NO>"
    if v in {"24/7", "24_7", "continuous"}:
        return "<LIT_24_7>"
    if v == "automatic":
        return "<LIT_AUTOMATIC>"
    if v in {"sunset-sunrise", "sunset_sunrise", "dusk-dawn"}:
        return "<LIT_SUNSET_SUNRISE>"
    return None


def _lanes_token(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    m = _INT_RE.search(str(raw))
    if not m:
        return None
    try:
        n = int(float(m.group(0)))
    except ValueError:
        return None
    if n < 1:
        return None
    if n >= 5:
        return "<LANES_5_PLUS>"
    return f"<LANES_{n}>"


def _access_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    if v in VS.ACCESS_VALUES:
        return f"<ACCESS_{v.upper()}>"
    return "<ACCESS_OTHER>"


def _oneway_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    if v in {"yes", "true", "1"}:
        return "<ONEWAY_YES>"
    if v in {"no", "false", "0"}:
        return "<ONEWAY_NO>"
    if v in {"-1", "reverse"}:
        return "<ONEWAY_REVERSE>"
    return None


def _bridge_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    return "<BRIDGE_YES>" if v in VS.BRIDGE_TRUTHY else None


def _tunnel_token(raw: Optional[str]) -> Optional[str]:
    v = _to_lower(raw)
    if v is None:
        return None
    return "<TUNNEL_YES>" if v in VS.TUNNEL_TRUTHY else None


def _levels_token(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    m = _INT_RE.search(str(raw))
    if not m:
        return None
    try:
        n = int(float(m.group(0)))
    except ValueError:
        return None
    if n < 1:
        return None
    if n <= 2: return "<LEVELS_1_2>"
    if n <= 5: return "<LEVELS_3_5>"
    if n <= 10: return "<LEVELS_6_10>"
    return "<LEVELS_11_PLUS>"


def _speed_token(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).lower()
    m = _INT_RE.search(s)
    if not m:
        return None
    try:
        value = float(m.group(0))
    except ValueError:
        return None
    if "mph" in s:
        value *= 1.609
    kph = int(round(value))
    if kph < 40: return "<SPEED_LOW>"
    if kph <= 70: return "<SPEED_MID>"
    return "<SPEED_HIGH>"


def _height_token(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    m = _INT_RE.search(str(raw))
    if not m:
        return None
    try:
        meters = float(m.group(0))
    except ValueError:
        return None
    if meters <= 0: return None
    if meters < 10: return "<HEIGHT_LOW>"
    if meters < 25: return "<HEIGHT_MID>"
    if meters < 75: return "<HEIGHT_HIGH>"
    return "<HEIGHT_TALL>"


def road_attribute_tokens(tags) -> List[str]:
    """Produce ordered attribute tokens for a highway way."""
    out: List[str] = []
    # Keep ordering stable for cleaner grammar.
    for tok in (
        _surface_token(tags.get("surface")),
        _lit_token(tags.get("lit")),
        _lanes_token(tags.get("lanes")),
        _access_token(tags.get("access")),
        _oneway_token(tags.get("oneway")),
        _bridge_token(tags.get("bridge")),
        _tunnel_token(tags.get("tunnel")),
        _speed_token(tags.get("maxspeed")),
    ):
        if tok is not None:
            out.append(tok)
    return out


def building_attribute_tokens(tags) -> List[str]:
    out: List[str] = []
    for tok in (
        _levels_token(tags.get("building:levels")),
        _height_token(tags.get("height")),
    ):
        if tok is not None:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _equirectangular_meters(
    lon_ref: float, lat_ref: float, lon: float, lat: float
) -> Tuple[float, float]:
    lat_rad = math.radians(lat_ref)
    dx = (lon - lon_ref) * 111_319.49 * math.cos(lat_rad)
    dy = (lat - lat_ref) * 111_319.49
    return dx, dy


def _split_dxdy(dx_m: float, dy_m: float, cap: int = VS.DELTA_RANGE) -> List[str]:
    """Split a per-edge meter delta into one or more (dx, dy) token
    pairs, each constrained to [-cap, +cap] on each axis.

    Zero-magnitude sub-pairs are fine and explicit (emits `<dx_0>`
    `<dy_N>` and vice-versa) so the model always sees an even number
    of tokens per edge.
    """
    tokens: List[str] = []
    remaining_x = int(round(dx_m))
    remaining_y = int(round(dy_m))
    if remaining_x == 0 and remaining_y == 0:
        return tokens
    while remaining_x != 0 or remaining_y != 0:
        step_x = max(-cap, min(cap, remaining_x))
        step_y = max(-cap, min(cap, remaining_y))
        tokens.append(f"<dx_{step_x}>")
        tokens.append(f"<dy_{step_y}>")
        remaining_x -= step_x
        remaining_y -= step_y
    return tokens


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


def tokenize_ring(
    coords: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], List[str]]:
    """Return (anchor_lonlat, dxdy_tokens) for one ring.

    Consecutive duplicate vertices are dropped. Edges with magnitude
    below 1 m are collapsed (still advance the cursor so drift doesn't
    accumulate).
    """
    if not coords:
        return ((0.0, 0.0), [])

    dedup: List[Tuple[float, float]] = [coords[0]]
    for pt in coords[1:]:
        if pt != dedup[-1]:
            dedup.append(pt)

    lon0, lat0 = dedup[0]
    delta_tokens: List[str] = []
    prev_lon, prev_lat = lon0, lat0
    for lon, lat in dedup[1:]:
        dx_m, dy_m = _equirectangular_meters(prev_lon, prev_lat, lon, lat)
        if abs(dx_m) < 0.5 and abs(dy_m) < 0.5:
            prev_lon, prev_lat = lon, lat
            continue
        delta_tokens.extend(_split_dxdy(dx_m, dy_m))
        prev_lon, prev_lat = lon, lat
    return ((lon0, lat0), delta_tokens)


# ---------------------------------------------------------------------------
# Z-order / Hilbert sort (unchanged — Hilbert better preserves 2D locality)
# ---------------------------------------------------------------------------


_HILBERT_BITS = 21


def _hilbert_xy_to_d(n: int, x: int, y: int) -> int:
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


def hilbert_key(lon: float, lat: float, bounds: Tuple[float, float, float, float]) -> int:
    min_lon, min_lat, max_lon, max_lat = bounds
    nx = (lon - min_lon) / max(max_lon - min_lon, 1e-12)
    ny = (lat - min_lat) / max(max_lat - min_lat, 1e-12)
    n = 1 << _HILBERT_BITS
    ix = max(0, min(n - 1, int(nx * n)))
    iy = max(0, min(n - 1, int(ny * n)))
    return _hilbert_xy_to_d(n, ix, iy)


# ---------------------------------------------------------------------------
# Anchor grid
# ---------------------------------------------------------------------------


def anchor_tokens(
    lon: float, lat: float,
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    min_lon, min_lat, max_lon, max_lat = bounds
    nx = (lon - min_lon) / max(max_lon - min_lon, 1e-12)
    ny = (lat - min_lat) / max(max_lat - min_lat, 1e-12)
    ix = max(0, min(VS.GRID_SIZE - 1, int(nx * VS.GRID_SIZE)))
    iy = max(0, min(VS.GRID_SIZE - 1, int(ny * VS.GRID_SIZE)))
    return [f"<X_{ix}>", f"<Y_{iy}>"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _RawObject:
    """One parsed feature ready for tokenization.

    `tag_key` + `tag_value` identify the primary OSM (key, value) of
    this feature (e.g. `("amenity", "hospital")` or
    `("building", "yes")`). `extra_tags` carries any extra OSM tags we
    want to inject as attribute tokens (surface, lit, lanes, ...).
    """
    geom: BaseGeometry
    kind: str
    tag_key: str
    tag_value: str
    extra_tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TokenizedObject:
    kind: str
    tag_token: str
    tokens: List[str]
    centroid_lon: float
    centroid_lat: float


def tokenize_row(
    raw: _RawObject,
    bounds: Tuple[float, float, float, float],
) -> Optional[TokenizedObject]:
    """Emit the full token stream for a single _RawObject."""
    tag_tok = tag_token(raw.tag_key, raw.tag_value)

    # POIs are point features — anchor only, no moves, no parts.
    if raw.kind == "POI":
        if raw.geom is None or raw.geom.is_empty:
            return None
        tokens: List[str] = [f"<{raw.kind}_START>", tag_tok]
        tokens.extend(anchor_tokens(raw.geom.x, raw.geom.y, bounds))
        tokens.append(f"<{raw.kind}_END>")
        return TokenizedObject(
            kind=raw.kind,
            tag_token=tag_tok,
            tokens=tokens,
            centroid_lon=float(raw.geom.x),
            centroid_lat=float(raw.geom.y),
        )

    rings = list(_iter_rings(raw.geom))
    if not rings:
        return None

    # Extra attribute tokens (road: surface/lit/lanes/..., building: levels/height)
    attr_tokens: List[str] = []
    if raw.kind == "ROAD" and raw.extra_tags:
        attr_tokens = road_attribute_tokens(raw.extra_tags)
    elif raw.kind == "BUILDING" and raw.extra_tags:
        attr_tokens = building_attribute_tokens(raw.extra_tags)

    tokens = [f"<{raw.kind}_START>", tag_tok]
    tokens.extend(attr_tokens)
    for i, ring in enumerate(rings):
        (lon0, lat0), deltas = tokenize_ring(ring)
        if i > 0:
            tokens.append("<PART_SEP>")
        tokens.extend(anchor_tokens(lon0, lat0, bounds))
        tokens.extend(deltas)
    tokens.append(f"<{raw.kind}_END>")

    centroid = raw.geom.centroid
    return TokenizedObject(
        kind=raw.kind,
        tag_token=tag_tok,
        tokens=tokens,
        centroid_lon=float(centroid.x),
        centroid_lat=float(centroid.y),
    )


# ---------------------------------------------------------------------------
# PBF parsing
# ---------------------------------------------------------------------------


_NATURAL_LINE_SET = set(VS.NATURAL_LINE_VALUES)
_POI_TAG_KEYS = tuple(VS.POI_TAG_KEYS)
_LANDUSE_TAG_KEYS = tuple(VS.LANDUSE_TAG_KEYS)


def _first_poi_tag(tags) -> Optional[Tuple[str, str]]:
    """Find the first (key, value) among POI-eligible keys. For
    `natural`, we only accept point-legal values (trees, peaks, etc.),
    not polygon values like `water` or `wood`.
    """
    for key in _POI_TAG_KEYS:
        v = tags.get(key)
        if v is None or v == "":
            continue
        if key == "natural":
            vl = v.lower()
            # Point-legal natural values: excludes area ones like `water`,
            # `wood`, `wetland`. Those come through the area handler.
            if vl in _NATURAL_LINE_SET:
                # coastline/cliff/ridge are lines, not nodes — skip here.
                continue
            if vl in {"water", "wood", "scrub", "wetland", "heath",
                      "grassland", "bare_rock", "sand", "beach",
                      "shingle", "scree", "fell", "reef", "bay",
                      "strait", "glacier", "mud"}:
                continue
        return (key, v)
    return None


class _OSMCollector(osmium.SimpleHandler):
    """Streams the PBF once and buckets features into per-kind lists.

    Each entry stores the primary (tag_key, tag_value) so the tokenizer
    can emit the correctly-keyed TAG token, plus a small dict of
    per-feature attributes (surface, lanes, levels, ...).
    """

    def __init__(self) -> None:
        super().__init__()
        self._wkbf = osmium.geom.WKBFactory()
        # POI: (lon, lat, tag_key, tag_value)
        self.pois: List[Tuple[float, float, str, str]] = []
        # ROAD: (geom, tag_value, attr_dict)
        self.roads: List[Tuple[BaseGeometry, str, Dict[str, str]]] = []
        # WATERWAY: (geom, tag_value)
        self.waterways: List[Tuple[BaseGeometry, str]] = []
        # RAILWAY: (geom, tag_value)
        self.railways: List[Tuple[BaseGeometry, str]] = []
        # NATURAL_LINE: (geom, tag_value) — tag_key is always "natural"
        self.natural_lines: List[Tuple[BaseGeometry, str]] = []
        # BUILDING: (geom, tag_value, attr_dict)
        self.buildings: List[Tuple[BaseGeometry, str, Dict[str, str]]] = []
        # LANDUSE: (geom, tag_key, tag_value)
        self.landuses: List[Tuple[BaseGeometry, str, str]] = []

        self._n_node_fail = 0
        self._n_way_fail = 0
        self._n_area_fail = 0

    # -- node ---------------------------------------------------------------

    def node(self, n) -> None:
        tags = n.tags
        found = _first_poi_tag(tags)
        if found is None:
            return
        key, value = found
        try:
            loc = n.location
            lon, lat = loc.lon, loc.lat
        except Exception:
            self._n_node_fail += 1
            return
        self.pois.append((lon, lat, key, value))

    # -- way ----------------------------------------------------------------

    def way(self, w) -> None:
        tags = w.tags
        hw = tags.get("highway")
        ww = tags.get("waterway")
        rw = tags.get("railway")
        nat = tags.get("natural")
        is_natural_line = nat is not None and nat.lower() in _NATURAL_LINE_SET

        if hw is None and ww is None and rw is None and not is_natural_line:
            return
        try:
            wkb_hex = self._wkbf.create_linestring(w)
        except Exception:
            self._n_way_fail += 1
            return
        geom = shp_wkb.loads(bytes.fromhex(wkb_hex))

        if hw is not None:
            # Keep a snapshot of useful road attributes.
            attrs: Dict[str, str] = {}
            for k in ("surface", "lit", "lanes", "access", "oneway",
                      "bridge", "tunnel", "maxspeed"):
                v = tags.get(k)
                if v not in (None, ""):
                    attrs[k] = v
            self.roads.append((geom, hw, attrs))
        elif ww is not None:
            self.waterways.append((geom, ww))
        elif rw is not None:
            self.railways.append((geom, rw))
        elif is_natural_line:
            self.natural_lines.append((geom, nat))

    # -- area ---------------------------------------------------------------

    def area(self, a) -> None:
        tags = a.tags
        building = tags.get("building")
        if building is not None:
            try:
                wkb_hex = self._wkbf.create_multipolygon(a)
            except Exception:
                self._n_area_fail += 1
                return
            geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
            attrs: Dict[str, str] = {}
            for k in ("building:levels", "height"):
                v = tags.get(k)
                if v not in (None, ""):
                    attrs[k] = v
            self.buildings.append((geom, building, attrs))
            return

        # Landuse-style polygon: first match in LANDUSE_TAG_KEYS wins.
        for key in _LANDUSE_TAG_KEYS:
            v = tags.get(key)
            if v is None or v == "":
                continue
            if key == "natural" and v.lower() in _NATURAL_LINE_SET:
                # Linear natural feature stored as area — skip (we take
                # the linestring form through `way`).
                continue
            try:
                wkb_hex = self._wkbf.create_multipolygon(a)
            except Exception:
                self._n_area_fail += 1
                return
            geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
            self.landuses.append((geom, key, v))
            return


def load_pbf(pbf_path: str) -> _OSMCollector:
    logger.info("loading PBF via pyosmium: %s", pbf_path)
    handler = _OSMCollector()
    handler.apply_file(pbf_path, locations=True, idx="flex_mem")
    logger.info(
        "loaded: %d buildings, %d roads, %d pois, %d landuses, "
        "%d waterways, %d railways, %d natural_lines",
        len(handler.buildings), len(handler.roads), len(handler.pois),
        len(handler.landuses), len(handler.waterways),
        len(handler.railways), len(handler.natural_lines),
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
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Point):
        return geom
    simplified = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
    if simplified.is_empty:
        return None
    return simplified


def build_raw_objects(collector: _OSMCollector) -> List[_RawObject]:
    out: List[_RawObject] = []

    logger.info("simplifying %d buildings", len(collector.buildings))
    for geom, tag_value, attrs in collector.buildings:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="BUILDING",
            tag_key="building", tag_value=tag_value, extra_tags=attrs,
        ))

    logger.info("simplifying %d roads", len(collector.roads))
    for geom, tag_value, attrs in collector.roads:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="ROAD",
            tag_key="highway", tag_value=tag_value, extra_tags=attrs,
        ))

    logger.info("packaging %d POIs", len(collector.pois))
    for lon, lat, key, value in collector.pois:
        out.append(_RawObject(
            geom=Point(lon, lat), kind="POI",
            tag_key=key, tag_value=value,
        ))

    logger.info("simplifying %d landuse polygons", len(collector.landuses))
    for geom, key, value in collector.landuses:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="LANDUSE",
            tag_key=key, tag_value=value,
        ))

    logger.info("simplifying %d waterways", len(collector.waterways))
    for geom, tag_value in collector.waterways:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="WATERWAY",
            tag_key="waterway", tag_value=tag_value,
        ))

    logger.info("simplifying %d railways", len(collector.railways))
    for geom, tag_value in collector.railways:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="RAILWAY",
            tag_key="railway", tag_value=tag_value,
        ))

    logger.info("simplifying %d natural lines", len(collector.natural_lines))
    for geom, tag_value in collector.natural_lines:
        g = _simplify(geom)
        if g is None:
            continue
        out.append(_RawObject(
            geom=g, kind="NATURAL_LINE",
            tag_key="natural", tag_value=tag_value,
        ))

    logger.info("total raw objects: %d", len(out))
    return out


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def compute_bounds(
    raws: Sequence[_RawObject],
    low_pct: float = BBOX_PERCENTILE_LOW,
    high_pct: float = BBOX_PERCENTILE_HIGH,
) -> Tuple[float, float, float, float]:
    if not raws:
        raise ValueError("no geometries to bound")
    lons = np.fromiter((r.geom.centroid.x for r in raws), dtype=np.float64, count=len(raws))
    lats = np.fromiter((r.geom.centroid.y for r in raws), dtype=np.float64, count=len(raws))
    min_lon = float(np.percentile(lons, low_pct))
    max_lon = float(np.percentile(lons, high_pct))
    min_lat = float(np.percentile(lats, low_pct))
    max_lat = float(np.percentile(lats, high_pct))
    bounds = (min_lon, min_lat, max_lon, max_lat)
    n_out_lon = int(((lons < min_lon) | (lons > max_lon)).sum())
    n_out_lat = int(((lats < min_lat) | (lats > max_lat)).sum())
    logger.info(
        "chunk bbox (p%.1f/%.1f): lon [%.5f, %.5f], lat [%.5f, %.5f]",
        low_pct, high_pct, bounds[0], bounds[2], bounds[1], bounds[3],
    )
    logger.info(
        "  centroids clamped to edge: %d lon, %d lat (%.2f%% of %d)",
        n_out_lon, n_out_lat,
        100.0 * max(n_out_lon, n_out_lat) / max(len(raws), 1), len(raws),
    )
    return bounds


def tokenize_objects(
    raws: Sequence[_RawObject],
    bounds: Tuple[float, float, float, float],
) -> List[TokenizedObject]:
    out: List[TokenizedObject] = []
    kind_counts: Counter = Counter()
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
    if not objects:
        return objects
    keyed = [
        (hilbert_key(o.centroid_lon, o.centroid_lat, bounds), o)
        for o in objects
    ]
    keyed.sort(key=lambda kv: kv[0])
    return [o for _, o in keyed]


# ---------------------------------------------------------------------------
# Macro-context row
# ---------------------------------------------------------------------------


def build_context_object(
    region: Optional[str],
    climate: Optional[str],
    density: Optional[str],
    bounds: Tuple[float, float, float, float],
) -> TokenizedObject:
    """Produce the synthetic `CONTEXT` row that gets prepended to the
    corpus. For planet-scale chunking, the three values come from H3
    cell metadata; for Stockholm PoC they come from CLI flags.
    """
    tokens: List[str] = ["<CONTEXT_START>"]
    if region:
        r = region.upper()
        if r in VS.REGIONS:
            tokens.append(f"<REGION_{r}>")
    if climate:
        c = climate.upper()
        if c in VS.CLIMATES:
            tokens.append(f"<CLIMATE_{c}>")
    if density:
        d = density.upper()
        if d in VS.DENSITIES:
            tokens.append(f"<DENSITY_{d}>")
    tokens.append("<CONTEXT_END>")

    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    return TokenizedObject(
        kind="CONTEXT",
        tag_token=f"<REGION_{(region or 'EUROPE').upper()}>",
        tokens=tokens,
        centroid_lon=cx,
        centroid_lat=cy,
    )


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------


def _object_batches(objects: Iterable[TokenizedObject], chunk_size: int) -> Iterator[pa.RecordBatch]:
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
    objects: List[TokenizedObject], out_path: str, chunk_size: int = CHUNK_SIZE,
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
    vocab: set[str] = set()
    for obj in objects:
        vocab.update(obj.tokens)
    payload = {"size": len(vocab), "tokens": sorted(vocab)}
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
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--vocab-name", default="stockholm_vocab.json")
    parser.add_argument("--output-name", default="stockholm_tokens.parquet")
    parser.add_argument("--vocab-limit", type=int, default=4096)
    parser.add_argument("--log-level", default="INFO")
    # Macro-context flags.
    parser.add_argument("--region", default=None,
                        help=f"One of {VS.REGIONS}")
    parser.add_argument("--climate", default=None,
                        help=f"One of {VS.CLIMATES}")
    parser.add_argument("--density", default=None,
                        help=f"One of {VS.DENSITIES}")
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

    # Stage 1: parse.
    collector = load_pbf(args.input)

    # Stage 2: simplify + assemble raw objects.
    raws = build_raw_objects(collector)
    if not raws:
        logger.error("no geometries survived simplification; aborting")
        return 3

    # Stage 3: robust bounding box.
    bounds = compute_bounds(raws)

    # Stage 4: tokenize.
    objects = tokenize_objects(raws, bounds)

    # Stage 5: Hilbert-curve sort by centroid.
    objects = sort_objects_hilbert(objects, bounds)

    # Stage 6: prepend the macro-context row.
    context_obj = build_context_object(
        region=args.region, climate=args.climate, density=args.density,
        bounds=bounds,
    )
    objects = [context_obj] + objects

    # Stage 7: persist.
    parquet_path = os.path.join(args.output_dir, args.output_name)
    vocab_path = os.path.join(args.output_dir, args.vocab_name)
    write_parquet(objects, parquet_path, chunk_size=args.chunk_size)
    vocab_size = dump_vocab(objects, vocab_path)

    status = "OK" if vocab_size <= args.vocab_limit else "OVER LIMIT"
    logger.info(
        "TOTAL VOCAB SIZE: %d (limit %d, spec-max %d) -> %s",
        vocab_size, args.vocab_limit, VS.vocabulary_size(), status,
    )
    if vocab_size > args.vocab_limit:
        logger.error("vocabulary exceeds --vocab-limit; failing")
        return 4
    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
tokenize_stockholm.py
=====================

Stockholm PoC — convert a BBBike `.osm.pbf` extract into a 1D stream of
"spatial tokens" that an autoregressive Transformer can ingest.

Pipeline
--------
1. Parse PBF with `pyosmium`; keep only `building=*` (polygons) and
   `highway=*` (lines).
2. Simplify polylines/polygons with Ramer-Douglas-Peucker
   (`shapely.simplify`, preserve_topology=False).
3. Compute the chunk bounding box from all surviving geometries, then
   for every geometry:
      anchor  = (ix, iy) on a chunk-local GRID_SIZE x GRID_SIZE grid,
                encoded as two tokens `<X_ix>` and `<Y_iy>`. This keeps
                the anchor vocabulary at exactly 2*GRID_SIZE tokens no
                matter how large the chunk gets, which is the key
                property we need when scaling from Stockholm to the
                planet in localised chunks.
      moves   = sequence of (direction_bucket, distance_bucket_m) tokens
                between consecutive simplified vertices, projected to
                local meters via an equirectangular approximation.
4. Assign a Morton (Z-order) key to the centroid of every object using
   the *same* bounding box as the anchor grid, sort globally, flatten
   the sorted objects to one big list of string tokens.
5. Write Apache Parquet, chunked, ready for `datasets` streaming.

The token alphabet is deliberately simple and self-describing so the
vocabulary file is reproducible from the Parquet output.

Design notes
------------
- We intentionally avoid geopandas for portability — pyrosm returns
  plain GeoDataFrames but we only need geometry + a couple of string
  fields, so we fall back to pure shapely where it keeps things simple.
- Long edges are split across multiple MOVE tokens: we snap the distance
  to the nearest bucket in a log-ish ladder and emit N copies of the
  max-bucket token for anything that overshoots. This bounds the per-
  token distance so the model sees a stable unit of motion.
- The Morton key uses a 21-bit-per-axis interleave (good to ~1 m
  resolution at Stockholm latitudes), which is overkill but makes the
  ordering deterministic and fully reversible for debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter

from shapely import wkb as shp_wkb
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

try:
    import osmium
    import osmium.geom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyosmium is required (pip install 'osmium>=3.7,<4')") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Local anchor grid: each chunk's bounding box is quantised into a
# GRID_SIZE x GRID_SIZE grid, and an object's first-vertex cell becomes
# two tokens `<X_ix>` and `<Y_iy>`. This caps the anchor vocabulary at
# exactly 2*GRID_SIZE tokens regardless of chunk size or resolution.
GRID_SIZE = 256

# Simplification tolerance in *degrees*. At Stockholm (~59 N) one degree
# of latitude is ~111 km; 1e-5 deg ≈ 1.1 m which is a good balance
# between corner preservation and token bloat.
SIMPLIFY_TOLERANCE_DEG = 1e-5

# 8-way compass. Bearings are CCW from east by convention in math, but
# the user-facing token reads more naturally as compass bearings, so we
# map (dx, dy) -> compass where N=+dy, E=+dx.
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

# Log-ish distance ladder in meters. An edge of length D is expressed as
# a greedy decomposition over this ladder (largest-first), which caps
# the maximum move distance the model has to generate and keeps the
# token vocabulary small. Last bucket doubles as a "long segment"
# primitive (repeated as needed).
DISTANCE_BUCKETS_M = [1000, 500, 250, 100, 50, 25, 15, 10, 5]

# Highway / building tag classes we keep. Everything else is rolled up
# into OTHER so the vocab stays bounded.
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

# Parquet chunk size (rows = objects). Small enough to stream friendly,
# big enough that row-group overhead stays negligible.
CHUNK_SIZE = 10_000

logger = logging.getLogger("tokenize_stockholm")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _equirectangular_meters(
    lon_ref: float, lat_ref: float, lon: float, lat: float
) -> Tuple[float, float]:
    """Project (lon, lat) to local meters relative to (lon_ref, lat_ref).

    Equirectangular approximation. Fine for per-object offsets, where
    the span is at most a few hundred meters.
    Returns (dx, dy) in meters, with +x east and +y north.
    """
    # 1 degree latitude ≈ 111_319.49 m; longitude shrinks by cos(lat).
    lat_rad = math.radians(lat_ref)
    dx = (lon - lon_ref) * 111_319.49 * math.cos(lat_rad)
    dy = (lat - lat_ref) * 111_319.49
    return dx, dy


def _bearing_bucket(dx: float, dy: float) -> str:
    """Snap a (dx, dy) vector to one of the 8 compass buckets."""
    angle = math.degrees(math.atan2(dy, dx))  # -180..180, 0 = east
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
    """Greedy decomposition of a meter distance onto DISTANCE_BUCKETS_M.

    Edges shorter than the smallest bucket are dropped entirely (they
    are noise after simplification). Long edges emit multiple tokens.
    """
    out: List[int] = []
    remaining = meters
    smallest = DISTANCE_BUCKETS_M[-1]
    while remaining >= smallest:
        for bucket in DISTANCE_BUCKETS_M:
            if remaining >= bucket:
                out.append(bucket)
                remaining -= bucket
                break
        else:  # pragma: no cover — loop invariant
            break
    return out


def _iter_rings(geom: BaseGeometry) -> Iterator[Sequence[Tuple[float, float]]]:
    """Yield (lon, lat) coordinate sequences for whatever kind of
    geometry we were handed. Building polygons become their exterior
    ring; MultiPolygons yield each part's exterior; LineStrings yield
    their coords as-is.
    """
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
        # GeometryCollection, MultiLineString, etc. — flatten children.
        geoms = getattr(geom, "geoms", None)
        if geoms is None:
            return
        for child in geoms:
            yield from _iter_rings(child)


# ---------------------------------------------------------------------------
# Z-order (Morton) encoding for 2D centroid sort
# ---------------------------------------------------------------------------


_MORTON_BITS = 21  # 21 bits per axis -> ~1 m at Stockholm latitudes


def _interleave_bits(x: int, y: int, bits: int = _MORTON_BITS) -> int:
    """Interleave the low `bits` of x and y into one Morton code."""
    result = 0
    for i in range(bits):
        result |= ((x >> i) & 1) << (2 * i)
        result |= ((y >> i) & 1) << (2 * i + 1)
    return result


def morton_key(lon: float, lat: float, bounds: Tuple[float, float, float, float]) -> int:
    """Z-order key for a centroid within a lon/lat bounding box.

    `bounds` is (min_lon, min_lat, max_lon, max_lat). The coordinate is
    normalised to [0, 1], scaled by 2**bits, clamped to integer range,
    then bit-interleaved.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    nx = (lon - min_lon) / max(max_lon - min_lon, 1e-12)
    ny = (lat - min_lat) / max(max_lat - min_lat, 1e-12)
    scale = (1 << _MORTON_BITS) - 1
    ix = max(0, min(scale, int(nx * scale)))
    iy = max(0, min(scale, int(ny * scale)))
    return _interleave_bits(ix, iy)


# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core per-object tokenization
# ---------------------------------------------------------------------------


@dataclass
class TokenizedObject:
    """One Stockholm object, ready to be globally sorted and serialized."""
    kind: str                # "BUILDING" or "ROAD"
    tag_token: str           # e.g. "<TAG_RESIDENTIAL>"
    tokens: List[str]        # full per-object token list
    centroid_lon: float
    centroid_lat: float


def anchor_tokens(
    lon: float,
    lat: float,
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """Map an absolute (lon, lat) onto the chunk-local anchor grid and
    return the two-token form `[<X_ix>, <Y_iy>]`.

    The grid has GRID_SIZE cells per axis. Coordinates outside the
    bounding box are clamped (should not happen on well-formed input,
    but defensive clamping avoids out-of-vocab tokens).
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
    """Turn a list of (lon, lat) vertices into an (anchor_lonlat,
    move_tokens) pair. The anchor is returned as raw lon/lat so the
    caller can convert it to `<X_*> <Y_*>` tokens against the chunk-
    level bounding box.

    Consecutive duplicate vertices are dropped. If after dedup the ring
    has fewer than 2 points, returns the anchor with no moves.
    """
    if not coords:
        return ((0.0, 0.0), [])

    # Dedup consecutive identical vertices (shapely simplify can leave them).
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
            # Below the smallest bucket — skip, but still advance the
            # "previous" anchor so we don't accumulate drift.
            prev_lon, prev_lat = lon, lat
            continue
        bearing = _bearing_bucket(dx, dy)
        for bucket in _split_distance(dist):
            move_tokens.append(f"<MOVE_{bearing}_{bucket}M>")
        prev_lon, prev_lat = lon, lat

    return ((lon0, lat0), move_tokens)


def tokenize_row(
    geom: BaseGeometry,
    kind: str,
    tag_token: str,
    bounds: Tuple[float, float, float, float],
) -> Optional[TokenizedObject]:
    """Produce a TokenizedObject for one geometry, or None if empty.

    `bounds` is the chunk-level (min_lon, min_lat, max_lon, max_lat)
    used to quantise the anchor onto the 256x256 grid.
    """
    rings = list(_iter_rings(geom))
    if not rings:
        return None

    tokens: List[str] = [f"<{kind}_START>", tag_token]
    for i, ring in enumerate(rings):
        (lon0, lat0), moves = tokenize_ring(ring)
        if i > 0:
            # MultiPolygon part boundary — emit a separator so the model
            # can learn "new ring starts here".
            tokens.append("<PART_SEP>")
        tokens.extend(anchor_tokens(lon0, lat0, bounds))
        tokens.extend(moves)
    tokens.append(f"<{kind}_END>")

    centroid = geom.centroid
    return TokenizedObject(
        kind=kind,
        tag_token=tag_token,
        tokens=tokens,
        centroid_lon=float(centroid.x),
        centroid_lat=float(centroid.y),
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class _OSMCollector(osmium.SimpleHandler):
    """pyosmium handler that accumulates highway LineStrings and
    building MultiPolygons with their tag values.

    We keep tag values as plain strings and geometries as shapely
    objects (converted from pyosmium's WKB output).
    """

    def __init__(self) -> None:
        super().__init__()
        self._wkbf = osmium.geom.WKBFactory()
        self.buildings: List[Tuple[BaseGeometry, str]] = []
        self.roads: List[Tuple[BaseGeometry, str]] = []
        self._n_way_fail = 0
        self._n_area_fail = 0

    def way(self, w) -> None:  # pyosmium callback
        hw = w.tags.get("highway")
        if hw is None:
            return
        try:
            wkb_hex = self._wkbf.create_linestring(w)
        except Exception:
            # Incomplete node refs, self-intersecting geometry, etc.
            self._n_way_fail += 1
            return
        geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
        self.roads.append((geom, hw))

    def area(self, a) -> None:  # pyosmium callback
        # The area handler fires for both closed-way and relation-built
        # multipolygons. We filter to building=* here.
        b = a.tags.get("building")
        if b is None:
            return
        try:
            wkb_hex = self._wkbf.create_multipolygon(a)
        except Exception:
            self._n_area_fail += 1
            return
        geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
        self.buildings.append((geom, b))


def load_pbf(pbf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the PBF with pyosmium and return (buildings_df, roads_df).

    Both dataframes are plain pandas with a `geometry` column (shapely)
    plus a string tag column (`building` or `highway`).

    pyosmium's SimpleHandler streams the file once: node locations are
    cached in a memory-mapped index, ways are reconstructed on the fly,
    and multipolygon areas are assembled from ways/relations before the
    `area()` callback fires. This keeps peak memory low even for
    country-scale extracts.
    """
    logger.info("loading PBF via pyosmium: %s", pbf_path)
    handler = _OSMCollector()
    handler.apply_file(pbf_path, locations=True, idx="flex_mem")

    buildings = pd.DataFrame(handler.buildings, columns=["geometry", "building"])
    roads = pd.DataFrame(handler.roads, columns=["geometry", "highway"])

    logger.info(
        "loaded: %d buildings, %d road segments (skipped %d ways, %d areas)",
        len(buildings), len(roads),
        handler._n_way_fail, handler._n_area_fail,
    )
    return buildings, roads


@dataclass
class _RawObject:
    """A simplified geometry plus its kind/tag, pre-tokenization."""
    geom: BaseGeometry
    kind: str        # "BUILDING" or "ROAD"
    tag_token: str   # e.g. "<TAG_RESIDENTIAL>"


def simplify_geometries(
    buildings: pd.DataFrame,
    roads: pd.DataFrame,
) -> List[_RawObject]:
    """Apply Ramer-Douglas-Peucker simplification and tag bucketing.
    Tokenization is deferred until the chunk bounding box is known.
    """
    out: List[_RawObject] = []

    logger.info("simplifying %d buildings", len(buildings))
    for geom, tag in zip(buildings["geometry"], buildings["building"]):
        if geom is None or geom.is_empty:
            continue
        geom = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
        if geom.is_empty:
            continue
        tag_token = f"<TAG_{_bucket_building(tag if isinstance(tag, str) else None)}>"
        out.append(_RawObject(geom=geom, kind="BUILDING", tag_token=tag_token))

    logger.info("simplifying %d road segments", len(roads))
    for geom, tag in zip(roads["geometry"], roads["highway"]):
        if geom is None or geom.is_empty:
            continue
        geom = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
        if geom.is_empty:
            continue
        tag_token = f"<TAG_{_bucket_highway(tag if isinstance(tag, str) else None)}>"
        out.append(_RawObject(geom=geom, kind="ROAD", tag_token=tag_token))

    logger.info("simplified objects: %d", len(out))
    return out


def compute_bounds(
    raws: Sequence[_RawObject],
) -> Tuple[float, float, float, float]:
    """Aggregate each geometry's bounding box into one chunk-level
    (min_lon, min_lat, max_lon, max_lat). Stable whether roads or
    buildings dominate, unlike centroid-only bounds.
    """
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
    """Tokenize simplified geometries against the chunk bounding box."""
    out: List[TokenizedObject] = []
    for r in raws:
        obj = tokenize_row(r.geom, kind=r.kind, tag_token=r.tag_token, bounds=bounds)
        if obj is not None:
            out.append(obj)
    logger.info("total tokenized objects: %d", len(out))
    return out


def sort_objects_zorder(
    objects: List[TokenizedObject],
    bounds: Tuple[float, float, float, float],
) -> List[TokenizedObject]:
    """Sort objects by the Morton (Z-order) key of their centroid,
    reusing the same chunk bounding box we used for anchor quantisation.
    """
    if not objects:
        return objects
    keyed = [(morton_key(o.centroid_lon, o.centroid_lat, bounds), o) for o in objects]
    keyed.sort(key=lambda kv: kv[0])
    return [o for _, o in keyed]


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------


def _object_batches(
    objects: Iterable[TokenizedObject], chunk_size: int
) -> Iterator[pa.RecordBatch]:
    """Yield Arrow RecordBatches of size <= chunk_size."""
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
# Vocab dump (handy for downstream tokenizer construction)
# ---------------------------------------------------------------------------


def _family_of(token: str) -> str:
    """Return the token family name, e.g. `<X_142>` -> `X`."""
    body = token.lstrip("<").rstrip(">")
    return body.split("_", 1)[0] if "_" in body else body


def dump_vocab(objects: Iterable[TokenizedObject], out_path: str) -> None:
    """Write a sorted, deduplicated vocabulary JSON from the corpus and
    log a family-level audit.

    This is NOT a trained tokenizer — it's the raw set of string tokens
    produced by the pipeline. HuggingFace tokenizers can be built from
    this by wrapping it as WordLevel.
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
        logger.info("  %-10s %d", family, count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the .osm.pbf extract (e.g. Stockholm.osm.pbf)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to write tokens + vocab into (created if missing)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Objects per Parquet row-group (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--vocab-name",
        default="stockholm_vocab.json",
        help="Filename for the dumped vocabulary JSON (default: stockholm_vocab.json)",
    )
    parser.add_argument(
        "--output-name",
        default="stockholm_tokens.parquet",
        help="Filename for the Parquet output (default: stockholm_tokens.parquet)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
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

    # Stage 1: parse
    buildings, roads = load_pbf(args.input)

    # Stage 2: simplify geometries (tokenization deferred — we need the
    # chunk bounding box first so the anchor grid is well-defined).
    raws = simplify_geometries(buildings, roads)
    if not raws:
        logger.error("no geometries survived simplification; aborting")
        return 3

    # Stage 3: compute the chunk bounding box used for both the anchor
    # grid and the Z-order sort.
    bounds = compute_bounds(raws)

    # Stage 4: tokenize with local anchor grid + relative moves.
    objects = tokenize_objects(raws, bounds)

    # Stage 5: global Z-order sort using the same bounding box.
    objects = sort_objects_zorder(objects, bounds)

    # Stage 6: persist
    parquet_path = os.path.join(args.output_dir, args.output_name)
    vocab_path = os.path.join(args.output_dir, args.vocab_name)
    write_parquet(objects, parquet_path, chunk_size=args.chunk_size)
    dump_vocab(objects, vocab_path)

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

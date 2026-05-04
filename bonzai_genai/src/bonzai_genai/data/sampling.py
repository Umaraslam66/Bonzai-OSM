"""Tile sampling and OSM PBF feature extraction.

For Phase 0a we use pyosmium (the Python binding for libosmium) to read
PBFs directly. The CLI scans a country PBF once via
``load_pbf_features``, optionally filtered by a country-level bbox, and
stores all relevant features (roads, buildings, land, POIs) keyed by a
coarse lat/lon spatial bucket. Per-tile extraction then queries only
the buckets overlapping the tile bbox -- a few thousand features per
tile instead of millions. For Phase 2 production we'll switch to direct
Overture parquet reads via DuckDB.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import osmium

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    POI,
    Building,
    LandPolygon,
    Road,
    TileGeometry,
)

EARTH_RADIUS_M = 6_378_137.0
_BUCKET_DEG = 0.05  # ~5 km cells: each tile (~2 km) hits 1-4 buckets


def _metres_to_lat(metres: float) -> float:
    return (metres / EARTH_RADIUS_M) * (180.0 / math.pi)


def _metres_to_lon(metres: float, at_lat: float) -> float:
    return (
        (metres / (EARTH_RADIUS_M * math.cos(math.radians(at_lat))))
        * (180.0 / math.pi)
    )


def iter_tile_centres(
    sw_lat: float, sw_lon: float, ne_lat: float, ne_lon: float,
    *,
    shuffle: bool = True,
) -> Iterator[tuple[float, float]]:
    """Yield (lat, lon) for the SW corner of every tile inside the bbox.

    Deterministically shuffled by default so a downstream ``max_tiles``
    cap samples uniformly across the bbox instead of biasing to the SW
    corner -- vital for big bboxes (e.g. Sweden) whose SW corner is
    mostly water.
    """
    centres: list[tuple[float, float]] = []
    lat = sw_lat
    while lat < ne_lat:
        dlat = _metres_to_lat(TILE_SIDE_M)
        lon = sw_lon
        while lon < ne_lon:
            centres.append((lat, lon))
            dlon = _metres_to_lon(TILE_SIDE_M, at_lat=lat)
            lon += dlon
        lat += dlat
    if shuffle:
        random.Random(20260504).shuffle(centres)
    yield from centres


# Mapping from OSM `highway` tag values to our road class names.
ROAD_TAG_MAP: dict[str, str] = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "residential", "service": "service",
    "living_street": "living_street", "pedestrian": "pedestrian",
    "cycleway": "cycleway", "footway": "footway", "path": "path",
    "track": "track", "unclassified": "unclassified",
}

# Known building class names already in our attribute vocab; everything else
# falls back to building_class=UNKNOWN.
KNOWN_BUILDING_CLASSES = {
    "residential", "apartments", "house", "detached", "terrace", "garage",
    "commercial", "retail", "office", "industrial", "warehouse",
    "school", "university", "kindergarten", "hospital", "clinic",
    "church", "mosque", "temple", "synagogue", "chapel", "cathedral",
    "civic", "government", "public", "barn", "farm", "greenhouse", "shed",
    "hotel", "dormitory", "station", "train_station", "parking",
    "fire_station", "police", "museum", "sport", "stadium",
    "hangar", "bunker", "silo", "container", "tower", "chimney",
}

LAND_USE_VALUES = {
    "forest", "meadow", "farmland", "grass", "orchard", "vineyard",
    "residential", "commercial", "industrial", "retail",
}

AMENITY_TO_POI = {
    "cafe": "cafe",
    "restaurant": "restaurant",
    "bar": "bar",
    "pharmacy": "pharmacy",
    "school": "school",
    "hospital": "hospital",
    "bank": "bank",
    "fuel": "gas_station",
    "parking": "parking",
}


# A point or polyline is stored as a list of (lon, lat) tuples in WGS84.
_LonLat = tuple[float, float]
# Feature bbox: (min_lon, min_lat, max_lon, max_lat)
_BBox = tuple[float, float, float, float]
# A way-shaped feature: class name, polyline coords, AABB.
_WayFeature = tuple[str, list[_LonLat], _BBox]


def _bucket_key(lat: float, lon: float) -> tuple[int, int]:
    return (int(math.floor(lat / _BUCKET_DEG)), int(math.floor(lon / _BUCKET_DEG)))


def _bucket_keys_for_bbox(bbox: _BBox) -> list[tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    keys: list[tuple[int, int]] = []
    lat_lo = int(math.floor(min_lat / _BUCKET_DEG))
    lat_hi = int(math.floor(max_lat / _BUCKET_DEG))
    lon_lo = int(math.floor(min_lon / _BUCKET_DEG))
    lon_hi = int(math.floor(max_lon / _BUCKET_DEG))
    for la in range(lat_lo, lat_hi + 1):
        for lo in range(lon_lo, lon_hi + 1):
            keys.append((la, lo))
    return keys


def _bbox_of(coords: list[_LonLat]) -> _BBox:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _bbox_intersects(a: _BBox, b: _BBox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


@dataclass
class _PbfFeatures:
    """Pre-extracted OSM features from a PBF, in WGS84 coordinates.

    Features are stored once in the ``roads``/``buildings``/``land`` lists.
    The ``*_buckets`` dicts map a coarse lat/lon cell key to the indices
    of features whose AABB overlaps that cell; per-tile lookup unions
    those buckets and does a precise bbox-intersect check.
    """
    roads: list[_WayFeature] = field(default_factory=list)
    buildings: list[_WayFeature] = field(default_factory=list)
    land: list[_WayFeature] = field(default_factory=list)
    pois: list[tuple[str, _LonLat]] = field(default_factory=list)
    roads_buckets: dict[tuple[int, int], list[int]] = field(default_factory=lambda: defaultdict(list))  # noqa: E501
    buildings_buckets: dict[tuple[int, int], list[int]] = field(default_factory=lambda: defaultdict(list))  # noqa: E501
    land_buckets: dict[tuple[int, int], list[int]] = field(default_factory=lambda: defaultdict(list))  # noqa: E501


class _FeatureCollector(osmium.SimpleHandler):
    """Walk a PBF once; classify roads / buildings / land / POIs.

    If ``country_bbox`` is given, drop any feature whose AABB doesn't
    intersect it -- vital for "Singapore extracted from a Malaysia +
    Singapore + Brunei bundle" which is otherwise huge.
    """

    def __init__(self, out: _PbfFeatures, country_bbox: _BBox | None):
        super().__init__()
        self._out = out
        self._cb = country_bbox

    def _bucket_way(
        self, kind: str, idx: int, bbox: _BBox,
    ) -> None:
        if kind == "roads":
            buckets = self._out.roads_buckets
        elif kind == "buildings":
            buckets = self._out.buildings_buckets
        else:
            buckets = self._out.land_buckets
        for key in _bucket_keys_for_bbox(bbox):
            buckets[key].append(idx)

    def node(self, n: osmium.osm.Node) -> None:
        amenity = n.tags.get("amenity")
        if amenity is None:
            return
        cls = AMENITY_TO_POI.get(amenity)
        if cls is None:
            return
        try:
            lon, lat = n.location.lon, n.location.lat
        except osmium.InvalidLocationError:
            return
        if self._cb is not None:
            min_lon, min_lat, max_lon, max_lat = self._cb
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                return
        self._out.pois.append((f"poi={cls}", (lon, lat)))

    def way(self, w: osmium.osm.Way) -> None:
        if len(w.nodes) < 2:
            return
        try:
            coords: list[_LonLat] = [(node.lon, node.lat) for node in w.nodes]
        except osmium.InvalidLocationError:
            return
        bbox = _bbox_of(coords)
        if self._cb is not None and not _bbox_intersects(bbox, self._cb):
            return

        tags = w.tags
        if "highway" in tags:
            mapped = ROAD_TAG_MAP.get(tags["highway"])
            if mapped is None:
                return
            idx = len(self._out.roads)
            self._out.roads.append((f"road_class={mapped}", coords, bbox))
            self._bucket_way("roads", idx, bbox)
            return

        is_closed = (
            w.nodes[0].ref == w.nodes[-1].ref and len(w.nodes) >= 4
        )
        if not is_closed:
            return

        if "building" in tags:
            raw = tags["building"]
            cls = raw if raw in KNOWN_BUILDING_CLASSES else "UNKNOWN"
            idx = len(self._out.buildings)
            self._out.buildings.append((f"building_class={cls}", coords, bbox))
            self._bucket_way("buildings", idx, bbox)
            return

        if tags.get("natural") == "water":
            idx = len(self._out.land)
            self._out.land.append(("water_class=lake", coords, bbox))
            self._bucket_way("land", idx, bbox)
            return

        leisure = tags.get("leisure")
        if leisure in ("park", "garden"):
            idx = len(self._out.land)
            self._out.land.append(("land_class=park", coords, bbox))
            self._bucket_way("land", idx, bbox)
            return

        landuse = tags.get("landuse")
        if landuse in LAND_USE_VALUES:
            idx = len(self._out.land)
            self._out.land.append((f"land_class={landuse}", coords, bbox))
            self._bucket_way("land", idx, bbox)


@lru_cache(maxsize=4)
def _load_pbf_features_cached(
    pbf: Path, country_bbox: _BBox | None,
) -> _PbfFeatures:
    out = _PbfFeatures()
    handler = _FeatureCollector(out, country_bbox)
    handler.apply_file(str(pbf), locations=True)
    return out


def load_pbf_features(
    pbf: Path,
    country_bbox: _BBox | None = None,
) -> _PbfFeatures:
    """Scan ``pbf`` once and index its roads/buildings/land/POIs.

    If ``country_bbox`` (min_lon, min_lat, max_lon, max_lat) is given,
    drop features that don't overlap it. Repeated calls with identical
    arguments are served from an in-process cache.
    """
    return _load_pbf_features_cached(Path(pbf).resolve(), country_bbox)


def _to_local(
    lon: float, lat: float, sw_lat: float, sw_lon: float, dlat: float, dlon: float,
) -> tuple[float, float]:
    """Approximate equirectangular projection inside this small tile."""
    x_m = (lon - sw_lon) / dlon * TILE_SIDE_M
    y_m = (lat - sw_lat) / dlat * TILE_SIDE_M
    x_m = max(0.0, min(TILE_SIDE_M - 0.001, x_m))
    y_m = max(0.0, min(TILE_SIDE_M - 0.001, y_m))
    return (x_m, y_m)


def _candidate_indices(
    buckets: dict[tuple[int, int], list[int]], tile_bbox: _BBox,
) -> set[int]:
    out: set[int] = set()
    for key in _bucket_keys_for_bbox(tile_bbox):
        out.update(buckets.get(key, ()))
    return out


def extract_tile_from_features(
    features: _PbfFeatures, sw_lat: float, sw_lon: float,
) -> TileGeometry:
    """Project pre-loaded country features into one tile-local TileGeometry."""
    dlat = _metres_to_lat(TILE_SIDE_M)
    dlon = _metres_to_lon(TILE_SIDE_M, at_lat=sw_lat)
    ne_lat = sw_lat + dlat
    ne_lon = sw_lon + dlon
    tile_bbox: _BBox = (sw_lon, sw_lat, ne_lon, ne_lat)
    geom = TileGeometry()

    for idx in _candidate_indices(features.roads_buckets, tile_bbox):
        cls, polyline, bbox = features.roads[idx]
        if not _bbox_intersects(bbox, tile_bbox):
            continue
        local = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in polyline]
        geom.roads.append(Road(class_name=cls, polyline=local))

    for idx in _candidate_indices(features.buildings_buckets, tile_bbox):
        cls, poly, bbox = features.buildings[idx]
        if not _bbox_intersects(bbox, tile_bbox):
            continue
        local = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in poly]
        geom.buildings.append(Building(
            class_name=cls,
            height_name="height=NA",
            vertices=local,
        ))

    for idx in _candidate_indices(features.land_buckets, tile_bbox):
        cls, poly, bbox = features.land[idx]
        if not _bbox_intersects(bbox, tile_bbox):
            continue
        local = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in poly]
        geom.land.append(LandPolygon(class_name=cls, vertices=local))

    for cls, (lon, lat) in features.pois:
        if not (sw_lon <= lon <= ne_lon and sw_lat <= lat <= ne_lat):
            continue
        xy = _to_local(lon, lat, sw_lat, sw_lon, dlat, dlon)
        geom.pois.append(POI(class_name=cls, point=xy))

    return geom


def extract_tile_geometry_from_osm(
    pbf: Path, sw_lat: float, sw_lon: float,
) -> TileGeometry:
    """Extract a single tile's TileGeometry from an OSM PBF.

    Convenience wrapper around ``load_pbf_features`` +
    ``extract_tile_from_features``: useful for tests and one-off
    extracts. CLI runs over many tiles should call ``load_pbf_features``
    once with a country bbox, then iterate
    ``extract_tile_from_features`` per tile.
    """
    features = load_pbf_features(pbf, country_bbox=None)
    return extract_tile_from_features(features, sw_lat, sw_lon)

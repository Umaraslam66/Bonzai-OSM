"""Parse an OSM PBF into overlapping heterogeneous city-graph chunks."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from pyproj import Transformer
from shapely import wkb as shp_wkb
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

try:
    from .constants import (
        CATEGORICAL_FIELD_NAMES,
        DEFAULT_PREP_CONFIG,
        EDGE_TYPES,
        EDGE_TYPE_TO_ID,
        LANDUSE_KEYS,
        NODE_TYPES,
        NODE_TYPE_TO_ID,
        POI_KEYS,
    )
except ImportError:  # pragma: no cover
    from city_graph_modal.constants import (
        CATEGORICAL_FIELD_NAMES,
        DEFAULT_PREP_CONFIG,
        EDGE_TYPES,
        EDGE_TYPE_TO_ID,
        LANDUSE_KEYS,
        NODE_TYPES,
        NODE_TYPE_TO_ID,
        POI_KEYS,
    )

try:
    import osmium
    import osmium.geom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyosmium is required (pip install 'osmium>=3.7,<4')") from exc


logger = logging.getLogger("prepare_city_graph")

INT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass
class RoadWay:
    tags: Dict[str, str]
    node_ids: List[int]
    coords_lonlat: List[Tuple[float, float]]


@dataclass
class PointFeature:
    key: str
    value: str
    lon: float
    lat: float


@dataclass
class AreaFeature:
    tag_key: str
    tag_value: str
    tags: Dict[str, str]
    geom_lonlat: BaseGeometry


@dataclass
class GraphNode:
    key: str
    node_type: str
    x: float
    y: float
    size_log1p: float = 0.0
    degree_norm: float = 0.0
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    edge_type: str


def _to_lower(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _parse_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    m = INT_RE.search(str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _bucket_length(length_m: float) -> str:
    if length_m < 30:
        return "short"
    if length_m < 100:
        return "medium"
    if length_m < 300:
        return "long"
    return "xlong"


def _bucket_area(area_m2: float) -> str:
    if area_m2 < 100:
        return "tiny"
    if area_m2 < 500:
        return "small"
    if area_m2 < 2_000:
        return "medium"
    return "large"


def _bucket_degree(degree: int) -> str:
    if degree <= 1:
        return "1"
    if degree == 2:
        return "2"
    if degree == 3:
        return "3"
    return "4_plus"


def _bucket_lanes(raw: Optional[str]) -> str:
    value = _parse_number(raw)
    if value is None or value < 1:
        return "unknown"
    lanes = int(round(value))
    if lanes >= 5:
        return "5_plus"
    return str(lanes)


def _bucket_speed(raw: Optional[str]) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).lower()
    value = _parse_number(raw)
    if value is None:
        return "unknown"
    if "mph" in text:
        value *= 1.60934
    if value < 40:
        return "low"
    if value <= 70:
        return "mid"
    return "high"


def _bucket_levels(raw: Optional[str]) -> str:
    value = _parse_number(raw)
    if value is None or value < 1:
        return "unknown"
    if value <= 2:
        return "1_2"
    if value <= 5:
        return "3_5"
    return "6_plus"


def _bucket_height(raw: Optional[str]) -> str:
    value = _parse_number(raw)
    if value is None or value <= 0:
        return "unknown"
    if value < 10:
        return "low"
    if value < 25:
        return "mid"
    return "high"


def _truthy_tag(raw: Optional[str]) -> str:
    value = _to_lower(raw)
    return "yes" if value in {"yes", "true", "1", "viaduct", "aqueduct", "culvert"} else "no"


def _surface_tag(raw: Optional[str]) -> str:
    value = _to_lower(raw)
    if value is None:
        return "unknown"
    known = {
        "asphalt",
        "concrete",
        "paved",
        "paving_stones",
        "cobblestone",
        "gravel",
        "unpaved",
        "dirt",
        "ground",
        "sand",
        "grass",
    }
    return value if value in known else "other"


def _building_shape_bucket(geom: BaseGeometry) -> str:
    env = geom.envelope
    env_area = env.area if env is not None else 0.0
    if env_area <= 0:
        return "irregular"
    fill = float(geom.area / env_area)
    if fill > 0.85:
        return "rectangular"
    if fill > 0.55:
        return "compact"
    return "irregular"


def _choose_poi_tag(tags) -> Optional[Tuple[str, str]]:
    for key in POI_KEYS:
        value = tags.get(key)
        if value not in (None, ""):
            return key, value
    return None


def _choose_landuse_tag(tags) -> Optional[Tuple[str, str]]:
    for key in LANDUSE_KEYS:
        value = tags.get(key)
        if value not in (None, ""):
            return key, value
    return None


def _resolve_tree_indices(hits, id_to_index: Dict[int, int]) -> Iterator[int]:
    raw = hits.tolist() if hasattr(hits, "tolist") else hits
    for item in raw:
        if isinstance(item, (int, np.integer)):
            yield int(item)
        else:
            idx = id_to_index.get(id(item))
            if idx is not None:
                yield idx


class OSMCollector(osmium.SimpleHandler):
    """Collect roads, buildings, landuse, and POIs from a PBF."""

    def __init__(self) -> None:
        super().__init__()
        self._wkbf = osmium.geom.WKBFactory()
        self.roads: List[RoadWay] = []
        self.buildings: List[AreaFeature] = []
        self.landuses: List[AreaFeature] = []
        self.pois: List[PointFeature] = []
        self.skipped_ways = 0
        self.skipped_areas = 0
        self.skipped_nodes = 0

    def node(self, n) -> None:
        found = _choose_poi_tag(n.tags)
        if found is None:
            return
        try:
            lon = float(n.location.lon)
            lat = float(n.location.lat)
        except Exception:
            self.skipped_nodes += 1
            return
        key, value = found
        self.pois.append(PointFeature(key=key, value=value, lon=lon, lat=lat))

    def way(self, w) -> None:
        highway = w.tags.get("highway")
        if highway in (None, ""):
            return
        coords: List[Tuple[float, float]] = []
        node_ids: List[int] = []
        try:
            for node in w.nodes:
                if not node.location.valid():
                    continue
                node_ids.append(int(node.ref))
                coords.append((float(node.location.lon), float(node.location.lat)))
        except Exception:
            self.skipped_ways += 1
            return

        if len(coords) < 2:
            self.skipped_ways += 1
            return

        tags = {}
        for key in (
            "highway",
            "name",
            "oneway",
            "lanes",
            "surface",
            "maxspeed",
            "bridge",
            "tunnel",
        ):
            value = w.tags.get(key)
            if value not in (None, ""):
                tags[key] = str(value)
        self.roads.append(RoadWay(tags=tags, node_ids=node_ids, coords_lonlat=coords))

    def area(self, a) -> None:
        building = a.tags.get("building")
        landuse = _choose_landuse_tag(a.tags)
        if building in (None, "") and landuse is None:
            return
        try:
            wkb_hex = self._wkbf.create_multipolygon(a)
            geom = shp_wkb.loads(bytes.fromhex(wkb_hex))
        except Exception:
            self.skipped_areas += 1
            return

        if geom.is_empty:
            self.skipped_areas += 1
            return

        tags = {}
        for key in (
            "building",
            "building:levels",
            "height",
            "landuse",
            "natural",
            "leisure",
            "name",
        ):
            value = a.tags.get(key)
            if value not in (None, ""):
                tags[key] = str(value)

        if building not in (None, ""):
            self.buildings.append(
                AreaFeature(
                    tag_key="building",
                    tag_value=str(building),
                    tags=tags,
                    geom_lonlat=geom,
                )
            )
            return

        if landuse is not None:
            key, value = landuse
            self.landuses.append(
                AreaFeature(
                    tag_key=key,
                    tag_value=str(value),
                    tags=tags,
                    geom_lonlat=geom,
                )
            )


def _ensure_polygon(geom: BaseGeometry) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = getattr(geom, "geoms", None)
    if parts is None:
        return None
    polys = [g for g in parts if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon([g for poly in polys for g in getattr(poly, "geoms", [poly])])


def load_osm(pbf_path: str) -> OSMCollector:
    collector = OSMCollector()
    collector.apply_file(pbf_path, locations=True, idx="flex_mem")
    logger.info(
        "loaded roads=%d buildings=%d landuse=%d pois=%d skipped(node/way/area)=%d/%d/%d",
        len(collector.roads),
        len(collector.buildings),
        len(collector.landuses),
        len(collector.pois),
        collector.skipped_nodes,
        collector.skipped_ways,
        collector.skipped_areas,
    )
    return collector


def build_global_graph(collector: OSMCollector) -> Tuple[List[GraphNode], List[GraphEdge], Dict[str, List[str]]]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    project_geom = lambda geom: shp_transform(transformer.transform, geom)

    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    aux_stats: Dict[str, List[str]] = defaultdict(list)

    node_use_counts: Counter[int] = Counter()
    endpoint_ids: set[int] = set()
    for way in collector.roads:
        node_use_counts.update(way.node_ids)
        endpoint_ids.add(way.node_ids[0])
        endpoint_ids.add(way.node_ids[-1])

    junction_ids = {node_id for node_id, count in node_use_counts.items() if count > 1}
    junction_ids.update(endpoint_ids)

    junction_degree: Counter[int] = Counter()

    segment_geoms: List[LineString] = []
    segment_keys: List[str] = []

    for way_idx, way in enumerate(collector.roads):
        split_points = [0]
        for idx in range(1, len(way.node_ids) - 1):
            if way.node_ids[idx] in junction_ids:
                split_points.append(idx)
        split_points.append(len(way.node_ids) - 1)
        split_points = sorted(set(split_points))

        for seg_idx, (start_idx, end_idx) in enumerate(zip(split_points[:-1], split_points[1:]), start=0):
            coords_lonlat = way.coords_lonlat[start_idx : end_idx + 1]
            ids = way.node_ids[start_idx : end_idx + 1]
            if len(coords_lonlat) < 2 or len(set(ids)) < 2:
                continue
            px, py = transformer.transform(
                [lon for lon, _ in coords_lonlat],
                [lat for _, lat in coords_lonlat],
            )
            proj_coords = list(zip(px, py))
            line = LineString(proj_coords)
            if line.is_empty or line.length <= 0:
                continue

            start_j = ids[0]
            end_j = ids[-1]
            junction_degree[start_j] += 1
            junction_degree[end_j] += 1

            for node_id, coord in ((start_j, proj_coords[0]), (end_j, proj_coords[-1])):
                key = f"junction:{node_id}"
                if key not in nodes:
                    nodes[key] = GraphNode(
                        key=key,
                        node_type="ROAD_JUNCTION",
                        x=float(coord[0]),
                        y=float(coord[1]),
                    )

            seg_key = f"segment:{way_idx}:{seg_idx}"
            segment_keys.append(seg_key)
            segment_geoms.append(line)
            centroid = line.centroid
            nodes[seg_key] = GraphNode(
                key=seg_key,
                node_type="ROAD_SEGMENT",
                x=float(centroid.x),
                y=float(centroid.y),
                size_log1p=float(math.log1p(line.length)),
                attrs={
                    "road_class": _to_lower(way.tags.get("highway")) or "other",
                    "oneway": _truthy_tag(way.tags.get("oneway")),
                    "lanes_bucket": _bucket_lanes(way.tags.get("lanes")),
                    "surface": _surface_tag(way.tags.get("surface")),
                    "speed_bucket": _bucket_speed(way.tags.get("maxspeed")),
                    "bridge": _truthy_tag(way.tags.get("bridge")),
                    "tunnel": _truthy_tag(way.tags.get("tunnel")),
                    "length_bucket": _bucket_length(line.length),
                },
            )

            start_key = f"junction:{start_j}"
            end_key = f"junction:{end_j}"
            edges.extend(
                [
                    GraphEdge(src=seg_key, dst=start_key, edge_type="SEGMENT_CONNECTS_JUNCTION"),
                    GraphEdge(src=start_key, dst=seg_key, edge_type="SEGMENT_CONNECTS_JUNCTION"),
                    GraphEdge(src=seg_key, dst=end_key, edge_type="SEGMENT_CONNECTS_JUNCTION"),
                    GraphEdge(src=end_key, dst=seg_key, edge_type="SEGMENT_CONNECTS_JUNCTION"),
                    GraphEdge(src=start_key, dst=end_key, edge_type="JUNCTION_ADJACENT_JUNCTION"),
                    GraphEdge(src=end_key, dst=start_key, edge_type="JUNCTION_ADJACENT_JUNCTION"),
                ]
            )

    for node_id, degree in junction_degree.items():
        key = f"junction:{node_id}"
        if key not in nodes:
            continue
        nodes[key].degree_norm = min(degree / 8.0, 1.0)
        nodes[key].attrs["degree_bucket"] = _bucket_degree(degree)

    segment_tree = STRtree(segment_geoms) if segment_geoms else None
    segment_geom_id_to_idx = {id(geom): idx for idx, geom in enumerate(segment_geoms)}

    landuse_geoms: List[BaseGeometry] = []
    landuse_keys: List[str] = []
    for idx, feature in enumerate(collector.landuses):
        geom = _ensure_polygon(project_geom(feature.geom_lonlat))
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        key = f"landuse:{idx}"
        nodes[key] = GraphNode(
            key=key,
            node_type="LANDUSE",
            x=float(centroid.x),
            y=float(centroid.y),
            size_log1p=float(math.log1p(geom.area)),
            attrs={
                "primary_tag_key": feature.tag_key,
                "primary_tag_value": _to_lower(feature.tag_value) or "other",
                "landuse_class": _to_lower(feature.tag_value) or "other",
                "area_bucket": _bucket_area(geom.area),
            },
        )
        landuse_geoms.append(geom)
        landuse_keys.append(key)

    landuse_tree = STRtree(landuse_geoms) if landuse_geoms else None
    landuse_geom_id_to_idx = {id(geom): idx for idx, geom in enumerate(landuse_geoms)}

    for idx, feature in enumerate(collector.buildings):
        geom = _ensure_polygon(project_geom(feature.geom_lonlat))
        if geom is None or geom.is_empty or geom.area <= 0:
            continue
        centroid = geom.centroid
        key = f"building:{idx}"
        nodes[key] = GraphNode(
            key=key,
            node_type="BUILDING",
            x=float(centroid.x),
            y=float(centroid.y),
            size_log1p=float(math.log1p(geom.area)),
            attrs={
                "building_class": _to_lower(feature.tag_value) or "yes",
                "area_bucket": _bucket_area(geom.area),
                "levels_bucket": _bucket_levels(feature.tags.get("building:levels")),
                "height_bucket": _bucket_height(feature.tags.get("height")),
                "shape_bucket": _building_shape_bucket(geom),
            },
        )

        if segment_tree is not None:
            hits = segment_tree.query(centroid.buffer(DEFAULT_PREP_CONFIG["nearest_segment_threshold_m"]))
            nearest_key = None
            nearest_dist = None
            for seg_idx in _resolve_tree_indices(hits, segment_geom_id_to_idx):
                dist = float(segment_geoms[seg_idx].distance(centroid))
                if dist <= DEFAULT_PREP_CONFIG["nearest_segment_threshold_m"] and (
                    nearest_dist is None or dist < nearest_dist
                ):
                    nearest_dist = dist
                    nearest_key = segment_keys[seg_idx]
            if nearest_key is not None:
                edges.extend(
                    [
                        GraphEdge(src=key, dst=nearest_key, edge_type="BUILDING_NEAR_SEGMENT"),
                        GraphEdge(src=nearest_key, dst=key, edge_type="BUILDING_NEAR_SEGMENT"),
                    ]
                )

        if landuse_tree is not None:
            hits = landuse_tree.query(centroid)
            for land_idx in _resolve_tree_indices(hits, landuse_geom_id_to_idx):
                if landuse_geoms[land_idx].contains(centroid):
                    land_key = landuse_keys[land_idx]
                    edges.extend(
                        [
                            GraphEdge(src=key, dst=land_key, edge_type="BUILDING_INSIDE_LANDUSE"),
                            GraphEdge(src=land_key, dst=key, edge_type="BUILDING_INSIDE_LANDUSE"),
                        ]
                    )
                    break

    for idx, poi in enumerate(collector.pois):
        x, y = transformer.transform(poi.lon, poi.lat)
        key = f"poi:{idx}"
        point = Point(x, y)
        nodes[key] = GraphNode(
            key=key,
            node_type="POI",
            x=float(x),
            y=float(y),
            attrs={
                "primary_tag_key": poi.key,
                "primary_tag_value": _to_lower(poi.value) or "other",
            },
        )

        if segment_tree is not None:
            hits = segment_tree.query(point.buffer(DEFAULT_PREP_CONFIG["nearest_segment_threshold_m"]))
            nearest_key = None
            nearest_dist = None
            for seg_idx in _resolve_tree_indices(hits, segment_geom_id_to_idx):
                dist = float(segment_geoms[seg_idx].distance(point))
                if dist <= DEFAULT_PREP_CONFIG["nearest_segment_threshold_m"] and (
                    nearest_dist is None or dist < nearest_dist
                ):
                    nearest_dist = dist
                    nearest_key = segment_keys[seg_idx]
            if nearest_key is not None:
                edges.extend(
                    [
                        GraphEdge(src=key, dst=nearest_key, edge_type="POI_NEAR_SEGMENT"),
                        GraphEdge(src=nearest_key, dst=key, edge_type="POI_NEAR_SEGMENT"),
                    ]
                )

        if landuse_tree is not None:
            hits = landuse_tree.query(point)
            for land_idx in _resolve_tree_indices(hits, landuse_geom_id_to_idx):
                if landuse_geoms[land_idx].contains(point):
                    land_key = landuse_keys[land_idx]
                    edges.extend(
                        [
                            GraphEdge(src=key, dst=land_key, edge_type="POI_INSIDE_LANDUSE"),
                            GraphEdge(src=land_key, dst=key, edge_type="POI_INSIDE_LANDUSE"),
                        ]
                    )
                    break

    aux_stats["node_types"] = [node.node_type for node in nodes.values()]
    aux_stats["edge_types"] = [edge.edge_type for edge in edges]
    return list(nodes.values()), edges, aux_stats


def build_vocabs(nodes: Sequence[GraphNode]) -> Dict[str, List[str]]:
    vocab_values: Dict[str, set[str]] = {field: set() for field in CATEGORICAL_FIELD_NAMES}
    for node in nodes:
        for field, value in node.attrs.items():
            if field in vocab_values and value not in (None, ""):
                vocab_values[field].add(str(value))
    return {field: sorted(values) for field, values in vocab_values.items()}


def encode_nodes(nodes: Sequence[GraphNode], vocabs: Dict[str, List[str]]) -> Tuple[List[GraphNode], Dict[str, Dict[str, int]]]:
    index_maps = {field: {value: idx for idx, value in enumerate(values)} for field, values in vocabs.items()}
    for node in nodes:
        for field in CATEGORICAL_FIELD_NAMES:
            if field not in node.attrs:
                continue
            node.attrs[field] = str(index_maps[field].get(node.attrs[field], -1))
    return list(nodes), index_maps


def assign_split(ix: int, iy: int, modulus: int) -> str:
    bucket = (ix * 92821 + iy * 68917) % modulus
    if bucket < modulus - 2:
        return "train"
    if bucket == modulus - 2:
        return "val"
    return "test"


def write_chunk(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def chunk_graph(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    vocabs: Dict[str, List[str]],
    output_dir: Path,
    tile_size_m: float,
    tile_overlap_m: float,
    min_nodes_per_chunk: int,
    min_edges_per_chunk: int,
    spatial_split_modulus: int,
) -> dict:
    key_to_index = {node.key: idx for idx, node in enumerate(nodes)}
    edge_triplets = [
        (key_to_index[e.src], key_to_index[e.dst], EDGE_TYPE_TO_ID[e.edge_type])
        for e in edges
        if e.src in key_to_index and e.dst in key_to_index
    ]

    xs = np.array([node.x for node in nodes], dtype=np.float64)
    ys = np.array([node.y for node in nodes], dtype=np.float64)
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    max_ix = int(math.floor((max_x - min_x) / tile_size_m))
    max_iy = int(math.floor((max_y - min_y) / tile_size_m))

    stats = {"chunks": {"train": 0, "val": 0, "test": 0}, "nodes": len(nodes), "edges": len(edge_triplets)}

    manifests: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for ix in range(max_ix + 1):
        for iy in range(max_iy + 1):
            inner_min_x = min_x + ix * tile_size_m
            inner_max_x = inner_min_x + tile_size_m
            inner_min_y = min_y + iy * tile_size_m
            inner_max_y = inner_min_y + tile_size_m
            outer_min_x = inner_min_x - tile_overlap_m
            outer_max_x = inner_max_x + tile_overlap_m
            outer_min_y = inner_min_y - tile_overlap_m
            outer_max_y = inner_max_y + tile_overlap_m

            selected = np.where(
                (xs >= outer_min_x) & (xs <= outer_max_x) & (ys >= outer_min_y) & (ys <= outer_max_y)
            )[0]
            if selected.size < min_nodes_per_chunk:
                continue
            selected_set = set(int(idx) for idx in selected.tolist())

            interior_mask = [
                1 if (inner_min_x <= nodes[idx].x <= inner_max_x and inner_min_y <= nodes[idx].y <= inner_max_y) else 0
                for idx in selected
            ]
            if sum(interior_mask) < max(8, min_nodes_per_chunk // 2):
                continue

            local_index = {global_idx: local_idx for local_idx, global_idx in enumerate(selected.tolist())}
            local_edges_src: List[int] = []
            local_edges_dst: List[int] = []
            local_edge_types: List[int] = []
            for src, dst, edge_type in edge_triplets:
                if src in selected_set and dst in selected_set:
                    local_edges_src.append(local_index[src])
                    local_edges_dst.append(local_index[dst])
                    local_edge_types.append(edge_type)
            if len(local_edge_types) < min_edges_per_chunk:
                continue

            split = assign_split(ix, iy, spatial_split_modulus)
            width = max(outer_max_x - outer_min_x, 1.0)
            height = max(outer_max_y - outer_min_y, 1.0)

            continuous: List[List[float]] = []
            node_types: List[int] = []
            categorical: Dict[str, List[int]] = {field: [] for field in CATEGORICAL_FIELD_NAMES}
            for global_idx in selected.tolist():
                node = nodes[global_idx]
                node_types.append(NODE_TYPE_TO_ID[node.node_type])
                continuous.append(
                    [
                        float((node.x - outer_min_x) / width),
                        float((node.y - outer_min_y) / height),
                        float(node.size_log1p),
                        float(node.degree_norm),
                    ]
                )
                for field in CATEGORICAL_FIELD_NAMES:
                    categorical[field].append(int(node.attrs.get(field, "-1")))

            chunk_id = f"tile_{ix}_{iy}"
            rel_path = Path("chunks") / split / f"{chunk_id}.json.gz"
            payload = {
                "chunk_id": chunk_id,
                "split": split,
                "tile": {"ix": ix, "iy": iy},
                "node_types": node_types,
                "continuous": continuous,
                "categorical": categorical,
                "interior_mask": interior_mask,
                "edge_index": [local_edges_src, local_edges_dst],
                "edge_types": local_edge_types,
                "bounds_m": {
                    "inner_min_x": inner_min_x,
                    "inner_max_x": inner_max_x,
                    "inner_min_y": inner_min_y,
                    "inner_max_y": inner_max_y,
                    "outer_min_x": outer_min_x,
                    "outer_max_x": outer_max_x,
                    "outer_min_y": outer_min_y,
                    "outer_max_y": outer_max_y,
                },
            }
            write_chunk(output_dir / rel_path, payload)
            manifests[split].append(str(rel_path).replace("\\", "/"))
            stats["chunks"][split] += 1

    for split, rel_paths in manifests.items():
        manifest_path = output_dir / f"{split}_manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            for rel_path in rel_paths:
                fh.write(json.dumps({"path": rel_path}) + "\n")

    metadata = {
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "categorical_fields": list(CATEGORICAL_FIELD_NAMES),
        "categorical_vocabs": vocabs,
        "continuous_fields": ["x_norm", "y_norm", "size_log1p", "degree_norm"],
        "stats": stats,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def prepare_dataset(
    input_pbf: str,
    output_dir: str,
    tile_size_m: float = DEFAULT_PREP_CONFIG["tile_size_m"],
    tile_overlap_m: float = DEFAULT_PREP_CONFIG["tile_overlap_m"],
    min_nodes_per_chunk: int = DEFAULT_PREP_CONFIG["min_nodes_per_chunk"],
    min_edges_per_chunk: int = DEFAULT_PREP_CONFIG["min_edges_per_chunk"],
    spatial_split_modulus: int = DEFAULT_PREP_CONFIG["spatial_split_modulus"],
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    collector = load_osm(input_pbf)
    nodes, edges, aux = build_global_graph(collector)
    logger.info(
        "global graph nodes=%d edges=%d node_types=%s edge_types=%s",
        len(nodes),
        len(edges),
        dict(Counter(aux["node_types"]).most_common()),
        dict(Counter(aux["edge_types"]).most_common()),
    )
    vocabs = build_vocabs(nodes)
    nodes, _ = encode_nodes(nodes, vocabs)
    metadata = chunk_graph(
        nodes=nodes,
        edges=edges,
        vocabs=vocabs,
        output_dir=output_root,
        tile_size_m=tile_size_m,
        tile_overlap_m=tile_overlap_m,
        min_nodes_per_chunk=min_nodes_per_chunk,
        min_edges_per_chunk=min_edges_per_chunk,
        spatial_split_modulus=spatial_split_modulus,
    )

    summary = {
        "input_pbf": input_pbf,
        "output_dir": output_dir,
        "roads": len(collector.roads),
        "buildings": len(collector.buildings),
        "landuses": len(collector.landuses),
        "pois": len(collector.pois),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "chunks": metadata["stats"]["chunks"],
    }
    with (output_root / "prepare_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .osm.pbf file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size-m", type=float, default=DEFAULT_PREP_CONFIG["tile_size_m"])
    parser.add_argument("--tile-overlap-m", type=float, default=DEFAULT_PREP_CONFIG["tile_overlap_m"])
    parser.add_argument("--min-nodes-per-chunk", type=int, default=DEFAULT_PREP_CONFIG["min_nodes_per_chunk"])
    parser.add_argument("--min-edges-per-chunk", type=int, default=DEFAULT_PREP_CONFIG["min_edges_per_chunk"])
    parser.add_argument("--spatial-split-modulus", type=int, default=DEFAULT_PREP_CONFIG["spatial_split_modulus"])
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not os.path.isfile(args.input):
        logger.error("input PBF not found: %s", args.input)
        return 2
    summary = prepare_dataset(
        input_pbf=args.input,
        output_dir=args.output_dir,
        tile_size_m=args.tile_size_m,
        tile_overlap_m=args.tile_overlap_m,
        min_nodes_per_chunk=args.min_nodes_per_chunk,
        min_edges_per_chunk=args.min_edges_per_chunk,
        spatial_split_modulus=args.spatial_split_modulus,
    )
    logger.info("prepare summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared constants for the Luxembourg city-graph smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


NODE_TYPES: Tuple[str, ...] = (
    "ROAD_JUNCTION",
    "ROAD_SEGMENT",
    "BUILDING",
    "POI",
    "LANDUSE",
)


EDGE_TYPES: Tuple[str, ...] = (
    "SEGMENT_CONNECTS_JUNCTION",
    "JUNCTION_ADJACENT_JUNCTION",
    "BUILDING_NEAR_SEGMENT",
    "POI_NEAR_SEGMENT",
    "BUILDING_INSIDE_LANDUSE",
    "POI_INSIDE_LANDUSE",
)


CONTINUOUS_FIELDS: Tuple[str, ...] = (
    "x_norm",
    "y_norm",
    "size_log1p",
    "degree_norm",
)


@dataclass(frozen=True)
class CategoricalField:
    name: str
    applies_to: Tuple[str, ...]


CATEGORICAL_FIELDS: Tuple[CategoricalField, ...] = (
    CategoricalField("primary_tag_key", ("POI", "LANDUSE")),
    CategoricalField("primary_tag_value", ("POI", "LANDUSE")),
    CategoricalField("road_class", ("ROAD_SEGMENT",)),
    CategoricalField("building_class", ("BUILDING",)),
    CategoricalField("landuse_class", ("LANDUSE",)),
    CategoricalField("oneway", ("ROAD_SEGMENT",)),
    CategoricalField("lanes_bucket", ("ROAD_SEGMENT",)),
    CategoricalField("surface", ("ROAD_SEGMENT",)),
    CategoricalField("speed_bucket", ("ROAD_SEGMENT",)),
    CategoricalField("bridge", ("ROAD_SEGMENT",)),
    CategoricalField("tunnel", ("ROAD_SEGMENT",)),
    CategoricalField("degree_bucket", ("ROAD_JUNCTION",)),
    CategoricalField("length_bucket", ("ROAD_SEGMENT",)),
    CategoricalField("area_bucket", ("BUILDING", "LANDUSE")),
    CategoricalField("levels_bucket", ("BUILDING",)),
    CategoricalField("height_bucket", ("BUILDING",)),
    CategoricalField("shape_bucket", ("BUILDING",)),
)


CATEGORICAL_FIELD_NAMES: Tuple[str, ...] = tuple(f.name for f in CATEGORICAL_FIELDS)


FIELD_TO_TYPES: Dict[str, Tuple[str, ...]] = {f.name: f.applies_to for f in CATEGORICAL_FIELDS}


POI_KEYS: Tuple[str, ...] = (
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "historic",
    "office",
    "craft",
    "healthcare",
)


LANDUSE_KEYS: Tuple[str, ...] = ("landuse", "natural", "leisure")


ROAD_PRIORITY_KEYS: Tuple[str, ...] = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
    "track",
    "path",
    "cycleway",
    "footway",
)


DEFAULT_PREP_CONFIG = {
    "tile_size_m": 1000.0,
    "tile_overlap_m": 192.0,
    "nearest_segment_threshold_m": 120.0,
    "min_nodes_per_chunk": 24,
    "min_edges_per_chunk": 16,
    "spatial_split_modulus": 10,
}


NODE_TYPE_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(EDGE_TYPES)}


def field_names() -> List[str]:
    return list(CATEGORICAL_FIELD_NAMES)


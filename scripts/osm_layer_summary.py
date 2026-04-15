#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from osgeo import gdal, ogr


POI_KEYS = {
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "sport",
    "office",
    "craft",
    "historic",
    "healthcare",
    "place",
}

TAG_PAIR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def parse_other_tags(raw_value: str) -> dict[str, str]:
    if not raw_value:
        return {}

    tags = {}
    for key, value in TAG_PAIR_RE.findall(raw_value):
        clean_key = key.replace('\\"', '"').replace("\\\\", "\\")
        clean_value = value.replace('\\"', '"').replace("\\\\", "\\")
        tags[clean_key] = clean_value
    return tags


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def field_definition(layer_defn):
    fields = []
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        fields.append(
            {
                "name": field_defn.GetName(),
                "type": field_defn.GetFieldTypeName(field_defn.GetType()),
                "width": field_defn.GetWidth(),
                "precision": field_defn.GetPrecision(),
            }
        )
    return fields


def sample_geometry(feature):
    geometry = feature.GetGeometryRef()
    if geometry is None:
        return None

    envelope = geometry.GetEnvelope()
    summary = {
        "geometry_type": geometry.GetGeometryName(),
        "bbox": {
            "min_x": envelope[0],
            "max_x": envelope[1],
            "min_y": envelope[2],
            "max_y": envelope[3],
        },
    }

    if geometry.GetGeometryType() == ogr.wkbPoint:
        summary["point"] = {"x": geometry.GetX(), "y": geometry.GetY()}

    return summary


def looks_like_poi(properties: dict[str, object], other_tags: dict[str, str]) -> bool:
    if any(properties.get(key) not in (None, "") for key in POI_KEYS):
        return True
    return any(key in other_tags for key in POI_KEYS)


def collect_layer_summary(layer, sample_limit: int):
    layer_name = layer.GetName()
    layer_defn = layer.GetLayerDefn()
    field_defs = field_definition(layer_defn)
    field_names = [field["name"] for field in field_defs]
    extent = layer.GetExtent(1)

    non_null_counts = Counter()
    other_tag_key_counts = Counter()
    samples = []

    layer_count = 0
    road_count = 0
    building_count = 0
    poi_like_count = 0

    layer.ResetReading()

    for feature in layer:
        layer_count += 1

        properties = {}
        for field_name in field_names:
            value = normalize_value(feature.GetField(field_name))
            if value not in (None, ""):
                non_null_counts[field_name] += 1
            properties[field_name] = value

        parsed_other_tags = parse_other_tags(properties.get("other_tags") or "")
        if parsed_other_tags:
            for key in parsed_other_tags:
                other_tag_key_counts[key] += 1

        if layer_name == "lines" and (
            properties.get("highway") not in (None, "") or "highway" in parsed_other_tags
        ):
            road_count += 1

        if layer_name == "multipolygons" and (
            properties.get("building") not in (None, "") or "building" in parsed_other_tags
        ):
            building_count += 1

        if looks_like_poi(properties, parsed_other_tags):
            poi_like_count += 1

        if len(samples) < sample_limit:
            sample_properties = {}
            for key, value in properties.items():
                if value not in (None, ""):
                    sample_properties[key] = value

            samples.append(
                {
                    "fid": feature.GetFID(),
                    "properties": sample_properties,
                    "parsed_other_tags": parsed_other_tags,
                    "geometry_summary": sample_geometry(feature),
                }
            )

    extent_summary = None
    if extent:
        extent_summary = {
            "min_x": extent[0],
            "max_x": extent[1],
            "min_y": extent[2],
            "max_y": extent[3],
        }

    return {
        "name": layer_name,
        "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
        "feature_count": layer_count,
        "extent": extent_summary,
        "fields": field_defs,
        "field_names": field_names,
        "non_null_field_counts": dict(sorted(non_null_counts.items())),
        "other_tags_key_counts": dict(sorted(other_tag_key_counts.items())),
        "top_other_tag_keys": [
            {"key": key, "count": count}
            for key, count in other_tag_key_counts.most_common(20)
        ],
        "samples": samples,
        "derived_counts": {
            "road_like_features": road_count,
            "building_like_features": building_count,
            "poi_like_features": poi_like_count,
        },
    }


def build_summary(input_path: Path, sample_limit: int) -> dict:
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")
    ogr.DontUseExceptions()
    dataset = ogr.Open(str(input_path))
    if dataset is None:
        raise RuntimeError(f"Unable to open input dataset: {input_path}")

    layer_summaries = []
    feature_class_counts = {
        "point_features": 0,
        "line_features": 0,
        "multipolygon_features": 0,
    }
    derived_theme_counts = Counter()

    for layer_index in range(dataset.GetLayerCount()):
        layer = dataset.GetLayerByIndex(layer_index)
        summary = collect_layer_summary(layer, sample_limit)
        layer_summaries.append(summary)

        if summary["name"] == "points":
            feature_class_counts["point_features"] = summary["feature_count"]
        elif summary["name"] == "lines":
            feature_class_counts["line_features"] = summary["feature_count"]
        elif summary["name"] == "multipolygons":
            feature_class_counts["multipolygon_features"] = summary["feature_count"]

        for key, value in summary["derived_counts"].items():
            derived_theme_counts[key] += value

    total_features = sum(layer["feature_count"] for layer in layer_summaries)
    feature_class_counts["node_like_features"] = feature_class_counts["point_features"]
    feature_class_counts["edge_like_features"] = feature_class_counts["line_features"]
    feature_class_counts["area_like_features"] = feature_class_counts["multipolygon_features"]

    known_primary_layers = {"points", "lines", "multipolygons"}
    layer_names = [layer["name"] for layer in layer_summaries]
    extra_layers = [
        {
            "name": layer["name"],
            "geometry_type": layer["geometry_type"],
            "feature_count": layer["feature_count"],
        }
        for layer in layer_summaries
        if layer["name"] not in known_primary_layers
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "driver": dataset.GetDriver().GetName(),
        "layer_count": dataset.GetLayerCount(),
        "layer_names": layer_names,
        "total_features": total_features,
        "feature_class_counts": feature_class_counts,
        "derived_theme_counts": dict(sorted(derived_theme_counts.items())),
        "extra_layers": extra_layers,
        "notes": [
            "node_like_features and edge_like_features are layer-level proxies from GDAL OSM output, not raw OSM primitive counts.",
            "poi_like_features are heuristic counts based on common POI tags in explicit fields and parsed other_tags.",
            "other_tags_key_counts capture keys that were not promoted to explicit schema fields by the OSM driver.",
        ],
        "layers": layer_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a JSON summary of an OSM dataset exposed through GDAL/OGR."
    )
    parser.add_argument("--input", required=True, help="Path to input .osm.pbf or other OGR dataset")
    parser.add_argument("--output", required=True, help="Path to output JSON summary")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of sample features to keep per layer",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(input_path, args.sample_limit)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()

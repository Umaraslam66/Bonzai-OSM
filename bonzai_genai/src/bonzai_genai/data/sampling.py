"""Tile sampling and OSM PBF feature extraction.

For Phase 0a we use ``osmium`` (system tool) to extract a bounding-box
subset to GeoJSON, then load it with the standard library. Fine for
small-country-scale prototyping (a few seconds per tile). For Phase 2
production we'll switch to direct Overture parquet reads in DuckDB.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
)

EARTH_RADIUS_M = 6_378_137.0


def _metres_to_lat(metres: float) -> float:
    return (metres / EARTH_RADIUS_M) * (180.0 / math.pi)


def _metres_to_lon(metres: float, at_lat: float) -> float:
    return (
        (metres / (EARTH_RADIUS_M * math.cos(math.radians(at_lat))))
        * (180.0 / math.pi)
    )


def iter_tile_centres(
    sw_lat: float, sw_lon: float, ne_lat: float, ne_lon: float,
) -> Iterator[tuple[float, float]]:
    """Yield (lat, lon) for the SW corner of every tile inside the bbox."""
    lat = sw_lat
    while lat < ne_lat:
        dlat = _metres_to_lat(TILE_SIDE_M)
        lon = sw_lon
        while lon < ne_lon:
            yield (lat, lon)
            dlon = _metres_to_lon(TILE_SIDE_M, at_lat=lat)
            lon += dlon
        lat += dlat


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


def _extract_bbox_geojson(
    pbf: Path, west: float, south: float, east: float, north: float,
) -> dict:
    """Run osmium to extract everything inside (W,S,E,N) and return as GeoJSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        subset_pbf = tmp / "subset.osm.pbf"
        out_path = tmp / "subset.geojson"

        subprocess.run(
            [
                "osmium", "extract",
                "--bbox", f"{west},{south},{east},{north}",
                "--strategy=smart",
                "--overwrite",
                "-o", str(subset_pbf),
                str(pbf),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "osmium", "export",
                "--overwrite",
                "-f", "geojson",
                "-o", str(out_path),
                str(subset_pbf),
            ],
            check=True,
            capture_output=True,
        )
        return json.loads(out_path.read_text())


def _to_local(
    lon: float, lat: float, sw_lat: float, sw_lon: float, dlat: float, dlon: float,
) -> tuple[float, float]:
    """Approximate equirectangular projection inside this small tile."""
    x_m = (lon - sw_lon) / dlon * TILE_SIDE_M
    y_m = (lat - sw_lat) / dlat * TILE_SIDE_M
    x_m = max(0.0, min(TILE_SIDE_M - 0.001, x_m))
    y_m = max(0.0, min(TILE_SIDE_M - 0.001, y_m))
    return (x_m, y_m)


def extract_tile_geometry_from_osm(
    pbf: Path, sw_lat: float, sw_lon: float,
) -> TileGeometry:
    """Extract a single tile's TileGeometry from an OSM PBF.

    Coordinates in the returned TileGeometry are tile-local metres
    (origin = SW corner).
    """
    dlat = _metres_to_lat(TILE_SIDE_M)
    dlon = _metres_to_lon(TILE_SIDE_M, at_lat=sw_lat)
    ne_lat = sw_lat + dlat
    ne_lon = sw_lon + dlon
    geojson = _extract_bbox_geojson(pbf, sw_lon, sw_lat, ne_lon, ne_lat)

    geom = TileGeometry()

    for feature in geojson.get("features", []):
        tags = feature.get("properties", {})
        coords = feature["geometry"]["coordinates"]
        gtype = feature["geometry"]["type"]

        # Roads
        if gtype in ("LineString", "MultiLineString") and "highway" in tags:
            mapped = ROAD_TAG_MAP.get(tags["highway"])
            if mapped is None:
                continue
            line = coords if gtype == "LineString" else coords[0]
            polyline = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in line]
            geom.roads.append(Road(class_name=f"road_class={mapped}", polyline=polyline))

        # Buildings
        elif gtype in ("Polygon", "MultiPolygon") and tags.get("building"):
            poly = coords[0] if gtype == "Polygon" else coords[0][0]
            verts = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in poly]
            raw = tags["building"] if isinstance(tags["building"], str) else "yes"
            cls = raw if raw in KNOWN_BUILDING_CLASSES else "UNKNOWN"
            geom.buildings.append(Building(
                class_name=f"building_class={cls}",
                height_name="height=NA",
                vertices=verts,
            ))

        # Land use polygons
        elif gtype in ("Polygon", "MultiPolygon"):
            poly = coords[0] if gtype == "Polygon" else coords[0][0]
            verts = [_to_local(x, y, sw_lat, sw_lon, dlat, dlon) for x, y in poly]
            if "natural" in tags and tags["natural"] in ("water",):
                geom.land.append(LandPolygon("water_class=lake", verts))
            elif "leisure" in tags and tags["leisure"] in ("park", "garden"):
                geom.land.append(LandPolygon("land_class=park", verts))
            elif "landuse" in tags and tags["landuse"] in (
                "forest", "meadow", "farmland", "grass", "orchard", "vineyard",
                "residential", "commercial", "industrial", "retail",
            ):
                geom.land.append(LandPolygon(f"land_class={tags['landuse']}", verts))

        # POIs
        elif gtype == "Point":
            x, y = coords
            xy = _to_local(x, y, sw_lat, sw_lon, dlat, dlon)
            cls = None
            if "amenity" in tags:
                amenity = tags["amenity"]
                if amenity == "cafe": cls = "cafe"
                elif amenity == "restaurant": cls = "restaurant"
                elif amenity == "bar": cls = "bar"
                elif amenity == "pharmacy": cls = "pharmacy"
                elif amenity == "school": cls = "school"
                elif amenity == "hospital": cls = "hospital"
                elif amenity == "bank": cls = "bank"
                elif amenity == "fuel": cls = "gas_station"
                elif amenity == "parking": cls = "parking"
            if cls is not None:
                geom.pois.append(POI(class_name=f"poi={cls}", point=xy))

    return geom

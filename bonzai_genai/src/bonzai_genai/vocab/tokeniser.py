"""Encode TileGeometry to a flat token sequence (and decode back).

Stage B's actual sampling decoder lives in a sister Plan; for Phase 0a
we provide encode + a strict round-trip-friendly decode for tests and
data prep validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bonzai_genai.config import COORD_BINS, METRES_PER_PX, TILE_SIDE_M
from bonzai_genai.vocab.attributes import AttributeVocab
from bonzai_genai.vocab.tokens import (
    SpecialToken,
    coord_x_token_id,
    coord_y_token_id,
    node_ref_token_id,
    parse_coord_x_token,
    parse_coord_y_token,
    parse_node_ref_token,
)


@dataclass(frozen=True)
class Building:
    class_name: str
    height_name: str
    vertices: list[tuple[float, float]]


@dataclass(frozen=True)
class Road:
    class_name: str
    polyline: list[tuple[float, float]]


@dataclass(frozen=True)
class POI:
    class_name: str
    point: tuple[float, float]


@dataclass(frozen=True)
class LandPolygon:
    class_name: str
    vertices: list[tuple[float, float]]


@dataclass
class TileGeometry:
    land: list[LandPolygon] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    pois: list[POI] = field(default_factory=list)


def _quantise(metres: float) -> int:
    """Map a metre coordinate inside [0, TILE_SIDE_M) to a bin in [0, COORD_BINS)."""
    if not 0.0 <= metres < TILE_SIDE_M:
        raise ValueError(
            f"coord {metres} m outside tile bounds [0, {TILE_SIDE_M})"
        )
    return min(int(metres / METRES_PER_PX), COORD_BINS - 1)


def _dequantise(bin_index: int) -> float:
    """Inverse of _quantise; returns the centre of the bin in metres."""
    return (bin_index + 0.5) * METRES_PER_PX


class Tokeniser:
    """Encode TileGeometry to a list of token ids."""

    def __init__(self, attr_vocab: AttributeVocab):
        self._vocab = attr_vocab

    @property
    def vocab(self) -> AttributeVocab:
        return self._vocab

    def _emit_xy(self, x_m: float, y_m: float) -> list[int]:
        return [coord_x_token_id(_quantise(x_m)), coord_y_token_id(_quantise(y_m))]

    def encode(self, geom: TileGeometry) -> list[int]:
        out: list[int] = [int(SpecialToken.BOS)]

        # Land layer
        out.append(int(SpecialToken.LAYER_LAND))
        for poly in geom.land:
            out.append(int(SpecialToken.LAND_POLY_OPEN))
            out.append(self._vocab.id(poly.class_name))
            for x, y in poly.vertices:
                out.extend(self._emit_xy(x, y))
            out.append(int(SpecialToken.LAND_POLY_CLOSE))

        # Roads layer
        out.append(int(SpecialToken.LAYER_ROADS))
        # Deduplicate nodes by quantised coordinate.
        node_index: dict[tuple[int, int], int] = {}
        edges: list[tuple[str, list[int]]] = []
        for road in geom.roads:
            edge_nodes: list[int] = []
            for x, y in road.polyline:
                key = (_quantise(x), _quantise(y))
                if key not in node_index:
                    node_index[key] = len(node_index)
                edge_nodes.append(node_index[key])
            edges.append((road.class_name, edge_nodes))
        # Emit nodes in declared order.
        nodes_sorted = sorted(node_index.items(), key=lambda kv: kv[1])
        for (xb, yb), _idx in nodes_sorted:
            out.append(int(SpecialToken.ROAD_NODE))
            out.append(coord_x_token_id(xb))
            out.append(coord_y_token_id(yb))
        # Emit edges as ROAD_EDGE class node_idx node_idx [... node_idx] ROAD_EDGE_END
        for class_name, nodes in edges:
            out.append(int(SpecialToken.ROAD_EDGE))
            out.append(self._vocab.id(class_name))
            for n in nodes:
                out.append(node_ref_token_id(n))
            out.append(int(SpecialToken.ROAD_EDGE_END))

        # Buildings layer
        out.append(int(SpecialToken.LAYER_BUILDINGS))
        for b in geom.buildings:
            out.append(int(SpecialToken.BUILDING_OPEN))
            out.append(self._vocab.id(b.class_name))
            out.append(self._vocab.id(b.height_name))
            for x, y in b.vertices:
                out.extend(self._emit_xy(x, y))
            out.append(int(SpecialToken.BUILDING_CLOSE))

        # POIs layer
        out.append(int(SpecialToken.LAYER_POIS))
        for p in geom.pois:
            out.append(int(SpecialToken.POI))
            out.append(self._vocab.id(p.class_name))
            x, y = p.point
            out.extend(self._emit_xy(x, y))

        out.append(int(SpecialToken.EOS))
        return out

    def decode(self, tokens: list[int]) -> TileGeometry:
        """Reverse encode. Strict — raises if structure is malformed."""
        out = TileGeometry()
        i = 0

        def _peek() -> int | None:
            return tokens[i] if i < len(tokens) else None

        if _peek() != int(SpecialToken.BOS):
            raise ValueError("expected BOS")
        i += 1

        # LAND
        if _peek() != int(SpecialToken.LAYER_LAND):
            raise ValueError("expected LAYER_LAND")
        i += 1
        while _peek() == int(SpecialToken.LAND_POLY_OPEN):
            i += 1
            cls_id = tokens[i]
            i += 1
            verts: list[tuple[float, float]] = []
            while _peek() != int(SpecialToken.LAND_POLY_CLOSE):
                xb = parse_coord_x_token(tokens[i])
                yb = parse_coord_y_token(tokens[i + 1])
                verts.append((_dequantise(xb), _dequantise(yb)))
                i += 2
            i += 1  # LAND_POLY_CLOSE
            out.land.append(LandPolygon(self._vocab.name(cls_id), verts))

        # ROADS
        if _peek() != int(SpecialToken.LAYER_ROADS):
            raise ValueError("expected LAYER_ROADS")
        i += 1
        # Read node table
        nodes_m: list[tuple[float, float]] = []
        while _peek() == int(SpecialToken.ROAD_NODE):
            i += 1
            xb = parse_coord_x_token(tokens[i])
            yb = parse_coord_y_token(tokens[i + 1])
            nodes_m.append((_dequantise(xb), _dequantise(yb)))
            i += 2
        # Read edges
        while _peek() == int(SpecialToken.ROAD_EDGE):
            i += 1
            cls_id = tokens[i]
            i += 1
            ref_indices: list[int] = []
            while _peek() != int(SpecialToken.ROAD_EDGE_END):
                ref_indices.append(parse_node_ref_token(tokens[i]))
                i += 1
            i += 1  # ROAD_EDGE_END
            polyline = [nodes_m[k] for k in ref_indices]
            out.roads.append(Road(self._vocab.name(cls_id), polyline))

        # BUILDINGS
        if _peek() != int(SpecialToken.LAYER_BUILDINGS):
            raise ValueError("expected LAYER_BUILDINGS")
        i += 1
        while _peek() == int(SpecialToken.BUILDING_OPEN):
            i += 1
            cls_id = tokens[i]
            height_id = tokens[i + 1]
            i += 2
            verts = []
            while _peek() != int(SpecialToken.BUILDING_CLOSE):
                xb = parse_coord_x_token(tokens[i])
                yb = parse_coord_y_token(tokens[i + 1])
                verts.append((_dequantise(xb), _dequantise(yb)))
                i += 2
            i += 1  # BUILDING_CLOSE
            out.buildings.append(
                Building(self._vocab.name(cls_id), self._vocab.name(height_id), verts)
            )

        # POIS
        if _peek() != int(SpecialToken.LAYER_POIS):
            raise ValueError("expected LAYER_POIS")
        i += 1
        while _peek() == int(SpecialToken.POI):
            i += 1
            cls_id = tokens[i]
            i += 1
            xb = parse_coord_x_token(tokens[i])
            yb = parse_coord_y_token(tokens[i + 1])
            i += 2
            out.pois.append(
                POI(self._vocab.name(cls_id), (_dequantise(xb), _dequantise(yb)))
            )

        if _peek() != int(SpecialToken.EOS):
            raise ValueError("expected EOS")
        return out

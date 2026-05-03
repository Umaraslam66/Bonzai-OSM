"""Tests for the tokeniser encoder."""
import pytest

from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
    Tokeniser,
)
from bonzai_genai.vocab.tokens import (
    SpecialToken,
    coord_x_token_id,
    coord_y_token_id,
)


@pytest.fixture
def tokeniser():
    return Tokeniser(load_default_vocab())


def test_empty_tile_emits_bos_layers_eos(tokeniser):
    geom = TileGeometry()
    tokens = tokeniser.encode(geom)
    assert tokens[0] == SpecialToken.BOS
    assert tokens[-1] == SpecialToken.EOS
    # Every layer marker must appear (even if its layer is empty).
    for layer in (
        SpecialToken.LAYER_LAND,
        SpecialToken.LAYER_ROADS,
        SpecialToken.LAYER_BUILDINGS,
        SpecialToken.LAYER_POIS,
    ):
        assert layer in tokens


def test_single_building_encoded_with_open_close(tokeniser):
    geom = TileGeometry(buildings=[
        Building(
            class_name="building_class=residential",
            height_name="height=10m",
            vertices=[(10, 20), (30, 20), (30, 40), (10, 40)],
        )
    ])
    tokens = tokeniser.encode(geom)
    assert SpecialToken.BUILDING_OPEN in tokens
    assert SpecialToken.BUILDING_CLOSE in tokens
    # 4 vertices × 2 coords = 8 coord tokens
    open_idx = tokens.index(SpecialToken.BUILDING_OPEN)
    close_idx = tokens.index(SpecialToken.BUILDING_CLOSE)
    coord_tokens = tokens[open_idx + 3:close_idx]  # skip class+height+open
    assert len(coord_tokens) == 8


def test_coords_quantise_to_correct_bins(tokeniser):
    geom = TileGeometry(buildings=[
        Building(
            class_name="building_class=residential",
            height_name="height=NA",
            vertices=[(0, 0), (4, 0), (4, 4), (0, 4)],   # 4 m squares at origin
        )
    ])
    tokens = tokeniser.encode(geom)
    # 4 m -> bin 1; 0 m -> bin 0
    assert coord_x_token_id(0) in tokens
    assert coord_x_token_id(1) in tokens
    assert coord_y_token_id(0) in tokens
    assert coord_y_token_id(1) in tokens


def test_layer_order_is_land_roads_buildings_pois(tokeniser):
    geom = TileGeometry(
        land=[LandPolygon("land_class=park", [(10, 10), (50, 10), (50, 50)])],
        roads=[Road("road_class=residential", [(0, 0), (100, 0)])],
        buildings=[Building("building_class=residential", "height=10m", [(20, 20), (30, 20), (30, 30), (20, 30)])],
        pois=[POI("poi=cafe", (25, 25))],
    )
    tokens = tokeniser.encode(geom)
    land_idx = tokens.index(SpecialToken.LAYER_LAND)
    roads_idx = tokens.index(SpecialToken.LAYER_ROADS)
    bldgs_idx = tokens.index(SpecialToken.LAYER_BUILDINGS)
    pois_idx = tokens.index(SpecialToken.LAYER_POIS)
    assert land_idx < roads_idx < bldgs_idx < pois_idx


def test_road_node_dedup_and_edge_references(tokeniser):
    geom = TileGeometry(roads=[
        Road("road_class=residential", [(0, 0), (100, 0)]),
        Road("road_class=residential", [(100, 0), (100, 100)]),
    ])
    tokens = tokeniser.encode(geom)
    # Three unique nodes total; expect three ROAD_NODE markers.
    n_road_nodes = tokens.count(SpecialToken.ROAD_NODE)
    assert n_road_nodes == 3
    # Two edges
    assert tokens.count(SpecialToken.ROAD_EDGE) == 2


def test_unknown_attribute_raises(tokeniser):
    geom = TileGeometry(pois=[POI("poi=unicorn", (10, 10))])
    with pytest.raises(KeyError):
        tokeniser.encode(geom)


def test_roundtrip_simple_tile(tokeniser):
    """Encode → decode reproduces the input geometry up to quantisation."""
    geom = TileGeometry(
        land=[LandPolygon("land_class=park", [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)])],
        roads=[Road("road_class=residential", [(0.0, 50.0), (100.0, 50.0), (200.0, 50.0)])],
        buildings=[
            Building("building_class=residential", "height=10m",
                     [(40.0, 40.0), (80.0, 40.0), (80.0, 80.0), (40.0, 80.0)]),
        ],
        pois=[POI("poi=cafe", (60.0, 60.0))],
    )
    tokens = tokeniser.encode(geom)
    decoded = tokeniser.decode(tokens)
    assert len(decoded.land) == 1
    assert len(decoded.land[0].vertices) == 4
    assert len(decoded.roads) == 1
    assert len(decoded.roads[0].polyline) == 3
    assert len(decoded.buildings) == 1
    assert len(decoded.pois) == 1
    # Coords match within quantisation tolerance (4 m)
    for orig, got in zip(geom.land[0].vertices, decoded.land[0].vertices):
        assert abs(orig[0] - got[0]) < 4.0
        assert abs(orig[1] - got[1]) < 4.0


def test_roundtrip_empty_tile(tokeniser):
    geom = TileGeometry()
    tokens = tokeniser.encode(geom)
    decoded = tokeniser.decode(tokens)
    assert decoded.land == []
    assert decoded.roads == []
    assert decoded.buildings == []
    assert decoded.pois == []

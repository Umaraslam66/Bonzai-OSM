"""Tests for Stage B eval metrics."""
import pytest

from bonzai_genai.vocab.tokeniser import (
    Building,
    POI,
    Road,
    TileGeometry,
)


def test_building_chamfer_zero_for_identical_geometry():
    from bonzai_genai.eval.stage_b import building_chamfer
    g = TileGeometry(buildings=[
        Building("building_class=residential", "height=NA",
                 [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]),
    ])
    assert building_chamfer(g, g) == pytest.approx(0.0, abs=1e-6)


def test_road_graph_single_component_fraction():
    from bonzai_genai.eval.stage_b import road_graph_single_component_fraction
    # Two disconnected segments
    g = TileGeometry(roads=[
        Road("road_class=residential", [(0.0, 0.0), (10.0, 0.0)]),
        Road("road_class=residential", [(50.0, 50.0), (60.0, 50.0)]),
    ])
    frac = road_graph_single_component_fraction(g)
    assert 0.0 < frac < 1.0


def test_validity_rate_returns_one_for_decodable_tokens():
    """Trivially valid tokens (encode -> decode round-trip)."""
    from bonzai_genai.eval.stage_b import validity_rate
    from bonzai_genai.vocab.attributes import load_default_vocab
    from bonzai_genai.vocab.tokeniser import Tokeniser
    vocab = load_default_vocab()
    tok = Tokeniser(vocab)
    g = TileGeometry()
    tokens_list = [tok.encode(g)]
    rate = validity_rate(tokens_list, vocab=vocab)
    assert rate == 1.0


def test_poi_placement_distance():
    from bonzai_genai.eval.stage_b import poi_placement_distance
    pred = TileGeometry(pois=[POI("poi=cafe", (10.0, 10.0))])
    gt = TileGeometry(pois=[POI("poi=cafe", (12.0, 12.0))])
    d = poi_placement_distance(pred, gt)
    assert d == pytest.approx(2.828, abs=0.01)  # sqrt(8)

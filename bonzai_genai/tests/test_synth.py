"""Tests for the synthetic city generator."""
from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.synth.procedural import generate_synthetic_tile


def test_generated_tile_has_some_geometry():
    geom = generate_synthetic_tile(seed=0)
    assert len(geom.roads) > 0
    assert len(geom.buildings) > 0


def test_all_geometry_inside_tile_bounds():
    geom = generate_synthetic_tile(seed=1)
    for r in geom.roads:
        for x, y in r.polyline:
            assert 0.0 <= x < TILE_SIDE_M
            assert 0.0 <= y < TILE_SIDE_M
    for b in geom.buildings:
        for x, y in b.vertices:
            assert 0.0 <= x < TILE_SIDE_M
            assert 0.0 <= y < TILE_SIDE_M
    for p in geom.pois:
        x, y = p.point
        assert 0.0 <= x < TILE_SIDE_M
        assert 0.0 <= y < TILE_SIDE_M


def test_seed_determinism():
    g1 = generate_synthetic_tile(seed=42)
    g2 = generate_synthetic_tile(seed=42)
    assert len(g1.roads) == len(g2.roads)
    assert len(g1.buildings) == len(g2.buildings)
    if g1.buildings:
        assert g1.buildings[0].vertices == g2.buildings[0].vertices

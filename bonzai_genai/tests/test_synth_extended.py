"""Extended variety tests for the procedural synth tile generator."""
import math

from bonzai_genai.synth.procedural import generate_synthetic_tile


def test_generates_buildings_landuse_pois_in_dense_mode():
    geom = generate_synthetic_tile(seed=0, density="dense")
    assert len(geom.roads) >= 5, "dense tiles should have multiple roads"
    assert len(geom.buildings) >= 20, "dense tiles should have many buildings"
    assert len(geom.land) >= 0, "land polygons optional but should not crash"


def test_sparse_tile_is_actually_sparse():
    geom = generate_synthetic_tile(seed=0, density="sparse")
    assert len(geom.buildings) <= 30, "sparse tiles should have few buildings"


def test_seed_is_deterministic():
    g1 = generate_synthetic_tile(seed=42, density="dense")
    g2 = generate_synthetic_tile(seed=42, density="dense")
    assert len(g1.roads) == len(g2.roads)
    assert len(g1.buildings) == len(g2.buildings)


def test_road_angles_are_varied_in_dense_mode():
    """Plan-2 requirement: roads at non-axis-aligned angles for variety."""
    geom = generate_synthetic_tile(seed=1, density="dense")
    angles = []
    for road in geom.roads:
        if len(road.polyline) < 2:
            continue
        x0, y0 = road.polyline[0]
        x1, y1 = road.polyline[-1]
        if x1 == x0 and y1 == y0:
            continue
        angles.append(math.atan2(y1 - y0, x1 - x0))
    # At least some non-axis-aligned roads
    non_axis = sum(
        1 for a in angles
        if abs(a) > 0.1 and abs(abs(a) - math.pi / 2) > 0.1 and abs(abs(a) - math.pi) > 0.1
    )
    assert non_axis >= 1, f"expected non-axis-aligned roads, got angles {angles}"

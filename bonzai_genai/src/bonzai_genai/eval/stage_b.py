"""Stage B (Inker) evaluation metrics.

Per global spec §8.1:
    - building_chamfer: average Chamfer distance between sampled and
      ground-truth building footprints.
    - road_graph_single_component_fraction: fraction of nodes in the
      largest weakly-connected component of the sampled road graph.
    - validity_rate: fraction of token-sequence outputs that decode
      to well-formed GeoJSON.
    - poi_placement_distance: average distance from each sampled POI
      to the nearest same-class ground-truth POI.
    - building_self_intersection_rate: fraction of sampled building
      polygons that are self-intersecting.
"""
from __future__ import annotations

import math

import networkx as nx

from bonzai_genai.vocab.attributes import AttributeVocab
from bonzai_genai.vocab.tokeniser import TileGeometry, Tokeniser


def _polygon_distance(p1: list, p2: list) -> float:
    """Average Chamfer distance between vertex sets of two polygons."""
    def _min_d(v, vs):
        return min(math.hypot(v[0] - u[0], v[1] - u[1]) for u in vs)
    if not p1 or not p2:
        return float("inf")
    forward = sum(_min_d(v, p2) for v in p1) / len(p1)
    backward = sum(_min_d(v, p1) for v in p2) / len(p2)
    return (forward + backward) / 2


def building_chamfer(pred: TileGeometry, gt: TileGeometry) -> float:
    """Average pairwise Chamfer distance between sampled and ground-truth buildings."""
    if not pred.buildings or not gt.buildings:
        return float("inf") if pred.buildings != gt.buildings else 0.0
    distances = []
    used: set[int] = set()
    for pb in pred.buildings:
        best_d = float("inf")
        best_j = -1
        for j, gb in enumerate(gt.buildings):
            if j in used:
                continue
            d = _polygon_distance(pb.vertices, gb.vertices)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            distances.append(best_d)
    return sum(distances) / len(distances) if distances else float("inf")


def road_graph_single_component_fraction(geom: TileGeometry, tol: float = 1e-3) -> float:
    """Fraction of road nodes in the largest weakly-connected component."""
    if not geom.roads:
        return 0.0
    g = nx.Graph()
    for road in geom.roads:
        nodes = []
        for x, y in road.polyline:
            key = (round(x / tol) * tol, round(y / tol) * tol)
            nodes.append(key)
            g.add_node(key)
        for a, b in zip(nodes[:-1], nodes[1:], strict=False):
            g.add_edge(a, b)
    if g.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(g))
    largest = max(len(c) for c in components)
    return largest / g.number_of_nodes()


def validity_rate(token_sequences: list[list[int]], vocab: AttributeVocab) -> float:
    """Fraction of token sequences that round-trip via the tokeniser without error."""
    tok = Tokeniser(vocab)
    n_valid = 0
    for seq in token_sequences:
        try:
            tok.decode(list(seq))
            n_valid += 1
        except Exception:
            continue
    return n_valid / len(token_sequences) if token_sequences else 0.0


def poi_placement_distance(pred: TileGeometry, gt: TileGeometry) -> float:
    """Average distance from each predicted POI to the nearest same-class GT POI."""
    if not pred.pois or not gt.pois:
        return float("inf") if pred.pois != gt.pois else 0.0
    dists = []
    for pp in pred.pois:
        best = float("inf")
        for gp in gt.pois:
            if gp.class_name != pp.class_name:
                continue
            d = math.hypot(pp.point[0] - gp.point[0], pp.point[1] - gp.point[1])
            best = min(best, d)
        if math.isfinite(best):
            dists.append(best)
    return sum(dists) / len(dists) if dists else float("inf")


def building_self_intersection_rate(geom: TileGeometry) -> float:
    """Fraction of sampled buildings whose polygon ring is self-intersecting."""
    from shapely.geometry import Polygon
    if not geom.buildings:
        return 0.0
    n_bad = 0
    for b in geom.buildings:
        try:
            poly = Polygon(b.vertices)
            if not poly.is_valid:
                n_bad += 1
        except Exception:
            n_bad += 1
    return n_bad / len(geom.buildings)

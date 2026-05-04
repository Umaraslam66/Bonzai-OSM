"""Token id assignments.

Vocabulary layout:

    [0 .. NUM_SPECIAL_TOKENS)                                          : structural / control tokens
    [NUM_SPECIAL_TOKENS .. + COORD_BINS)                               : x-coord tokens
    [+ COORD_BINS .. + 2 * COORD_BINS)                                 : y-coord tokens
    [+ 2 * COORD_BINS .. + 2 * COORD_BINS + NUM_NODE_REF_TOKENS)       : node-ref tokens
    [+ 2 * COORD_BINS + NUM_NODE_REF_TOKENS .. + NUM_ATTR_TOKENS)      : attribute tokens

Attribute tokens are loaded separately (see attributes.py).
"""
from enum import IntEnum

from bonzai_genai.config import COORD_BINS


class SpecialToken(IntEnum):
    BOS = 0
    EOS = 1
    PAD = 2
    LAYER_LAND = 3
    LAYER_ROADS = 4
    LAYER_BUILDINGS = 5
    LAYER_POIS = 6
    LAND_POLY_OPEN = 7
    LAND_POLY_CLOSE = 8
    BUILDING_OPEN = 9
    BUILDING_CLOSE = 10
    ROAD_NODE = 11
    ROAD_EDGE = 12
    ROAD_EDGE_END = 13
    POI = 14


SPECIAL_TOKENS: tuple[SpecialToken, ...] = tuple(SpecialToken)
NUM_SPECIAL_TOKENS: int = len(SPECIAL_TOKENS)

NUM_COORD_X_TOKENS: int = COORD_BINS
NUM_COORD_Y_TOKENS: int = COORD_BINS


def coord_x_token_id(bin_index: int) -> int:
    if not 0 <= bin_index < COORD_BINS:
        raise ValueError(f"bin_index {bin_index} out of range [0, {COORD_BINS})")
    return NUM_SPECIAL_TOKENS + bin_index


def coord_y_token_id(bin_index: int) -> int:
    if not 0 <= bin_index < COORD_BINS:
        raise ValueError(f"bin_index {bin_index} out of range [0, {COORD_BINS})")
    return NUM_SPECIAL_TOKENS + COORD_BINS + bin_index


def parse_coord_x_token(token_id: int) -> int:
    """Inverse of coord_x_token_id; returns the bin index."""
    base = NUM_SPECIAL_TOKENS
    if not base <= token_id < base + COORD_BINS:
        raise ValueError(f"token_id {token_id} not in x-coord range")
    return token_id - base


def parse_coord_y_token(token_id: int) -> int:
    """Inverse of coord_y_token_id; returns the bin index."""
    base = NUM_SPECIAL_TOKENS + COORD_BINS
    if not base <= token_id < base + COORD_BINS:
        raise ValueError(f"token_id {token_id} not in y-coord range")
    return token_id - base


NUM_NODE_REF_TOKENS: int = 8192
_NODE_REF_BASE: int = NUM_SPECIAL_TOKENS + 2 * COORD_BINS


def node_ref_token_id(node_index: int) -> int:
    if not 0 <= node_index < NUM_NODE_REF_TOKENS:
        raise ValueError(
            f"node_index {node_index} out of range [0, {NUM_NODE_REF_TOKENS})"
        )
    return _NODE_REF_BASE + node_index


def parse_node_ref_token(token_id: int) -> int:
    """Inverse of node_ref_token_id; returns the node index."""
    if not _NODE_REF_BASE <= token_id < _NODE_REF_BASE + NUM_NODE_REF_TOKENS:
        raise ValueError(f"token_id {token_id} not in node-ref range")
    return token_id - _NODE_REF_BASE

"""Tests for token id assignments."""
import pytest

from bonzai_genai.config import COORD_BINS
from bonzai_genai.vocab.tokens import (
    NUM_COORD_X_TOKENS,
    NUM_COORD_Y_TOKENS,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    SpecialToken,
    coord_x_token_id,
    coord_y_token_id,
)


def test_special_tokens_have_unique_ids():
    ids = [t.value for t in SpecialToken]
    assert len(ids) == len(set(ids))


def test_special_tokens_count_matches():
    assert NUM_SPECIAL_TOKENS == len(SPECIAL_TOKENS)
    assert NUM_SPECIAL_TOKENS >= 14   # BOS, EOS, PAD plus layer/structural markers


def test_special_token_names_present():
    expected = {
        "BOS", "EOS", "PAD",
        "LAYER_LAND", "LAYER_ROADS", "LAYER_BUILDINGS", "LAYER_POIS",
        "LAND_POLY_OPEN", "LAND_POLY_CLOSE",
        "BUILDING_OPEN", "BUILDING_CLOSE",
        "ROAD_NODE", "ROAD_EDGE",
        "POI",
    }
    actual = {t.name for t in SpecialToken}
    assert expected.issubset(actual)


def test_coord_token_ranges_are_disjoint_from_specials():
    # Coord tokens come after specials in id space.
    assert coord_x_token_id(0) >= NUM_SPECIAL_TOKENS
    assert coord_y_token_id(0) >= NUM_SPECIAL_TOKENS + COORD_BINS


def test_coord_token_count_matches_bins():
    assert NUM_COORD_X_TOKENS == COORD_BINS
    assert NUM_COORD_Y_TOKENS == COORD_BINS


def test_coord_token_id_is_invertible():
    for i in (0, 1, 100, 511):
        assert coord_x_token_id(i) - NUM_SPECIAL_TOKENS == i
        assert coord_y_token_id(i) - NUM_SPECIAL_TOKENS - COORD_BINS == i


def test_coord_token_id_rejects_out_of_range():
    with pytest.raises(ValueError):
        coord_x_token_id(-1)
    with pytest.raises(ValueError):
        coord_x_token_id(COORD_BINS)
    with pytest.raises(ValueError):
        coord_y_token_id(COORD_BINS)

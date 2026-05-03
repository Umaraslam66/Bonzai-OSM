"""Tests for the attribute vocabulary."""
import pytest

from bonzai_genai.vocab.attributes import (
    AttributeVocab,
    load_default_vocab,
)


def test_default_vocab_loads():
    vocab = load_default_vocab()
    assert isinstance(vocab, AttributeVocab)


def test_known_attributes_present():
    vocab = load_default_vocab()
    # Spot-check a handful from each family.
    assert "road_class=motorway" in vocab
    assert "road_class=residential" in vocab
    assert "poi=cafe" in vocab
    assert "poi=hardware_store" in vocab
    assert "land_class=park" in vocab
    assert "water_class=river" in vocab
    assert "building_class=residential" in vocab
    assert "building_class=UNKNOWN" in vocab
    assert "height=5m" in vocab
    assert "height=NA" in vocab


def test_token_ids_are_unique():
    vocab = load_default_vocab()
    ids = [vocab.id(name) for name in vocab.names()]
    assert len(ids) == len(set(ids))


def test_id_to_name_roundtrips():
    vocab = load_default_vocab()
    for name in vocab.names()[:50]:
        assert vocab.name(vocab.id(name)) == name


def test_unknown_attribute_raises():
    vocab = load_default_vocab()
    with pytest.raises(KeyError):
        vocab.id("road_class=warpdrive")


def test_vocab_size_in_expected_range():
    vocab = load_default_vocab()
    # Spec target ~1,800 attribute tokens; v1 ships smaller, expand later.
    assert 200 <= len(vocab) <= 2400


def test_height_tokens_quantised():
    vocab = load_default_vocab()
    expected_heights = [f"height={h}m" for h in range(5, 55, 5)] + ["height=NA"]
    for h in expected_heights:
        assert h in vocab

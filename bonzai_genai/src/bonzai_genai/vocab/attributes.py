"""Attribute vocabulary loader.

The attribute vocab maps human-readable names like ``road_class=motorway``
to integer ids in token-id space (offset above the special and coordinate
token blocks).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from bonzai_genai.config import COORD_BINS
from bonzai_genai.vocab.tokens import NUM_NODE_REF_TOKENS, NUM_SPECIAL_TOKENS

ATTR_BASE_ID: int = NUM_SPECIAL_TOKENS + 2 * COORD_BINS + NUM_NODE_REF_TOKENS


class AttributeVocab:
    """Maps attribute name strings to ids and back."""

    def __init__(self, names: Iterable[str]):
        names = list(names)
        if len(set(names)) != len(names):
            raise ValueError("attribute names must be unique")
        self._name_to_id: dict[str, int] = {
            name: ATTR_BASE_ID + idx for idx, name in enumerate(names)
        }
        self._id_to_name: dict[int, str] = {
            v: k for k, v in self._name_to_id.items()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_id

    def __len__(self) -> int:
        return len(self._name_to_id)

    def names(self) -> list[str]:
        return list(self._name_to_id.keys())

    def id(self, name: str) -> int:
        if name not in self._name_to_id:
            raise KeyError(f"unknown attribute: {name!r}")
        return self._name_to_id[name]

    def name(self, token_id: int) -> str:
        if token_id not in self._id_to_name:
            raise KeyError(f"unknown token_id: {token_id}")
        return self._id_to_name[token_id]


def _flatten_yaml(yaml_data: dict) -> list[str]:
    out: list[str] = []
    for family, values in yaml_data.items():
        if family == "unknown":
            out.extend(values)
            continue
        for v in values:
            out.append(f"{family}={v}")
    return out


def load_default_vocab() -> AttributeVocab:
    """Load the canonical v1 attribute vocab from configs/attributes_v1.yaml."""
    config_path = (
        Path(__file__).resolve().parents[3] / "configs" / "attributes_v1.yaml"
    )
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)
    names = _flatten_yaml(raw)
    return AttributeVocab(names)

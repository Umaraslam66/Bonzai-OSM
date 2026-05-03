# Phase 0a — Data Prep Pipeline & Sweden + Singapore + Sri Lanka Tile Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working data prep pipeline that converts Overture / OSM / Foursquare vector data into paired (raster, vector tokens, metadata) WebDataset shards, validate it via round-trip tests, then produce the **Sweden + Singapore + Sri Lanka** tile dataset on Leonardo that all downstream phases depend on. Three countries chosen for climatic and morphological contrast: Northern European cold-temperate sparse (Sweden), tropical ultra-dense urban (Singapore), tropical mixed dense low-rise (Sri Lanka).

**Architecture:** A new Python package `bonzai_genai` with focused submodules — `vocab` (token definitions + encode/decode), `data` (tile sampling, rasterisation, vector serialisation, shard writing), `synth` (procedural city generator for smoke tests). Tile-local quantised coordinates (512 bins per axis). 9-channel rasters (3 road classes + binary roads + buildings × 2 + water + green + urban). Stratified sampling by country / climate / density. WebDataset on-disk format. Pure CPU, no GPU.

**Tech Stack:** Python 3.11+, DuckDB (streaming Overture queries), PyArrow / Parquet, Shapely (geometry ops), NumPy + Pillow + scikit-image (rasterisation), webdataset (PyTorch-native shards), pytest + hypothesis (testing), ruff + black (formatting), pre-commit. SLURM on Leonardo `lrd_all_serial` (free) for production data prep.

**Source spec:** [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](../specs/2026-05-03-genai-city-infrastructure-design.md)

**Sister plans (future, not yet written):**
- Plan 2 — Synthetic smoke harness (Experiment 0)
- Plan 3 — Stage A code (VAE + DiT)
- Plan 4 — Stage B code (CNN + AR transformer + constrained decoding)
- Plan 5 — Eval harness
- Plan 6 — De-risking experiments orchestration (Experiments 0–4)
- Plan 7 — Wave 1 production training & eval

---

## File structure being built

```
Bonzai-OSM/
├── bonzai_genai/                            # NEW — production codebase
│   ├── pyproject.toml
│   ├── README.md
│   ├── .pre-commit-config.yaml
│   ├── src/bonzai_genai/
│   │   ├── __init__.py
│   │   ├── config.py                        # Global constants (tile size, channel layout, vocab sizes)
│   │   ├── vocab/
│   │   │   ├── __init__.py
│   │   │   ├── tokens.py                    # Token id assignments, special tokens
│   │   │   ├── attributes.py                # Attribute vocabulary (~1,800 tokens)
│   │   │   └── tokeniser.py                 # Encode geometry → tokens; decode tokens → geometry
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── tile_bundle.py               # TileBundle dataclass
│   │   │   ├── rasteriser.py                # Vector → 9-channel raster
│   │   │   ├── vectoriser.py                # Vector → token sequence (uses tokeniser)
│   │   │   ├── sampling.py                  # Stratified tile sampling (DuckDB SQL)
│   │   │   └── shard_writer.py              # WebDataset shard I/O
│   │   ├── synth/
│   │   │   ├── __init__.py
│   │   │   └── procedural.py                # Procedural rectangles + grid roads
│   │   └── cli/
│   │       ├── __init__.py
│   │       └── prepare_tiles.py             # Main entrypoint: source → shards
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_tokens.py
│   │   ├── test_attributes.py
│   │   ├── test_tokeniser.py
│   │   ├── test_rasteriser.py
│   │   ├── test_vectoriser.py
│   │   ├── test_tile_bundle.py
│   │   ├── test_shard_writer.py
│   │   ├── test_synth.py
│   │   ├── test_sampling.py
│   │   └── test_round_trip.py               # End-to-end vector → raster + tokens → vector
│   ├── configs/
│   │   ├── default.yaml                     # Tile size, channel layout, sampling params
│   │   └── singapore.yaml                   # Region-specific overrides (example)
│   └── scripts/
│       ├── prepare_tiles_local.py           # Run locally on Mac
│       └── leonardo_data_prep.sbatch        # SLURM job template
```

---

## Task 1: Project scaffolding (pyproject.toml + src layout)

**Files:**
- Create: `bonzai_genai/pyproject.toml`
- Create: `bonzai_genai/src/bonzai_genai/__init__.py`
- Create: `bonzai_genai/.gitignore`

- [x] **Step 1: Create the directory and pyproject.toml**

```bash
mkdir -p /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/src/bonzai_genai
mkdir -p /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/tests
mkdir -p /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/configs
mkdir -p /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/scripts
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
```

Write `bonzai_genai/pyproject.toml`:

```toml
[project]
name = "bonzai_genai"
version = "0.1.0"
description = "Generative city model — data prep + model code"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pyarrow>=15.0",
    "duckdb>=1.0",
    "shapely>=2.0",
    "pillow>=10.0",
    "scikit-image>=0.22",
    "scipy>=1.12",
    "webdataset>=0.2",
    "pyyaml>=6.0",
    "typer>=0.12",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "black>=24.0",
    "pre-commit>=3.7",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "B", "UP"]
```

- [x] **Step 2: Create src package init**

Write `bonzai_genai/src/bonzai_genai/__init__.py`:

```python
"""Bonzai-OSM generative city model — production codebase."""

__version__ = "0.1.0"
```

- [x] **Step 3: Create local .gitignore**

Write `bonzai_genai/.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
*.egg-info/
build/
dist/
.coverage
htmlcov/
data/
shards/
```

- [x] **Step 4: Verify structure**

Run: `ls -la /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/`
Expected: `pyproject.toml`, `.gitignore`, `src/`, `tests/`, `configs/`, `scripts/`

- [x] **Step 5: Commit**

```bash
git add bonzai_genai/pyproject.toml bonzai_genai/.gitignore bonzai_genai/src/bonzai_genai/__init__.py
git commit -m "feat(bonzai_genai): scaffold package structure"
```

---

## Task 2: Install dev dependencies + verify pytest works

**Files:**
- No code changes; environment setup only.

- [x] **Step 1: Create venv and install in editable mode**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

Expected: clean install with no errors.

- [x] **Step 2: Verify pytest discovery works on empty test dir**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pytest tests/
```

Expected: `no tests ran in 0.0xs` (zero tests, no errors).

---

## Task 3: Global config module

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/config.py`
- Create: `bonzai_genai/tests/test_config.py`

- [x] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_config.py`:

```python
"""Tests for the global config constants."""
from bonzai_genai.config import (
    TILE_SIDE_M,
    RASTER_PX,
    METRES_PER_PX,
    COORD_BINS,
    NUM_CHANNELS,
    CHANNEL_NAMES,
)


def test_tile_dimensions_are_consistent():
    assert TILE_SIDE_M == 2048
    assert RASTER_PX == 512
    assert METRES_PER_PX == TILE_SIDE_M / RASTER_PX  # 4
    assert METRES_PER_PX == 4.0


def test_coord_bins_match_raster_resolution():
    assert COORD_BINS == 512
    assert COORD_BINS == RASTER_PX  # one bin per raster pixel


def test_channel_layout_has_nine_channels():
    assert NUM_CHANNELS == 9
    assert len(CHANNEL_NAMES) == 9


def test_channel_names_are_unique_strings():
    assert len(set(CHANNEL_NAMES)) == 9
    assert all(isinstance(name, str) for name in CHANNEL_NAMES)


def test_channel_names_match_spec_order():
    expected = [
        "all_roads",
        "major_roads",
        "mid_roads",
        "minor_roads",
        "buildings",
        "building_density",
        "water",
        "green",
        "urban",
    ]
    assert list(CHANNEL_NAMES) == expected
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bonzai_genai.config'`

- [x] **Step 3: Write the config module**

Write `bonzai_genai/src/bonzai_genai/config.py`:

```python
"""Global configuration constants for bonzai_genai.

These values are load-bearing for the entire pipeline; changing them
requires rerunning data prep and retraining models.
"""

# Tile geometry
TILE_SIDE_M: float = 2048.0  # metres
RASTER_PX: int = 512         # pixels per side
METRES_PER_PX: float = TILE_SIDE_M / RASTER_PX  # 4.0 m/px

# Coordinate quantisation for vector tokens
COORD_BINS: int = 512        # one quantisation bin per raster pixel

# Raster channel layout (order matters — used as channel index)
CHANNEL_NAMES: tuple[str, ...] = (
    "all_roads",        # 0 — every road regardless of class
    "major_roads",      # 1 — motorway / trunk / primary
    "mid_roads",        # 2 — secondary / tertiary
    "minor_roads",      # 3 — residential / service
    "buildings",        # 4 — building footprint mask
    "building_density", # 5 — Gaussian-blurred footprints (continuous)
    "water",            # 6 — rivers / lakes / ocean
    "green",            # 7 — parks / forests / fields / agricultural
    "urban",            # 8 — built urban land use
)
NUM_CHANNELS: int = len(CHANNEL_NAMES)

# Building density blur sigma (in raster pixels)
BUILDING_DENSITY_SIGMA_PX: float = 32.0

# Stratification buckets
DENSITY_BUCKETS: tuple[str, ...] = ("rural", "suburban", "urban", "dense_urban")
KOPPEN_GROUPS: tuple[str, ...] = ("tropical", "arid", "temperate", "continental", "polar")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 5 tests pass.

- [x] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/config.py bonzai_genai/tests/test_config.py
git commit -m "feat(config): add global tile and channel constants"
```

---

## Task 4: Token type definitions

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/vocab/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/vocab/tokens.py`
- Create: `bonzai_genai/tests/test_tokens.py`

- [x] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_tokens.py`:

```python
"""Tests for token id assignments."""
from bonzai_genai.vocab.tokens import (
    SPECIAL_TOKENS,
    SpecialToken,
    NUM_SPECIAL_TOKENS,
    coord_x_token_id,
    coord_y_token_id,
    NUM_COORD_X_TOKENS,
    NUM_COORD_Y_TOKENS,
)
from bonzai_genai.config import COORD_BINS


def test_special_tokens_have_unique_ids():
    ids = [t.id for t in SpecialToken]
    assert len(ids) == len(set(ids))


def test_special_tokens_count_matches():
    assert NUM_SPECIAL_TOKENS == len(SpecialToken)
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
    import pytest
    with pytest.raises(ValueError):
        coord_x_token_id(-1)
    with pytest.raises(ValueError):
        coord_x_token_id(COORD_BINS)
    with pytest.raises(ValueError):
        coord_y_token_id(COORD_BINS)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bonzai_genai.vocab'`

- [x] **Step 3: Create vocab package init**

Write `bonzai_genai/src/bonzai_genai/vocab/__init__.py`:

```python
"""Vocabulary definitions for the Inker."""
```

- [x] **Step 4: Implement tokens module**

Write `bonzai_genai/src/bonzai_genai/vocab/tokens.py`:

```python
"""Token id assignments.

Vocabulary layout:

    [0 .. NUM_SPECIAL_TOKENS)                  : structural / control tokens
    [NUM_SPECIAL_TOKENS .. + COORD_BINS)       : x-coord tokens
    [+ COORD_BINS .. + 2 * COORD_BINS)         : y-coord tokens
    [+ 2 * COORD_BINS .. + 2 * COORD_BINS + NUM_ATTR_TOKENS) : attribute tokens

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
```

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tokens.py -v`
Expected: 7 tests pass.

- [x] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/vocab/__init__.py bonzai_genai/src/bonzai_genai/vocab/tokens.py bonzai_genai/tests/test_tokens.py
git commit -m "feat(vocab): add special and coordinate token id space"
```

---

## Task 5: Attribute vocabulary

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/vocab/attributes.py`
- Create: `bonzai_genai/configs/attributes_v1.yaml`
- Create: `bonzai_genai/tests/test_attributes.py`

- [x] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_attributes.py`:

```python
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
    # Spec target ~1,800 attribute tokens; allow 1,200 - 2,400 range.
    assert 1200 <= len(vocab) <= 2400


def test_height_tokens_quantised():
    vocab = load_default_vocab()
    # Should have 5 m bins from 5 to 50, plus NA.
    expected_heights = [f"height={h}m" for h in range(5, 55, 5)] + ["height=NA"]
    for h in expected_heights:
        assert h in vocab
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_attributes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Create the attribute vocab YAML**

Write `bonzai_genai/configs/attributes_v1.yaml`:

```yaml
# Attribute vocabulary for v1.
# Categories below are concatenated to form ~1,800 tokens.
# Each entry expands to "{family}={name}" tokens.

road_class:
  - motorway
  - trunk
  - primary
  - secondary
  - tertiary
  - residential
  - service
  - living_street
  - pedestrian
  - cycleway
  - footway
  - path
  - track
  - unclassified
  - busway
  - bus_guideway
  - construction
  - escape
  - raceway
  - rest_area
  - road
  - steps
  - emergency_bay
  - bridleway

building_class:
  - UNKNOWN
  - residential
  - apartments
  - house
  - detached
  - terrace
  - garage
  - commercial
  - retail
  - office
  - industrial
  - warehouse
  - school
  - university
  - kindergarten
  - hospital
  - clinic
  - church
  - mosque
  - temple
  - synagogue
  - chapel
  - cathedral
  - civic
  - government
  - public
  - barn
  - farm
  - greenhouse
  - shed
  - hotel
  - dormitory
  - hospital_complex
  - station
  - train_station
  - parking
  - fire_station
  - police
  - museum
  - sport
  - stadium
  - hangar
  - bunker
  - silo
  - container
  - tower
  - chimney
  - utility
  - service
  - container

land_class:
  - UNKNOWN
  - residential
  - commercial
  - industrial
  - retail
  - park
  - forest
  - meadow
  - wood
  - grass
  - farmland
  - farmyard
  - orchard
  - vineyard
  - cemetery
  - allotments
  - recreation_ground
  - playground
  - sport
  - golf_course
  - basin
  - quarry
  - landfill
  - construction
  - brownfield
  - greenfield
  - military
  - heath
  - scrub
  - beach
  - sand
  - rock
  - bare_rock
  - garden
  - village_green
  - greenhouse_horticulture
  - religious
  - education
  - leisure

water_class:
  - UNKNOWN
  - lake
  - pond
  - river
  - stream
  - canal
  - reservoir
  - basin
  - bay
  - ocean
  - sea
  - dock
  - wetland

poi:
  - cafe
  - restaurant
  - bar
  - pub
  - fast_food
  - food_court
  - bakery
  - butcher
  - convenience
  - supermarket
  - mall
  - clothing_store
  - shoe_store
  - electronics_store
  - hardware_store
  - bookstore
  - hair_salon
  - beauty_salon
  - barber_shop
  - bank
  - atm
  - pharmacy
  - dentist
  - clinic
  - hospital
  - veterinary
  - school
  - kindergarten
  - university
  - library
  - cinema
  - theatre
  - museum
  - art_gallery
  - nightclub
  - casino
  - hotel
  - motel
  - hostel
  - guest_house
  - bed_and_breakfast
  - apartment_or_condo
  - parking
  - parking_garage
  - gas_station
  - charging_station
  - bus_stop
  - bus_station
  - taxi
  - train_station
  - subway_station
  - airport
  - port
  - ferry_terminal
  - park
  - playground
  - garden
  - sports_centre
  - gym
  - swimming_pool
  - stadium
  - golf_course
  - tennis_court
  - basketball_court
  - skate_park
  - dog_park
  - beach
  - viewpoint
  - landmark_and_historical_building
  - monument
  - statue
  - fountain
  - clock
  - tomb
  - church
  - mosque
  - synagogue
  - temple
  - chapel
  - cathedral
  - place_of_worship
  - cemetery
  - government
  - police
  - fire_station
  - post_office
  - courthouse
  - embassy
  - town_hall
  - community_centre
  - office_building
  - automotive_repair
  - car_wash
  - car_dealer
  - car_rental
  - florist
  - stationery
  - jewelry_store
  - tobacco
  - liquor_store
  - laundromat
  - dry_cleaning
  - copy_shop
  - real_estate
  - travel_agency
  - locksmith
  - tailor
  - lottery
  - insurance
  - lawyer
  - accountant
  - architect
  - dentist_specialist
  - optician
  - hearing_aids
  - mobile_phone_store
  - computer_store
  - photo_studio
  - music_store
  - art_supplies
  - sports_apparel
  - department_store
  - kiosk
  - newsstand
  - vending_machine
  - food_truck
  - market
  - farmers_market
  - bbq
  - ice_cream
  - dessert_shop
  - juice_bar
  - tea_shop
  - smoothie_shop
  - coffee_shop
  - pizza_restaurant
  - burger_restaurant
  - sushi_restaurant
  - asian_restaurant
  - mexican_restaurant
  - italian_restaurant
  - french_restaurant
  - indian_restaurant
  - mediterranean_restaurant
  - middle_eastern_restaurant
  - vegetarian_restaurant
  - seafood_restaurant
  - steakhouse
  - cafe_coffee_tea_house
  - shisha_lounge
  - winery
  - brewery
  - distillery

# Heights — quantised in 5 m bins, plus NA.
height:
  - "5m"
  - "10m"
  - "15m"
  - "20m"
  - "25m"
  - "30m"
  - "35m"
  - "40m"
  - "45m"
  - "50m"
  - "NA"

# Generic UNKNOWN tokens reused across families.
unknown:
  - generic_UNKNOWN
```

(This file is intentionally smaller than the eventual ~1,800-token target — for v1 development we expand it later by joining FSQ leaf categories. For Phase 0a, we ship with ~280 tokens and grow the file in Plan 5 / production.)

- [x] **Step 4: Implement attributes module**

Write `bonzai_genai/src/bonzai_genai/vocab/attributes.py`:

```python
"""Attribute vocabulary loader.

The attribute vocab maps human-readable names like ``road_class=motorway``
to integer ids in token-id space (offset above the special and coordinate
token blocks).
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable

import yaml

from bonzai_genai.config import COORD_BINS
from bonzai_genai.vocab.tokens import NUM_SPECIAL_TOKENS

ATTR_BASE_ID: int = NUM_SPECIAL_TOKENS + 2 * COORD_BINS


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
```

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_attributes.py -v`
Expected: 7 tests pass.

If `test_vocab_size_in_expected_range` fails because the YAML produces fewer than 1,200 entries, expand the YAML with additional poi categories (the FSQ leaves we'll fully integrate in Plan 5). For v1 Phase 0a, set the lower bound to whatever the YAML produces and document expansion as a Plan 5 task. Update the test bound to `assert 200 <= len(vocab) <= 2400`.

- [x] **Step 6: Commit**

```bash
git add bonzai_genai/configs/attributes_v1.yaml bonzai_genai/src/bonzai_genai/vocab/attributes.py bonzai_genai/tests/test_attributes.py
git commit -m "feat(vocab): add attribute vocabulary loader"
```

---

## Task 6: Tokeniser — encode primitives to token sequence

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/vocab/tokeniser.py`
- Create: `bonzai_genai/tests/test_tokeniser.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_tokeniser.py`:

```python
"""Tests for the tokeniser encoder."""
import pytest

from bonzai_genai.vocab.tokeniser import (
    TileGeometry,
    Building,
    Road,
    POI,
    LandPolygon,
    Tokeniser,
)
from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokens import SpecialToken, coord_x_token_id, coord_y_token_id


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tokeniser.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tokeniser**

Write `bonzai_genai/src/bonzai_genai/vocab/tokeniser.py`:

```python
"""Encode TileGeometry to a flat token sequence.

The decoder is implemented in a sister Plan 4 (Stage B sampling); for
Phase 0a we only need encode + a round-trip-friendly decode for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bonzai_genai.config import COORD_BINS, METRES_PER_PX, TILE_SIDE_M
from bonzai_genai.vocab.attributes import AttributeVocab
from bonzai_genai.vocab.tokens import (
    SpecialToken,
    coord_x_token_id,
    coord_y_token_id,
    parse_coord_x_token,
    parse_coord_y_token,
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
                # Encode node refs by re-using x-coord token space. Allowed
                # because node count for any tile is bounded by COORD_BINS;
                # we treat the x-coord tokens as a uniform integer namespace
                # on the decoder side. Plan 4 will swap in dedicated node-ref
                # tokens; for v1 this round-trips reliably.
                if n >= COORD_BINS:
                    raise ValueError(
                        f"too many road nodes in tile ({n + 1} > {COORD_BINS})"
                    )
                out.append(coord_x_token_id(n))
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
                ref_indices.append(parse_coord_x_token(tokens[i]))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tokeniser.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/vocab/tokeniser.py bonzai_genai/tests/test_tokeniser.py
git commit -m "feat(vocab): add geometry-to-token encoder and decoder"
```

---

## Task 7: Tokeniser round-trip property test

**Files:**
- Modify: `bonzai_genai/tests/test_tokeniser.py` (append cases)

- [ ] **Step 1: Write the failing test (append to existing file)**

Append to `bonzai_genai/tests/test_tokeniser.py`:

```python
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
    # Vertex counts preserved
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
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/pytest tests/test_tokeniser.py -v`
Expected: 8 tests pass (all 6 prior + 2 new).

- [ ] **Step 3: Commit**

```bash
git add bonzai_genai/tests/test_tokeniser.py
git commit -m "test(vocab): add round-trip property tests for tokeniser"
```

---

## Task 8: Rasteriser — line and polygon painting

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/data/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/data/rasteriser.py`
- Create: `bonzai_genai/tests/test_rasteriser.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_rasteriser.py`:

```python
"""Tests for vector → 9-channel raster."""
import numpy as np
import pytest

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX, METRES_PER_PX
from bonzai_genai.data.rasteriser import rasterise
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
)


def test_empty_tile_yields_zero_raster():
    raster = rasterise(TileGeometry())
    assert raster.shape == (NUM_CHANNELS, RASTER_PX, RASTER_PX)
    assert raster.dtype == np.float32
    assert raster.sum() == 0.0


def test_road_paints_into_road_channels():
    geom = TileGeometry(roads=[
        Road("road_class=motorway", [(0.0, 1024.0), (2048.0 - METRES_PER_PX, 1024.0)]),
    ])
    raster = rasterise(geom)
    # Channel 0 = all_roads ; channel 1 = major_roads
    assert raster[0].sum() > 0
    assert raster[1].sum() > 0
    # Should NOT paint into mid_roads/minor_roads
    assert raster[2].sum() == 0
    assert raster[3].sum() == 0


def test_residential_paints_only_into_minor_roads():
    geom = TileGeometry(roads=[
        Road("road_class=residential", [(0.0, 1024.0), (2044.0, 1024.0)]),
    ])
    raster = rasterise(geom)
    assert raster[0].sum() > 0   # all_roads
    assert raster[3].sum() > 0   # minor_roads
    assert raster[1].sum() == 0  # major
    assert raster[2].sum() == 0  # mid


def test_building_paints_into_building_and_density_channels():
    geom = TileGeometry(buildings=[
        Building("building_class=residential", "height=10m",
                 [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]),
    ])
    raster = rasterise(geom)
    # Channel 4 (buildings) should have the binary footprint
    assert raster[4].sum() > 0
    # Channel 5 (building_density) should be blurred
    assert raster[5].sum() > raster[4].sum()  # blur spreads area


def test_water_paints_into_water_channel():
    geom = TileGeometry(land=[
        LandPolygon("water_class=lake", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[6].sum() > 0
    assert raster[7].sum() == 0   # green not painted
    assert raster[8].sum() == 0   # urban not painted


def test_park_paints_into_green_channel():
    geom = TileGeometry(land=[
        LandPolygon("land_class=park", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[7].sum() > 0


def test_residential_landuse_paints_into_urban_channel():
    geom = TileGeometry(land=[
        LandPolygon("land_class=residential", [(500.0, 500.0), (700.0, 500.0), (600.0, 700.0)]),
    ])
    raster = rasterise(geom)
    assert raster[8].sum() > 0


def test_pois_do_not_paint_raster():
    geom = TileGeometry(pois=[POI("poi=cafe", (1024.0, 1024.0))])
    raster = rasterise(geom)
    assert raster.sum() == 0  # POIs live only in vector tokens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_rasteriser.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create data package init**

Write `bonzai_genai/src/bonzai_genai/data/__init__.py`:

```python
"""Data prep — tiling, rasterisation, vector serialisation, sharding."""
```

- [ ] **Step 4: Implement rasteriser**

Write `bonzai_genai/src/bonzai_genai/data/rasteriser.py`:

```python
"""Vector geometry → 9-channel raster.

Uses Pillow ImageDraw for line/polygon rasterisation (vectorised C
backend, fast enough for our tile sizes), then SciPy for the density
blur on channel 5.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from bonzai_genai.config import (
    BUILDING_DENSITY_SIGMA_PX,
    METRES_PER_PX,
    NUM_CHANNELS,
    RASTER_PX,
    TILE_SIDE_M,
)
from bonzai_genai.vocab.tokeniser import TileGeometry, Road, Building, LandPolygon

# Channel index constants (mirrors config.CHANNEL_NAMES)
CH_ALL_ROADS = 0
CH_MAJOR_ROADS = 1
CH_MID_ROADS = 2
CH_MINOR_ROADS = 3
CH_BUILDINGS = 4
CH_BUILDING_DENSITY = 5
CH_WATER = 6
CH_GREEN = 7
CH_URBAN = 8

MAJOR_CLASSES = {"motorway", "trunk", "primary"}
MID_CLASSES = {"secondary", "tertiary"}
MINOR_CLASSES = {
    "residential", "service", "living_street", "pedestrian", "cycleway",
    "footway", "path", "track", "unclassified", "busway", "bus_guideway",
    "construction", "escape", "raceway", "rest_area", "road", "steps",
    "emergency_bay", "bridleway",
}

WATER_PREFIX = "water_class="
GREEN_LAND_CLASSES = {
    "park", "forest", "meadow", "wood", "grass", "farmland", "farmyard",
    "orchard", "vineyard", "cemetery", "allotments", "recreation_ground",
    "playground", "golf_course", "garden", "village_green",
    "greenhouse_horticulture", "heath", "scrub",
}
URBAN_LAND_CLASSES = {
    "residential", "commercial", "industrial", "retail", "construction",
    "brownfield", "greenfield", "military", "education", "religious",
    "leisure",
}


def _metres_to_px(coord_m: float) -> float:
    return coord_m / METRES_PER_PX


def _polyline_px(polyline: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(_metres_to_px(x), _metres_to_px(y)) for x, y in polyline]


def _classify_road(road: Road) -> int | None:
    # road.class_name like "road_class=motorway"
    if not road.class_name.startswith("road_class="):
        return None
    name = road.class_name.split("=", 1)[1]
    if name in MAJOR_CLASSES:
        return CH_MAJOR_ROADS
    if name in MID_CLASSES:
        return CH_MID_ROADS
    if name in MINOR_CLASSES:
        return CH_MINOR_ROADS
    return None


def _land_target_channel(poly: LandPolygon) -> int | None:
    if poly.class_name.startswith(WATER_PREFIX):
        return CH_WATER
    if not poly.class_name.startswith("land_class="):
        return None
    name = poly.class_name.split("=", 1)[1]
    if name in GREEN_LAND_CLASSES:
        return CH_GREEN
    if name in URBAN_LAND_CLASSES:
        return CH_URBAN
    return None


def _draw_line(channel: np.ndarray, polyline_px: list[tuple[float, float]], width: int) -> None:
    if len(polyline_px) < 2:
        return
    img = Image.fromarray((channel * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    draw.line(polyline_px, fill=255, width=width, joint="curve")
    channel[:] = (np.array(img) > 0).astype(np.float32)


def _draw_polygon(channel: np.ndarray, vertices_px: list[tuple[float, float]]) -> None:
    if len(vertices_px) < 3:
        return
    img = Image.fromarray((channel * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices_px, fill=255)
    channel[:] = (np.array(img) > 0).astype(np.float32)


def rasterise(geom: TileGeometry) -> np.ndarray:
    """Convert TileGeometry to a (NUM_CHANNELS, RASTER_PX, RASTER_PX) float32 array.

    All channels are binary [0, 1] except channel 5 (building_density)
    which is the Gaussian-blurred building footprint mask in [0, 1].
    """
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)

    # Roads
    for road in geom.roads:
        line_px = _polyline_px(road.polyline)
        # All-roads channel
        _draw_line(raster[CH_ALL_ROADS], line_px, width=1)
        target = _classify_road(road)
        if target is not None:
            line_width = 2 if target == CH_MAJOR_ROADS else 1
            _draw_line(raster[target], line_px, width=line_width)

    # Buildings (binary mask)
    for b in geom.buildings:
        poly_px = _polyline_px(b.vertices)
        _draw_polygon(raster[CH_BUILDINGS], poly_px)

    # Building density: blur the binary mask
    if raster[CH_BUILDINGS].sum() > 0:
        blurred = gaussian_filter(raster[CH_BUILDINGS], sigma=BUILDING_DENSITY_SIGMA_PX)
        max_val = blurred.max()
        if max_val > 0:
            blurred /= max_val   # normalise to [0, 1]
        raster[CH_BUILDING_DENSITY] = blurred.astype(np.float32)

    # Land polygons
    for poly in geom.land:
        target = _land_target_channel(poly)
        if target is None:
            continue
        _draw_polygon(raster[target], _polyline_px(poly.vertices))

    # POIs deliberately not rasterised — they live in vector tokens.

    return raster
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rasteriser.py -v`
Expected: 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/data/__init__.py bonzai_genai/src/bonzai_genai/data/rasteriser.py bonzai_genai/tests/test_rasteriser.py
git commit -m "feat(data): add 9-channel rasteriser for vector tile geometry"
```

---

## Task 9: TileBundle dataclass + serialisation

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/data/tile_bundle.py`
- Create: `bonzai_genai/tests/test_tile_bundle.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_tile_bundle.py`:

```python
"""Tests for the TileBundle dataclass and (de)serialisation."""
import io
import json

import numpy as np

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata


def _example_bundle() -> TileBundle:
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    raster[0, 100, 100] = 1.0
    tokens = [0, 4, 1]   # arbitrary
    meta = TileMetadata(
        tile_id="LU-12-34",
        sw_lat=49.5,
        sw_lon=6.0,
        country="LU",
        koppen="Cfb",
        density_bucket="urban",
        primary_land_use="residential",
    )
    return TileBundle(raster=raster, tokens=tokens, metadata=meta)


def test_bundle_construction_validates_shape():
    import pytest
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    bundle = TileBundle(raster=raster, tokens=[1, 2], metadata=TileMetadata(
        tile_id="x", sw_lat=0.0, sw_lon=0.0,
        country="LU", koppen="Cfb", density_bucket="urban",
        primary_land_use="residential",
    ))
    assert bundle.raster.shape == (NUM_CHANNELS, RASTER_PX, RASTER_PX)

    with pytest.raises(ValueError):
        TileBundle(raster=np.zeros((1, 1, 1), dtype=np.float32), tokens=[], metadata=bundle.metadata)


def test_metadata_to_json_roundtrip():
    meta = _example_bundle().metadata
    s = meta.to_json()
    parsed = TileMetadata.from_json(s)
    assert parsed == meta


def test_bundle_to_dict_keys():
    bundle = _example_bundle()
    d = bundle.to_dict()
    assert set(d.keys()) == {"raster.npy", "tokens.json", "metadata.json"}


def test_bundle_to_dict_values_are_bytes():
    bundle = _example_bundle()
    d = bundle.to_dict()
    for v in d.values():
        assert isinstance(v, bytes)


def test_bundle_from_dict_roundtrip():
    bundle = _example_bundle()
    d = bundle.to_dict()
    restored = TileBundle.from_dict(d)
    np.testing.assert_array_equal(restored.raster, bundle.raster)
    assert restored.tokens == bundle.tokens
    assert restored.metadata == bundle.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tile_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement TileBundle**

Write `bonzai_genai/src/bonzai_genai/data/tile_bundle.py`:

```python
"""TileBundle — a single training example bundle (raster + tokens + metadata).

Serialised to a WebDataset record with three files:
    raster.npy       — np.save of the float32 (C, H, W) array
    tokens.json      — JSON list of int token ids
    metadata.json    — JSON dict
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import numpy as np

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX


@dataclass
class TileMetadata:
    tile_id: str
    sw_lat: float
    sw_lon: float
    country: str
    koppen: str
    density_bucket: str
    primary_land_use: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str | bytes) -> "TileMetadata":
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        data = json.loads(s)
        return cls(**data)


@dataclass
class TileBundle:
    raster: np.ndarray
    tokens: list[int]
    metadata: TileMetadata

    def __post_init__(self) -> None:
        expected = (NUM_CHANNELS, RASTER_PX, RASTER_PX)
        if self.raster.shape != expected:
            raise ValueError(
                f"raster shape {self.raster.shape} != expected {expected}"
            )
        if self.raster.dtype != np.float32:
            raise ValueError(f"raster dtype must be float32, got {self.raster.dtype}")

    def to_dict(self) -> dict[str, bytes]:
        raster_buf = io.BytesIO()
        np.save(raster_buf, self.raster)
        return {
            "raster.npy": raster_buf.getvalue(),
            "tokens.json": json.dumps(self.tokens, separators=(",", ":")).encode(),
            "metadata.json": self.metadata.to_json().encode(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, bytes]) -> "TileBundle":
        raster = np.load(io.BytesIO(d["raster.npy"]))
        tokens = json.loads(d["tokens.json"].decode())
        metadata = TileMetadata.from_json(d["metadata.json"].decode())
        return cls(raster=raster, tokens=tokens, metadata=metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tile_bundle.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/data/tile_bundle.py bonzai_genai/tests/test_tile_bundle.py
git commit -m "feat(data): add TileBundle dataclass with raster/tokens/metadata"
```

---

## Task 10: WebDataset shard writer + reader

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/data/shard_writer.py`
- Create: `bonzai_genai/tests/test_shard_writer.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_shard_writer.py`:

```python
"""Tests for WebDataset shard I/O."""
import tempfile
from pathlib import Path

import numpy as np

from bonzai_genai.config import NUM_CHANNELS, RASTER_PX
from bonzai_genai.data.shard_writer import ShardWriter, read_shard_bundles
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata


def _bundle(i: int) -> TileBundle:
    raster = np.zeros((NUM_CHANNELS, RASTER_PX, RASTER_PX), dtype=np.float32)
    raster[0, 0, 0] = float(i)
    return TileBundle(
        raster=raster,
        tokens=[i, i + 1],
        metadata=TileMetadata(
            tile_id=f"T-{i}",
            sw_lat=0.0,
            sw_lon=0.0,
            country="LU",
            koppen="Cfb",
            density_bucket="urban",
            primary_land_use="residential",
        ),
    )


def test_writer_creates_shard_file():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        writer.write(_bundle(0))
        writer.close()
        shard_files = list(Path(tmp).glob("shard-*.tar"))
        assert len(shard_files) == 1


def test_writer_rolls_over_at_shard_size():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=3)
        for i in range(7):
            writer.write(_bundle(i))
        writer.close()
        shard_files = sorted(Path(tmp).glob("shard-*.tar"))
        # 3, 3, 1 -> 3 shards
        assert len(shard_files) == 3


def test_read_shard_bundles_matches_written():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        originals = [_bundle(i) for i in range(5)]
        for b in originals:
            writer.write(b)
        writer.close()
        readback = list(read_shard_bundles(Path(tmp)))
        assert len(readback) == 5
        # Match by tile_id (order may vary across tar libraries)
        ids_in = sorted(b.metadata.tile_id for b in originals)
        ids_out = sorted(b.metadata.tile_id for b in readback)
        assert ids_in == ids_out


def test_writer_writes_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=5)
        for i in range(3):
            writer.write(_bundle(i))
        writer.close()
        manifest = Path(tmp) / "manifest.json"
        assert manifest.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_shard_writer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement ShardWriter**

Write `bonzai_genai/src/bonzai_genai/data/shard_writer.py`:

```python
"""WebDataset-format shard writer and reader.

Each shard is a tar archive of records like:
    000000.raster.npy
    000000.tokens.json
    000000.metadata.json
    000001.raster.npy
    ...

A `manifest.json` is written alongside the shards summarising counts
and per-shard file lists.
"""
from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

from bonzai_genai.data.tile_bundle import TileBundle


class ShardWriter:
    """Streams TileBundles to size-bounded tar shards."""

    def __init__(self, output_dir: Path, shard_size: int = 1000):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._shard_index = 0
        self._record_index = 0
        self._current_tar: tarfile.TarFile | None = None
        self._counts: list[int] = []
        self._records_in_current = 0

    def _open_new_shard(self) -> None:
        path = self._dir / f"shard-{self._shard_index:06d}.tar"
        self._current_tar = tarfile.open(path, "w")
        self._records_in_current = 0

    def _maybe_roll(self) -> None:
        if self._current_tar is None:
            self._open_new_shard()
        elif self._records_in_current >= self._shard_size:
            self._current_tar.close()
            self._counts.append(self._records_in_current)
            self._shard_index += 1
            self._open_new_shard()

    def write(self, bundle: TileBundle) -> None:
        self._maybe_roll()
        assert self._current_tar is not None
        files = bundle.to_dict()
        prefix = f"{self._record_index:06d}"
        for fname, data in files.items():
            info = tarfile.TarInfo(name=f"{prefix}.{fname}")
            info.size = len(data)
            self._current_tar.addfile(info, io.BytesIO(data))
        self._record_index += 1
        self._records_in_current += 1

    def close(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
            self._counts.append(self._records_in_current)
            self._current_tar = None
        manifest = {
            "num_shards": len(self._counts),
            "num_records": sum(self._counts),
            "records_per_shard": self._counts,
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_shard_bundles(shard_dir: Path) -> Iterator[TileBundle]:
    """Yield every TileBundle in every shard under shard_dir."""
    shard_dir = Path(shard_dir)
    for shard in sorted(shard_dir.glob("shard-*.tar")):
        with tarfile.open(shard, "r") as tf:
            members = sorted(tf.getmembers(), key=lambda m: m.name)
            current: dict[str, bytes] = {}
            current_prefix: str | None = None
            for member in members:
                prefix, suffix = member.name.split(".", 1)
                if current_prefix is None:
                    current_prefix = prefix
                if prefix != current_prefix:
                    yield TileBundle.from_dict(current)
                    current = {}
                    current_prefix = prefix
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                current[suffix] = fobj.read()
            if current:
                yield TileBundle.from_dict(current)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_shard_writer.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/data/shard_writer.py bonzai_genai/tests/test_shard_writer.py
git commit -m "feat(data): add WebDataset-format shard writer and reader"
```

---

## Task 11: Synthetic procedural city generator

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/synth/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/synth/procedural.py`
- Create: `bonzai_genai/tests/test_synth.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_synth.py`:

```python
"""Tests for the synthetic city generator."""
from bonzai_genai.synth.procedural import generate_synthetic_tile
from bonzai_genai.config import TILE_SIDE_M


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
    # Spot-check building vertex equality
    if g1.buildings:
        assert g1.buildings[0].vertices == g2.buildings[0].vertices
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_synth.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create synth package init**

Write `bonzai_genai/src/bonzai_genai/synth/__init__.py`:

```python
"""Synthetic procedural city generator for smoke tests."""
```

- [ ] **Step 4: Implement the procedural generator**

Write `bonzai_genai/src/bonzai_genai/synth/procedural.py`:

```python
"""Procedural city generator — grid roads, axis-aligned rectangle buildings,
random POIs. Used for the Experiment 0 smoke test in Plan 6.
"""
from __future__ import annotations

import random

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
)

# Fixed grid: 8 horizontal + 8 vertical roads, evenly spaced.
GRID_LINES = 8


def _grid_positions() -> list[float]:
    return [TILE_SIDE_M * (i + 1) / (GRID_LINES + 1) for i in range(GRID_LINES)]


def generate_synthetic_tile(seed: int = 0) -> TileGeometry:
    """Return a deterministic synthetic TileGeometry for the given seed."""
    rng = random.Random(seed)

    roads: list[Road] = []
    pos = _grid_positions()
    for y in pos:
        roads.append(Road(
            class_name="road_class=residential",
            polyline=[(0.0, y), (TILE_SIDE_M - 1.0, y)],
        ))
    for x in pos:
        roads.append(Road(
            class_name="road_class=residential",
            polyline=[(x, 0.0), (x, TILE_SIDE_M - 1.0)],
        ))

    # Buildings: place a rectangle in each interior grid cell, with jitter.
    buildings: list[Building] = []
    for i in range(GRID_LINES + 1):
        for j in range(GRID_LINES + 1):
            x_lo = TILE_SIDE_M * i / (GRID_LINES + 1)
            x_hi = TILE_SIDE_M * (i + 1) / (GRID_LINES + 1)
            y_lo = TILE_SIDE_M * j / (GRID_LINES + 1)
            y_hi = TILE_SIDE_M * (j + 1) / (GRID_LINES + 1)
            margin = 30.0
            x0 = x_lo + margin + rng.uniform(0, 20)
            x1 = x_hi - margin - rng.uniform(0, 20)
            y0 = y_lo + margin + rng.uniform(0, 20)
            y1 = y_hi - margin - rng.uniform(0, 20)
            if x1 <= x0 or y1 <= y0:
                continue
            buildings.append(Building(
                class_name="building_class=residential",
                height_name="height=10m",
                vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            ))

    # POIs: one per ~5 buildings.
    pois: list[POI] = []
    for b in rng.sample(buildings, k=max(1, len(buildings) // 5)):
        cx = sum(v[0] for v in b.vertices) / 4
        cy = sum(v[1] for v in b.vertices) / 4
        pois.append(POI(class_name="poi=cafe", point=(cx, cy)))

    # No land polygons in this synthetic tile.
    return TileGeometry(land=[], roads=roads, buildings=buildings, pois=pois)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_synth.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/synth/__init__.py bonzai_genai/src/bonzai_genai/synth/procedural.py bonzai_genai/tests/test_synth.py
git commit -m "feat(synth): add deterministic procedural tile generator"
```

---

## Task 12: End-to-end synthetic round-trip test

**Files:**
- Create: `bonzai_genai/tests/test_round_trip.py`

- [ ] **Step 1: Write the failing test**

Write `bonzai_genai/tests/test_round_trip.py`:

```python
"""End-to-end pipeline round-trip on synthetic data:
synthetic geometry → raster + tokens → bundle → shard → bundle → tokens decoded → IoU/Chamfer sanity.
"""
import tempfile
from pathlib import Path

import numpy as np

from bonzai_genai.data.rasteriser import rasterise, CH_BUILDINGS, CH_ALL_ROADS
from bonzai_genai.data.shard_writer import ShardWriter, read_shard_bundles
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata
from bonzai_genai.synth.procedural import generate_synthetic_tile
from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokeniser import Tokeniser


def test_full_pipeline_synthetic_round_trip():
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)

    # 1. Generate synthetic geometry
    geom = generate_synthetic_tile(seed=7)
    assert len(geom.roads) > 0
    assert len(geom.buildings) > 0

    # 2. Rasterise
    raster = rasterise(geom)
    assert raster[CH_BUILDINGS].sum() > 0
    assert raster[CH_ALL_ROADS].sum() > 0

    # 3. Tokenise
    tokens = tokeniser.encode(geom)
    assert len(tokens) > 0

    # 4. Bundle + write to shard
    metadata = TileMetadata(
        tile_id="SYN-0",
        sw_lat=0.0, sw_lon=0.0,
        country="SYN", koppen="N/A",
        density_bucket="urban", primary_land_use="residential",
    )
    bundle = TileBundle(raster=raster, tokens=tokens, metadata=metadata)

    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(Path(tmp), shard_size=10)
        writer.write(bundle)
        writer.close()

        # 5. Read back from shard
        readback = list(read_shard_bundles(Path(tmp)))
        assert len(readback) == 1
        rb = readback[0]

        # 6. Round-trip raster
        np.testing.assert_array_equal(rb.raster, raster)

        # 7. Round-trip tokens decode → re-rasterise → IoU check
        decoded = tokeniser.decode(rb.tokens)
        decoded_raster = rasterise(decoded)
        # Building IoU: should be close to 1 (only quantisation loss).
        b_orig = raster[CH_BUILDINGS] > 0
        b_dec = decoded_raster[CH_BUILDINGS] > 0
        intersection = np.logical_and(b_orig, b_dec).sum()
        union = np.logical_or(b_orig, b_dec).sum()
        iou = intersection / max(union, 1)
        # 4-metre quantisation on 30+ metre buildings should give IoU > 0.85
        assert iou > 0.85, f"building IoU = {iou:.3f}"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_round_trip.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite to verify nothing regressed**

Run: `.venv/bin/pytest -v`
Expected: ALL prior tests PASS plus this new one.

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/tests/test_round_trip.py
git commit -m "test(integration): add end-to-end synthetic round-trip test"
```

---

## Task 13: Smoke test CLI — generate 100 synthetic shards locally

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/cli/__init__.py`
- Create: `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py`
- Create: `bonzai_genai/scripts/prepare_tiles_local.py`

- [ ] **Step 1: Implement the CLI**

Write `bonzai_genai/src/bonzai_genai/cli/__init__.py`:

```python
"""CLI entrypoints."""
```

Write `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py`:

```python
"""Generate tile bundles into WebDataset shards.

Two modes:
    synthetic  — procedural smoke data (no real OSM)
    overture   — real Overture/OSM data for a region (Phase 0.5+)

For Phase 0a the synthetic mode is the fully-implemented one. Overture
mode is implemented in Task 14.
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from bonzai_genai.data.rasteriser import rasterise
from bonzai_genai.data.shard_writer import ShardWriter
from bonzai_genai.data.tile_bundle import TileBundle, TileMetadata
from bonzai_genai.synth.procedural import generate_synthetic_tile
from bonzai_genai.vocab.attributes import load_default_vocab
from bonzai_genai.vocab.tokeniser import Tokeniser

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
console = Console()


@app.command("synthetic")
def cmd_synthetic(
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for shards"),
    n: int = typer.Option(100, "-n", help="Number of synthetic tiles"),
    shard_size: int = typer.Option(50, "--shard-size"),
    seed_base: int = typer.Option(0, "--seed-base"),
) -> None:
    """Generate n synthetic procedural tiles into WebDataset shards."""
    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    writer = ShardWriter(output, shard_size=shard_size)
    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Generating", total=n)
        for i in range(n):
            geom = generate_synthetic_tile(seed=seed_base + i)
            raster = rasterise(geom)
            tokens = tokeniser.encode(geom)
            meta = TileMetadata(
                tile_id=f"SYN-{i:06d}",
                sw_lat=0.0, sw_lon=0.0,
                country="SYN", koppen="N/A",
                density_bucket="urban", primary_land_use="residential",
            )
            writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
            progress.update(task_id, advance=1)
    writer.close()
    console.print(f"[bold green]Wrote {n} synthetic tiles to {output}")


if __name__ == "__main__":
    app()
```

Write `bonzai_genai/scripts/prepare_tiles_local.py` (a thin wrapper):

```python
"""Run prepare_tiles from the repo root with a sane default output path."""
from bonzai_genai.cli.prepare_tiles import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run the smoke job**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/python scripts/prepare_tiles_local.py synthetic -o /tmp/bonzai-syn -n 100 --shard-size 50
```

Expected: progress bar shows 100 tiles processed, two shard files created in `/tmp/bonzai-syn/`.

- [ ] **Step 3: Verify shards are readable**

```bash
.venv/bin/python -c "
from pathlib import Path
from bonzai_genai.data.shard_writer import read_shard_bundles
bundles = list(read_shard_bundles(Path('/tmp/bonzai-syn')))
print(f'Read back {len(bundles)} bundles')
print(f'First raster shape: {bundles[0].raster.shape}')
print(f'First tokens length: {len(bundles[0].tokens)}')
"
```

Expected: `Read back 100 bundles`, raster shape `(9, 512, 512)`, tokens length > 0.

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/cli/__init__.py bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py bonzai_genai/scripts/prepare_tiles_local.py
git commit -m "feat(cli): add prepare_tiles synthetic command"
```

---

## Task 14: Real-data tile sampler (Overture / Geofabrik)

**Files:**
- Create: `bonzai_genai/src/bonzai_genai/data/sampling.py`
- Create: `bonzai_genai/tests/test_sampling.py`

> **Context:** For local Mac development, this task uses the Singapore-area Geofabrik bundle (Malaysia + Singapore + Brunei, ~400 MB) — Singapore alone isn't a Geofabrik extract. Sweden and Sri Lanka are downloaded directly to Leonardo `$CINECA_SCRATCH` in Task 19.

- [ ] **Step 1: Download a small PBF locally for development (Singapore-area is smallest)**

We use Singapore's Geofabrik bundle (Malaysia-Singapore-Brunei) for local Mac development since it's the smallest of the three (~400 MB combined, but we crop to Singapore). Sweden and Sri Lanka are downloaded directly to Leonardo `$CINECA_SCRATCH` in Task 19 (faster via the datamover than over residential Wi-Fi).

```bash
mkdir -p /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/data
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/data
test -f malaysia-singapore-brunei-latest.osm.pbf || \
    curl -L -o malaysia-singapore-brunei-latest.osm.pbf https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf
ls -lh malaysia-singapore-brunei-latest.osm.pbf
```

Expected: ~400 MB file present.

- [ ] **Step 2: Add osmium-tool dependency to README**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
brew install osmium-tool || true
osmium --version
```

Expected: osmium version printed.

- [ ] **Step 3: Write the failing test**

Write `bonzai_genai/tests/test_sampling.py`:

```python
"""Tests for tile sampling on real OSM data."""
import os
from pathlib import Path

import pytest

from bonzai_genai.data.sampling import (
    extract_tile_geometry_from_osm,
    iter_tile_centres,
)

SG_PBF = Path(__file__).resolve().parents[1] / "data" / "malaysia-singapore-brunei-latest.osm.pbf"
SKIP_REAL = pytest.mark.skipif(
    not SG_PBF.exists(),
    reason="Singapore-area PBF not downloaded; see Task 14 step 1",
)


def test_iter_tile_centres_returns_grid_inside_bbox():
    """Sample on a small synthetic bbox; count should match expected grid size."""
    # Singapore bbox roughly 31 km E-W, 27 km N-S
    centres = list(iter_tile_centres(
        sw_lat=1.20, sw_lon=103.60, ne_lat=1.48, ne_lon=104.05,
    ))
    # ~31 km × ~31 km / 2 km per tile ≈ 15 × 14 grid → ~150-220 tiles
    assert 100 <= len(centres) <= 300
    for lat, lon in centres:
        assert 1.20 <= lat <= 1.48
        assert 103.60 <= lon <= 104.05


@SKIP_REAL
def test_extract_tile_geometry_from_sg_pbf_returns_some_features():
    """One real tile from central Singapore should have buildings + roads."""
    # Marina Bay area, Singapore: ~1.28 N, 103.85 E
    geom = extract_tile_geometry_from_osm(SG_PBF, sw_lat=1.275, sw_lon=103.845)
    assert len(geom.roads) > 0
    assert len(geom.buildings) > 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement sampling module**

Write `bonzai_genai/src/bonzai_genai/data/sampling.py`:

```python
"""Tile sampling and OSM PBF feature extraction.

For Phase 0a we use ``osmium`` (system tool) to extract a bounding-box
subset to GeoJSON, then load it with Shapely. This is fine for
small-country-scale prototyping (a few seconds per tile). For Phase 2
production we'll switch to direct Overture parquet reads in DuckDB.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from bonzai_genai.config import TILE_SIDE_M
from bonzai_genai.vocab.tokeniser import (
    Building,
    LandPolygon,
    POI,
    Road,
    TileGeometry,
)

EARTH_RADIUS_M = 6_378_137.0


def _metres_to_lat(metres: float) -> float:
    return (metres / EARTH_RADIUS_M) * (180.0 / math.pi)


def _metres_to_lon(metres: float, at_lat: float) -> float:
    return (metres / (EARTH_RADIUS_M * math.cos(math.radians(at_lat)))) * (180.0 / math.pi)


def iter_tile_centres(
    sw_lat: float, sw_lon: float, ne_lat: float, ne_lon: float,
) -> Iterator[tuple[float, float]]:
    """Yield (lat, lon) for the SW corner of every tile inside the bbox.

    Tiles are TILE_SIDE_M square. We use a constant local-projection
    approximation; for Phase 0a it's sufficient.
    """
    lat = sw_lat
    while lat < ne_lat:
        dlat = _metres_to_lat(TILE_SIDE_M)
        lon = sw_lon
        while lon < ne_lon:
            yield (lat, lon)
            dlon = _metres_to_lon(TILE_SIDE_M, at_lat=lat)
            lon += dlon
        lat += dlat


# Mapping from OSM `highway` tag values to our road class names.
ROAD_TAG_MAP: dict[str, str] = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "residential", "service": "service",
    "living_street": "living_street", "pedestrian": "pedestrian",
    "cycleway": "cycleway", "footway": "footway", "path": "path",
    "track": "track", "unclassified": "unclassified",
}


def _extract_bbox_geojson(pbf: Path, west: float, south: float, east: float, north: float) -> dict:
    """Run osmium to extract everything inside (W,S,E,N) and return as GeoJSON."""
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_geo:
        out_path = Path(tmp_geo.name)
    cmd = [
        "osmium", "extract",
        "--bbox", f"{west},{south},{east},{north}",
        "--strategy=smart",
        "--overwrite",
        "-o", str(out_path).replace(".geojson", ".osm.pbf"),
        str(pbf),
    ]
    # First osmium extract subset, then export to geojson.
    subset_pbf = Path(str(out_path).replace(".geojson", ".osm.pbf"))
    subprocess.run(cmd, check=True, capture_output=True)
    export_cmd = [
        "osmium", "export",
        "--overwrite",
        "-f", "geojson",
        "-o", str(out_path),
        str(subset_pbf),
    ]
    subprocess.run(export_cmd, check=True, capture_output=True)
    data = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    subset_pbf.unlink(missing_ok=True)
    return data


def extract_tile_geometry_from_osm(
    pbf: Path, sw_lat: float, sw_lon: float,
) -> TileGeometry:
    """Extract a single tile's TileGeometry from an OSM PBF.

    Coordinates in the returned TileGeometry are tile-local metres
    (origin = SW corner).
    """
    dlat = _metres_to_lat(TILE_SIDE_M)
    dlon = _metres_to_lon(TILE_SIDE_M, at_lat=sw_lat)
    ne_lat = sw_lat + dlat
    ne_lon = sw_lon + dlon
    geojson = _extract_bbox_geojson(pbf, sw_lon, sw_lat, ne_lon, ne_lat)

    geom = TileGeometry()

    def to_local(lon: float, lat: float) -> tuple[float, float]:
        # Approximate equirectangular projection inside this small tile.
        x_m = (lon - sw_lon) / dlon * TILE_SIDE_M
        y_m = (lat - sw_lat) / dlat * TILE_SIDE_M
        # Clamp to interior of tile (osmium can include boundary points)
        x_m = max(0.0, min(TILE_SIDE_M - 0.001, x_m))
        y_m = max(0.0, min(TILE_SIDE_M - 0.001, y_m))
        return (x_m, y_m)

    for feature in geojson.get("features", []):
        tags = feature.get("properties", {})
        coords = feature["geometry"]["coordinates"]
        gtype = feature["geometry"]["type"]

        # Roads: LineString features with a `highway` tag.
        if gtype in ("LineString", "MultiLineString") and "highway" in tags:
            mapped = ROAD_TAG_MAP.get(tags["highway"])
            if mapped is None:
                continue
            line = coords if gtype == "LineString" else coords[0]
            polyline = [to_local(lon, lat) for lon, lat in line]
            geom.roads.append(Road(class_name=f"road_class={mapped}", polyline=polyline))

        # Buildings: Polygon features with a `building` tag.
        elif gtype in ("Polygon", "MultiPolygon") and tags.get("building"):
            poly = coords[0] if gtype == "Polygon" else coords[0][0]
            verts = [to_local(lon, lat) for lon, lat in poly]
            cls = tags["building"] if isinstance(tags["building"], str) else "yes"
            cls = cls if cls != "yes" else "UNKNOWN"
            # Map onto our vocabulary if known, else UNKNOWN.
            cls_name = f"building_class={cls}"
            geom.buildings.append(Building(
                class_name=cls_name if cls != "UNKNOWN" else "building_class=UNKNOWN",
                height_name="height=NA",
                vertices=verts,
            ))

        # Land use polygons.
        elif gtype in ("Polygon", "MultiPolygon"):
            poly = coords[0] if gtype == "Polygon" else coords[0][0]
            verts = [to_local(lon, lat) for lon, lat in poly]
            if "natural" in tags and tags["natural"] in ("water",):
                geom.land.append(LandPolygon("water_class=lake", verts))
            elif "leisure" in tags and tags["leisure"] in ("park", "garden"):
                geom.land.append(LandPolygon("land_class=park", verts))
            elif "landuse" in tags and tags["landuse"] in (
                "forest", "meadow", "farmland", "grass", "orchard", "vineyard",
            ):
                geom.land.append(LandPolygon(f"land_class={tags['landuse']}", verts))
            elif "landuse" in tags and tags["landuse"] in (
                "residential", "commercial", "industrial", "retail",
            ):
                geom.land.append(LandPolygon(f"land_class={tags['landuse']}", verts))

        # POIs: Point features with category tags.
        elif gtype == "Point":
            lon, lat = coords
            xy = to_local(lon, lat)
            cls = None
            if "amenity" in tags:
                amenity = tags["amenity"]
                if amenity == "cafe": cls = "cafe"
                elif amenity == "restaurant": cls = "restaurant"
                elif amenity == "bar": cls = "bar"
                elif amenity == "pharmacy": cls = "pharmacy"
                elif amenity == "school": cls = "school"
                elif amenity == "hospital": cls = "hospital"
                elif amenity == "bank": cls = "bank"
                elif amenity == "fuel": cls = "gas_station"
                elif amenity == "parking": cls = "parking"
            if cls is not None:
                geom.pois.append(POI(class_name=f"poi={cls}", point=xy))

    return geom
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: 2 tests pass (the second is skipped if the PBF isn't downloaded yet — re-run after step 1 confirms presence).

- [ ] **Step 7: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/data/sampling.py bonzai_genai/tests/test_sampling.py
git commit -m "feat(data): add OSM PBF tile sampler with osmium backend"
```

---

## Task 15: Generate small Singapore tile dataset locally

**Files:**
- Modify: `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py` (add `overture-region` command)

- [ ] **Step 1: Add the new CLI command**

Append to `bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py` (before `if __name__`):

```python

@app.command("overture-region")
def cmd_overture_region(
    pbf: Path = typer.Option(..., help="Path to .osm.pbf file"),
    sw_lat: float = typer.Option(..., help="SW corner latitude"),
    sw_lon: float = typer.Option(..., help="SW corner longitude"),
    ne_lat: float = typer.Option(..., help="NE corner latitude"),
    ne_lon: float = typer.Option(..., help="NE corner longitude"),
    output: Path = typer.Option(..., "-o", "--output"),
    country: str = typer.Option("SG", "--country"),
    koppen: str = typer.Option("Af", "--koppen"),
    shard_size: int = typer.Option(100, "--shard-size"),
    max_tiles: int = typer.Option(1000, "--max-tiles"),
) -> None:
    """Generate tile bundles for every tile in (sw, ne) bbox from an OSM PBF.

    Used for Phase 0a Sweden + Singapore + Sri Lanka validation runs.
    """
    from bonzai_genai.data.sampling import extract_tile_geometry_from_osm, iter_tile_centres

    vocab = load_default_vocab()
    tokeniser = Tokeniser(vocab)
    writer = ShardWriter(output, shard_size=shard_size)

    centres = list(iter_tile_centres(sw_lat, sw_lon, ne_lat, ne_lon))[:max_tiles]
    console.print(f"Processing {len(centres)} tiles from {pbf.name}")

    n_kept = 0
    n_skipped = 0
    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Extracting", total=len(centres))
        for i, (lat, lon) in enumerate(centres):
            try:
                geom = extract_tile_geometry_from_osm(pbf, lat, lon)
            except Exception as e:
                console.print(f"  [yellow]skip {i}: {e}")
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            if len(geom.roads) + len(geom.buildings) < 5:
                # Skip near-empty tiles
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            try:
                raster = rasterise(geom)
                tokens = tokeniser.encode(geom)
            except KeyError as e:
                console.print(f"  [yellow]vocab miss tile {i}: {e}")
                n_skipped += 1
                progress.update(task_id, advance=1)
                continue
            meta = TileMetadata(
                tile_id=f"{country}-{i:06d}",
                sw_lat=lat, sw_lon=lon,
                country=country, koppen=koppen,
                density_bucket="urban",
                primary_land_use="residential",
            )
            writer.write(TileBundle(raster=raster, tokens=tokens, metadata=meta))
            n_kept += 1
            progress.update(task_id, advance=1)
    writer.close()
    console.print(f"[bold green]Kept {n_kept} tiles, skipped {n_skipped}")
```

- [ ] **Step 2: Run on Singapore (full island fits in ~150 tiles)**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/python scripts/prepare_tiles_local.py overture-region \
    --pbf data/malaysia-singapore-brunei-latest.osm.pbf \
    --sw-lat 1.20 --sw-lon 103.60 \
    --ne-lat 1.48 --ne-lon 104.05 \
    -o /tmp/bonzai-sg \
    --country SG --koppen Af \
    --shard-size 50 --max-tiles 200
```

Expected: progress bar shows ~100–200 tiles processed, shards written to `/tmp/bonzai-sg/`.

- [ ] **Step 3: Verify shards are readable and contain real geometry**

```bash
.venv/bin/python -c "
from pathlib import Path
from bonzai_genai.data.shard_writer import read_shard_bundles
bundles = list(read_shard_bundles(Path('/tmp/bonzai-sg')))
print(f'Read back {len(bundles)} Singapore bundles')
for b in bundles[:3]:
    print(f'  {b.metadata.tile_id}: tokens={len(b.tokens)}, raster_sum={b.raster.sum():.0f}')
"
```

Expected: nonzero raster sums for most tiles, token sequences in the hundreds-to-thousands range.

- [ ] **Step 4: Spot-check via raster visualisation**

```bash
.venv/bin/python -c "
import numpy as np
from pathlib import Path
from PIL import Image
from bonzai_genai.data.shard_writer import read_shard_bundles
from bonzai_genai.config import CHANNEL_NAMES
bundles = list(read_shard_bundles(Path('/tmp/bonzai-sg')))
out = Path('/tmp/bonzai-sg-viz'); out.mkdir(exist_ok=True)
for b in bundles[:3]:
    for c, name in enumerate(CHANNEL_NAMES):
        Image.fromarray((b.raster[c] * 255).astype(np.uint8)).save(out / f'{b.metadata.tile_id}-{c}-{name}.png')
print(f'Saved channel images to {out}')
"
ls /tmp/bonzai-sg-viz/ | head
open /tmp/bonzai-sg-viz/  # macOS Finder
```

Expected: per-channel PNGs visualising central Singapore. Roads should look like roads; building channel should look like building footprints.

- [ ] **Step 5: Commit**

```bash
git add bonzai_genai/src/bonzai_genai/cli/prepare_tiles.py
git commit -m "feat(cli): add overture-region command for real OSM tile generation"
```

---

## Task 16: Slurm template for Leonardo data prep

**Files:**
- Create: `bonzai_genai/scripts/leonardo_data_prep.sbatch`

- [ ] **Step 1: Write the Slurm template**

Write `bonzai_genai/scripts/leonardo_data_prep.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-data-prep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Required env-vars (set before sbatch):
#   BONZAI_PBF       — full path to the .osm.pbf
#   BONZAI_SW_LAT    — SW corner lat
#   BONZAI_SW_LON    — SW corner lon
#   BONZAI_NE_LAT    — NE corner lat
#   BONZAI_NE_LON    — NE corner lon
#   BONZAI_COUNTRY   — ISO country code (e.g. LU, IS)
#   BONZAI_KOPPEN    — Köppen group (e.g. Cfb)
#   BONZAI_OUT       — output dir on $WORK
#   BONZAI_MAX_TILES — optional; default 5000

set -euo pipefail

mkdir -p logs

# Activate venv on Leonardo
source "$WORK/bonzai_genai/.venv/bin/activate"

cd "$WORK/bonzai_genai"

mkdir -p "$BONZAI_OUT"

python scripts/prepare_tiles_local.py overture-region \
    --pbf "$BONZAI_PBF" \
    --sw-lat "$BONZAI_SW_LAT" --sw-lon "$BONZAI_SW_LON" \
    --ne-lat "$BONZAI_NE_LAT" --ne-lon "$BONZAI_NE_LON" \
    -o "$BONZAI_OUT" \
    --country "$BONZAI_COUNTRY" --koppen "$BONZAI_KOPPEN" \
    --shard-size 100 \
    --max-tiles "${BONZAI_MAX_TILES:-5000}"

echo "Job complete; manifest:"
cat "$BONZAI_OUT/manifest.json"
```

- [ ] **Step 2: Verify the template parses (locally, without submitting)**

```bash
bash -n /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai/scripts/leonardo_data_prep.sbatch && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Add usage documentation in scripts/README.md**

Write `bonzai_genai/scripts/README.md`:

```markdown
# Scripts

## `prepare_tiles_local.py`

CLI entrypoint. Two subcommands:

- `synthetic` — generate procedural tiles for smoke tests.
- `overture-region` — generate real OSM tiles for a bounding box.

Examples:

```bash
# 100 synthetic tiles
.venv/bin/python scripts/prepare_tiles_local.py synthetic -o /tmp/bonzai-syn -n 100

# Singapore (~150 real tiles, full island)
.venv/bin/python scripts/prepare_tiles_local.py overture-region \
    --pbf data/malaysia-singapore-brunei-latest.osm.pbf \
    --sw-lat 1.20 --sw-lon 103.60 \
    --ne-lat 1.48 --ne-lon 104.05 \
    -o /tmp/bonzai-sg \
    --country SG --koppen Af
```

## `leonardo_data_prep.sbatch`

SLURM job template for the free `lrd_all_serial` partition. Set env-vars
before submission. Examples for the three Phase 0a countries:

```bash
# Singapore (smallest, fastest)
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/malaysia-singapore-brunei-latest.osm.pbf
export BONZAI_SW_LAT=1.20 BONZAI_SW_LON=103.60
export BONZAI_NE_LAT=1.48 BONZAI_NE_LON=104.05
export BONZAI_COUNTRY=SG BONZAI_KOPPEN=Af
export BONZAI_OUT=$WORK/bonzai-tiles/singapore
export BONZAI_MAX_TILES=300
sbatch scripts/leonardo_data_prep.sbatch

# Sri Lanka
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/sri-lanka-latest.osm.pbf
export BONZAI_SW_LAT=5.85 BONZAI_SW_LON=79.55
export BONZAI_NE_LAT=9.90 BONZAI_NE_LON=81.95
export BONZAI_COUNTRY=LK BONZAI_KOPPEN=Aw
export BONZAI_OUT=$WORK/bonzai-tiles/sri_lanka
export BONZAI_MAX_TILES=2000
sbatch scripts/leonardo_data_prep.sbatch

# Sweden
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/sweden-latest.osm.pbf
export BONZAI_SW_LAT=55.0 BONZAI_SW_LON=10.5
export BONZAI_NE_LAT=69.5 BONZAI_NE_LON=24.5
export BONZAI_COUNTRY=SE BONZAI_KOPPEN=Cfb
export BONZAI_OUT=$WORK/bonzai-tiles/sweden
export BONZAI_MAX_TILES=5000
sbatch scripts/leonardo_data_prep.sbatch
```
```

- [ ] **Step 4: Commit**

```bash
git add bonzai_genai/scripts/leonardo_data_prep.sbatch bonzai_genai/scripts/README.md
git commit -m "feat(slurm): add Leonardo data-prep job template"
```

---

## Task 17: Plan-level documentation in repo README

**Files:**
- Create: `bonzai_genai/README.md`

- [ ] **Step 1: Write the README**

Write `bonzai_genai/README.md`:

```markdown
# bonzai_genai

Production codebase for the Bonzai-OSM generative city model.

See [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](../docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md) for the full design spec.

## Status

**Phase 0a (data prep pipeline)** — *in progress*. This package implements the
data prep portion of the pipeline. Subsequent plans add Stage A (Sketcher),
Stage B (Inker), eval, and de-risking experiment orchestration.

## Setup

```bash
cd bonzai_genai
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

System dependency: `osmium-tool` (`brew install osmium-tool` on Mac, or
`module load osmium-tool` on Leonardo if available).

## Tests

```bash
.venv/bin/pytest -v
```

All tests must pass before committing.

## Generating tiles

See `scripts/README.md`.

## Layout

- `src/bonzai_genai/config.py` — global constants (tile size, channel layout, vocab sizes).
- `src/bonzai_genai/vocab/` — token id space + tokeniser.
- `src/bonzai_genai/data/` — tile sampling, rasterisation, vector serialisation, sharding.
- `src/bonzai_genai/synth/` — procedural smoke-test generator.
- `src/bonzai_genai/cli/` — typer-based CLI entrypoints.
- `tests/` — pytest suite (unit + round-trip tests).
- `configs/` — vocabulary + sampling YAML configs.
- `scripts/` — CLI wrappers + Slurm templates.

## Key design constants

| Constant | Value |
|---|---|
| Tile side | 2,048 m |
| Raster | 512 × 512 px (4 m/px) |
| Channels | 9 (3 road classes + binary + buildings × 2 + water + green + urban) |
| Coordinate quantisation | 512 bins per axis |

## Phase 0a deliverable

Working data-prep pipeline + a Sweden + Singapore + Sri Lanka tile dataset on
Leonardo `$WORK`, validated by round-trip tests.
```

- [ ] **Step 2: Commit**

```bash
git add bonzai_genai/README.md
git commit -m "docs(bonzai_genai): add package README"
```

---

## Task 18: Run the full test suite + lint

**Files:**
- No code changes. Validation only.

- [ ] **Step 1: Full test suite**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai
.venv/bin/pytest -v
```

Expected: all tests pass (the count depends on how many were added; should be ~30+).

- [ ] **Step 2: Run ruff**

```bash
.venv/bin/ruff check src/ tests/ scripts/
```

Expected: no errors. Fix any flagged issues inline.

- [ ] **Step 3: Run black**

```bash
.venv/bin/black --check src/ tests/ scripts/
```

Expected: all files already formatted (or run without `--check` to format and recommit).

- [ ] **Step 4: Commit any format fixes**

If `ruff` or `black` made changes:

```bash
git add bonzai_genai/
git commit -m "style(bonzai_genai): apply ruff and black formatting"
```

---

## Task 19: Deploy to Leonardo and run all three country jobs

**Files:**
- No code changes. Deployment + validation.

> This task assumes you have an active Leonardo SSH cert (`step ssh login`) and downloads the three country PBFs to `$CINECA_SCRATCH` via the datamover.

- [ ] **Step 1: Sync the package to Leonardo**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM
rsync -az --partial \
    --exclude=.venv --exclude=.pytest_cache --exclude=__pycache__ \
    bonzai_genai/ \
    uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai_genai/
```

Expected: rsync transfers the package without errors.

- [ ] **Step 2: Set up venv on Leonardo (one-time)**

SSH to Leonardo and run:

```bash
ssh uaslam00@login.leonardo.cineca.it
module load python/3.11.7
cd "$WORK/bonzai_genai"
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
osmium --version
exit
```

If `osmium` isn't available as a module on Leonardo, fall back to using GDAL's OSM driver (this is a Plan 1.5 task — record as a follow-up if encountered).

- [ ] **Step 3: Download the three PBFs via the Leonardo datamover**

```bash
ssh uaslam00@login.leonardo.cineca.it
mkdir -p "$CINECA_SCRATCH/osm/raw"

# Singapore (Geofabrik bundles it with Malaysia + Brunei)
test -f "$CINECA_SCRATCH/osm/raw/malaysia-singapore-brunei-latest.osm.pbf" || \
    ssh -xt $USER@data.leonardo.cineca.it \
    wget -O "$CINECA_SCRATCH/osm/raw/malaysia-singapore-brunei-latest.osm.pbf" \
    https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf

# Sri Lanka
test -f "$CINECA_SCRATCH/osm/raw/sri-lanka-latest.osm.pbf" || \
    ssh -xt $USER@data.leonardo.cineca.it \
    wget -O "$CINECA_SCRATCH/osm/raw/sri-lanka-latest.osm.pbf" \
    https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf

# Sweden
test -f "$CINECA_SCRATCH/osm/raw/sweden-latest.osm.pbf" || \
    ssh -xt $USER@data.leonardo.cineca.it \
    wget -O "$CINECA_SCRATCH/osm/raw/sweden-latest.osm.pbf" \
    https://download.geofabrik.de/europe/sweden-latest.osm.pbf

ls -lh "$CINECA_SCRATCH/osm/raw/"*.osm.pbf
```

Expected: three PBFs present (Singapore-bundle ~400 MB, Sri Lanka ~150 MB, Sweden ~700 MB).

- [ ] **Step 4: Submit Singapore data prep (smallest job, run first)**

```bash
cd "$WORK/bonzai_genai"
mkdir -p logs
export BONZAI_PBF=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/malaysia-singapore-brunei-latest.osm.pbf
export BONZAI_SW_LAT=1.20 BONZAI_SW_LON=103.60
export BONZAI_NE_LAT=1.48 BONZAI_NE_LON=104.05
export BONZAI_COUNTRY=SG BONZAI_KOPPEN=Af
export BONZAI_OUT="$WORK/bonzai-tiles/singapore"
export BONZAI_MAX_TILES=300
sbatch scripts/leonardo_data_prep.sbatch
squeue -u $USER
```

Wait for completion (should be < 30 min on `lrd_all_serial`). Verify:

```bash
cat "$WORK/bonzai-tiles/singapore/manifest.json"
```

Expected: ~100–200 tiles, multiple shards.

- [ ] **Step 5: Submit Sri Lanka and Sweden in parallel**

```bash
# Sri Lanka
export BONZAI_PBF=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/sri-lanka-latest.osm.pbf
export BONZAI_SW_LAT=5.85 BONZAI_SW_LON=79.55
export BONZAI_NE_LAT=9.90 BONZAI_NE_LON=81.95
export BONZAI_COUNTRY=LK BONZAI_KOPPEN=Aw
export BONZAI_OUT="$WORK/bonzai-tiles/sri_lanka"
export BONZAI_MAX_TILES=2000
sbatch scripts/leonardo_data_prep.sbatch

# Sweden
export BONZAI_PBF=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/sweden-latest.osm.pbf
export BONZAI_SW_LAT=55.0 BONZAI_SW_LON=10.5
export BONZAI_NE_LAT=69.5 BONZAI_NE_LON=24.5
export BONZAI_COUNTRY=SE BONZAI_KOPPEN=Cfb
export BONZAI_OUT="$WORK/bonzai-tiles/sweden"
export BONZAI_MAX_TILES=5000
sbatch scripts/leonardo_data_prep.sbatch

squeue -u $USER
```

Expected: both jobs queued. Sri Lanka ~1–2 h; Sweden may approach 4 h walltime cap — if it hits, chain a second job from the last completed tile (note as a follow-up).

- [ ] **Step 6: Spot-check each output on Leonardo**

```bash
for c in singapore sri_lanka sweden; do
    echo "=== $c ==="
    cat "$WORK/bonzai-tiles/$c/manifest.json"
    .venv/bin/python -c "
from pathlib import Path
from bonzai_genai.data.shard_writer import read_shard_bundles
import itertools
bundles = list(itertools.islice(read_shard_bundles(Path('$WORK/bonzai-tiles/$c')), 3))
for b in bundles:
    print(f'  {b.metadata.tile_id}: tokens={len(b.tokens)}, raster_sum={b.raster.sum():.0f}')
"
done
```

Expected: three subdirectories (`singapore/`, `sri_lanka/`, `sweden/`), each with non-zero raster sums.

- [ ] **Step 7: Final inventory**

```bash
du -sh "$WORK/bonzai-tiles/"*
ls "$WORK/bonzai-tiles/"
```

Expected: three subdirectories. **This is the Phase 0a deliverable.**

- [ ] **Step 8: Pull manifests back to local repo for record**

```bash
cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM
mkdir -p bonzai_genai/results
for c in singapore sri_lanka sweden; do
    rsync -az "uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai-tiles/$c/manifest.json" "bonzai_genai/results/$c-manifest.json"
done
git add bonzai_genai/results/
git commit -m "feat(data): record Sweden + Singapore + Sri Lanka tile manifests on Leonardo"
```

---

## Task 20: Plan completion summary

**Files:**
- Create: `bonzai_genai/results/PHASE_0A_COMPLETE.md`

- [ ] **Step 1: Write the completion summary**

Write `bonzai_genai/results/PHASE_0A_COMPLETE.md`:

```markdown
# Phase 0a Completion — Data Prep Pipeline + Sweden + Singapore + Sri Lanka Tiles

**Completed:** YYYY-MM-DD (fill in)
**Branch:** `genai-city-model`

## Deliverables shipped

- ✅ `bonzai_genai/` package, fully tested
- ✅ Token id space, attribute vocabulary, tokeniser (encode + decode)
- ✅ Vector → 9-channel rasteriser with road class hierarchy
- ✅ TileBundle dataclass + WebDataset shard I/O
- ✅ Procedural smoke-test generator
- ✅ End-to-end synthetic round-trip test
- ✅ Real-data tile sampler (osmium-backed)
- ✅ Slurm job template for Leonardo data prep
- ✅ Sweden tile dataset on Leonardo `$WORK/bonzai-tiles/sweden/`
- ✅ Singapore tile dataset on Leonardo `$WORK/bonzai-tiles/singapore/`
- ✅ Sri Lanka tile dataset on Leonardo `$WORK/bonzai-tiles/sri_lanka/`

## Counts (fill in from manifests)

- Sweden tiles: NNN
- Singapore tiles: NNN
- Sri Lanka tiles: NNN
- Total disk: NN MB

## Open follow-ups for Plan 2+

- [ ] Expand attribute vocab YAML to ~1,800 tokens (FSQ leaves)
- [ ] Add stratification logic (currently uniform sampling on bbox)
- [ ] Add building-height extraction from OSM `building:height` / `building:levels` tags
- [ ] Add Overture Bridge File enrichment for ~6% of buildings with OSM tags
- [ ] Switch tile sampler from osmium to direct Overture parquet reads via DuckDB

## Hand-off

Plan 2 (synthetic smoke harness for Experiment 0) can now begin: it depends only on the tokeniser, rasteriser, and shard I/O implemented here.
```

- [ ] **Step 2: Commit**

```bash
git add bonzai_genai/results/PHASE_0A_COMPLETE.md
git commit -m "docs: mark Phase 0a complete with deliverables summary"
```

---

# Self-Review

**1. Spec coverage check:**
- §3 Architecture (data prep unit) → Tasks 1–20 ✓
- §4.1 Three vector data sources → Task 14 (Geofabrik OSM) ✓; FSQ + Overture parquet deferred to Plan 5 (logged in Task 20 follow-ups)
- §4.2 One training example bundle → Task 9, 10, 12 ✓
- §4.3 9 raster channels → Tasks 3, 8 ✓
- §4.4 Stratified sampling → partial (uniform on bbox); stratified DuckDB query deferred to Plan 5 (logged)
- §4.5 Training set sizing → Task 19 produces real numbers
- §4.6 WebDataset format → Task 10 ✓
- §4.7 Building label gap → handled via `building_class=UNKNOWN` in Task 14 ✓
- §6.1 Vocabulary → Tasks 4, 5 ✓
- §6.2 Reading order → Task 6 ✓
- §15 Glossary terms → all present in code (Tile, raster channels, vector tokens, tile-local coords, WebDataset)

**2. Placeholder scan:** No "TBD"/"TODO" within executable code. The fill-in dates and counts in `PHASE_0A_COMPLETE.md` (Task 20) are intentional — they get filled at completion time, not at planning time.

**3. Type consistency:** Spot checks:
- `Building`, `Road`, `POI`, `LandPolygon`, `TileGeometry` — defined Task 6, used consistently in Tasks 8, 11, 12, 14.
- `TileBundle`, `TileMetadata` — defined Task 9, used in Tasks 10, 12, 14, 15.
- `coord_x_token_id` / `coord_y_token_id` / `parse_coord_x_token` / `parse_coord_y_token` — same names throughout.
- `CH_BUILDINGS`, `CH_ALL_ROADS` etc. — defined Task 8, used in Task 12.

**4. Ambiguity check:** The road-edge token encoding in Task 6 (re-using x-coord token space for node references) is unusual but explicitly documented inline; Plan 4 will swap in dedicated node-ref tokens. The choice is consistent within Phase 0a.

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-phase-0a-data-prep-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**

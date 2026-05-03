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
python3.12 -m venv .venv     # or python3.11 if you have it
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
- `configs/` — vocabulary YAML.
- `scripts/` — CLI wrappers + Slurm templates.
- `conftest.py` — adds `src/` to `sys.path` (workaround for an editable-install issue with pip 26).

## Key design constants

| Constant | Value |
|---|---|
| Tile side | 2,048 m |
| Raster | 512 × 512 px (4 m/px) |
| Channels | 9 (3 road classes + binary roads + buildings × 2 + water + green + urban) |
| Coordinate quantisation | 512 bins per axis |
| Vocab (v1) | ~1,329 tokens (15 special + 1024 coord + 290 attribute) |

## Phase 0a deliverable

Working data-prep pipeline + a Sweden + Singapore + Sri Lanka tile dataset on
Leonardo `$WORK`, validated by round-trip tests.

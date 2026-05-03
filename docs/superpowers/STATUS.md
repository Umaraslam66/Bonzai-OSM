# Bonzai-OSM — Live Status

> **Last updated:** 2026-05-03 (this session in progress, paused after Plan Task 13)
>
> If you are a new agent starting a session: read this file first, then [`PROJECT.md`](../../PROJECT.md), then the [spec](specs/2026-05-03-genai-city-infrastructure-design.md), then the [active plan](plans/2026-05-03-phase-0a-data-prep-pipeline.md). Pick up from the next unchecked task.

---

## Current state

| | |
|---|---|
| Branch | `genai-city-model` (long-lived dev branch — **never merge to main**) |
| Active phase | **Phase 0a — Data Prep Pipeline + 3-country tile dataset** |
| Active plan | [`plans/2026-05-03-phase-0a-data-prep-pipeline.md`](plans/2026-05-03-phase-0a-data-prep-pipeline.md) (20 tasks) |
| Last completed plan task | **Plan Task 13: Smoke test CLI** (commit `64036e2`) |
| Next action | **Plan Task 14: Real-data tile sampler (Overture / Geofabrik)** |
| Tests passing | **48 / 48** |
| GPU-h burned this session | 0 |
| GPU-h burned cumulative on project | 14 (from prior sessions, pre-v1) |

## Recent commit history (most recent first)

```
64036e2 feat(cli): add prepare_tiles synthetic command
484d5d3 test(integration): add end-to-end synthetic round-trip test
f45e379 feat(synth): add deterministic procedural tile generator
2e08d1f feat(data): add WebDataset-format shard writer and reader
07068ba feat(data): add TileBundle dataclass with raster/tokens/metadata
454e358 feat(data): add 9-channel rasteriser with road class hierarchy
5bab527 feat(vocab): add geometry tokeniser with encode/decode + round-trip tests
4143124 docs: mark Plan Tasks 1-5 complete; update STATUS.md handoff
21501a2 feat(vocab): add attribute vocabulary loader
a4f4ca7 feat(vocab): add special and coordinate token id space
90b5e4f feat(config): add global tile and channel constants
e52373f feat(bonzai_genai): scaffold package structure
c83fed3 chore(genai-city-model): branch off, clean obsolete files...
cef2396 Add Phase 0a implementation plan for data prep pipeline
5d37e84 Add generative city model design spec and brainstorm log
```

## Plan progress (Phase 0a — 20 tasks)

- [x] Task 1: Project scaffolding (pyproject.toml + src layout)
- [x] Task 2: Install dev dependencies + verify pytest works
- [x] Task 3: Global config module
- [x] Task 4: Token type definitions
- [x] Task 5: Attribute vocabulary
- [x] Task 6: Tokeniser — encode primitives to token sequence
- [x] Task 7: Tokeniser round-trip property test
- [x] Task 8: Rasteriser — line and polygon painting
- [x] Task 9: TileBundle dataclass + serialisation
- [x] Task 10: WebDataset shard writer + reader
- [x] Task 11: Synthetic procedural city generator
- [x] Task 12: End-to-end synthetic round-trip test
- [x] Task 13: Smoke test CLI — generate 100 synthetic shards locally
- [ ] **Task 14: Real-data tile sampler (Overture / Geofabrik)  ← NEXT**
- [ ] Task 15: Generate small Singapore tile dataset locally
- [ ] Task 16: Slurm template for Leonardo data prep
- [ ] Task 17: Plan-level documentation in repo README
- [ ] Task 18: Run the full test suite + lint
- [ ] Task 19: Deploy to Leonardo and run all three country jobs
- [ ] Task 20: Plan completion summary

## What was built so far in `bonzai_genai/`

```
bonzai_genai/
├── pyproject.toml                     ✅ Python 3.11+, deps + dev deps
├── .gitignore                          ✅ ignores top-level data/, shards/, .venv/, etc.
├── conftest.py                         ✅ adds src/ to sys.path (editable-install workaround)
├── .venv/                              ✅ Python 3.12 venv (pip 26 / setuptools-editable .pth files
│                                          aren't honoured during pytest collection or script
│                                          execution; conftest + sys.path-prepend in scripts works
│                                          around it. Cause unknown — likely setuptools/pip bug.)
├── src/bonzai_genai/
│   ├── __init__.py                    ✅
│   ├── config.py                      ✅ TILE_SIDE_M=2048, RASTER_PX=512, METRES_PER_PX=4, COORD_BINS=512, NUM_CHANNELS=9
│   ├── vocab/
│   │   ├── tokens.py                  ✅ SpecialToken IntEnum (15), coord_x/y_token_id, parse_*
│   │   ├── attributes.py              ✅ AttributeVocab + load_default_vocab()
│   │   └── tokeniser.py               ✅ Tokeniser.encode/decode + Building/Road/POI/LandPolygon/TileGeometry
│   ├── data/
│   │   ├── rasteriser.py              ✅ rasterise(geom) → 9-channel float32 (PIL + scipy.gaussian_filter)
│   │   ├── tile_bundle.py             ✅ TileBundle + TileMetadata (raster.npy/tokens.json/metadata.json)
│   │   └── shard_writer.py            ✅ ShardWriter + read_shard_bundles() (WebDataset tar shards)
│   ├── synth/
│   │   └── procedural.py              ✅ generate_synthetic_tile(seed) - 8x8 grid roads + jittered rectangles + POIs
│   └── cli/
│       └── prepare_tiles.py           ✅ Typer CLI with `synthetic` subcommand (default for now; multi when Task 15 lands)
├── configs/
│   └── attributes_v1.yaml             ✅ 290 attribute tokens (road/building/land/water/poi families + heights)
├── scripts/
│   └── prepare_tiles_local.py         ✅ wrapper with sys.path fallback
└── tests/                              ✅ 48 tests, all passing
    ├── test_config.py                  (5)
    ├── test_tokens.py                  (7)
    ├── test_attributes.py              (7)
    ├── test_tokeniser.py               (8 — 6 unit + 2 round-trip)
    ├── test_rasteriser.py              (8)
    ├── test_tile_bundle.py             (5)
    ├── test_shard_writer.py            (4)
    ├── test_synth.py                   (3)
    └── test_round_trip.py              (1 end-to-end integration)
```

**Total vocab: 15 special + 1024 coord + 290 attribute = 1,329 tokens.** Will grow to ~1,800 in Plan 5 when full FSQ leaf categories are integrated.

## Smoke run verified

```bash
cd bonzai_genai
.venv/bin/python scripts/prepare_tiles_local.py -o /tmp/bonzai-syn -n 50 --shard-size 25
# → 50 tiles, 2 shards, manifest.json, ~50 KB each, 1,218-token sequences
```

Round-trip via `read_shard_bundles` returns 50 bundles with intact `(9, 512, 512)` rasters and 1,218-element token lists.

## Notes for the next agent / next session

### How to resume execution

1. `git status` on branch `genai-city-model` should be clean.
2. `cd bonzai_genai && .venv/bin/pytest -v` should report 48/48 passing.
3. Open the active plan and find **Task 14: Real-data tile sampler**. It introduces:
   - `osmium` system dependency (`brew install osmium-tool` on Mac).
   - Singapore-area Geofabrik bundle download (~400 MB).
   - `bonzai_genai/src/bonzai_genai/data/sampling.py` with `iter_tile_centres` + `extract_tile_geometry_from_osm`.
   - `bonzai_genai/tests/test_sampling.py` with one bbox test + one PBF-skipped real-data test.
4. Then Task 15 adds the `overture-region` CLI command and runs ~100 Singapore tiles locally.

### Key environmental facts

- **Python:** macOS dev machine has `python3.12`, not `python3.11`. The pyproject `requires-python = ">=3.11"` permits this.
- **Bash session loses cwd between tool calls.** Always `cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai && ...` for venv-based commands.
- **Editable-install gotcha:** `pip install -e .` writes a `.pth` file at `.venv/lib/python3.12/site-packages/__editable__.bonzai_genai-0.1.0.pth` containing the absolute path to `src/`, but Python's site processing isn't picking it up. Workarounds in place:
  - `bonzai_genai/conftest.py` inserts `src/` on `sys.path` for pytest.
  - `bonzai_genai/scripts/prepare_tiles_local.py` prepends `src/` to `sys.path` before importing.
  - Cause unknown — probably a setuptools 80.x / pip 26 interaction. Worth investigating in Plan 2 cleanup.

### TDD discipline (per-task pattern)

1. Write the failing test from the plan.
2. Run pytest, confirm `ModuleNotFoundError` or assertion failure.
3. Write the implementation from the plan.
4. Run pytest, confirm green.
5. Commit with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
6. Mark step checkboxes `[x]` in the plan file.

### Cross-session continuity rules

- Update **this STATUS.md file** every 3–5 plan tasks with: last completed task, latest commit hash, next action, any new follow-ups discovered.
- Add memory entries (`~/.claude/projects/-Users-umaraslam-Documents-dynamo-Bonzai-OSM/memory/`) for: any user feedback that should persist across sessions, any project decisions, any external resources (tracker URLs, dashboard links).
- Commit cadence: at least once per plan task. `git log --oneline` should tell the story.
- **No subagents.** PI explicitly requested inline execution by the main session.

### Open follow-ups discovered during execution

1. Attribute vocab YAML ships ~290 tokens for v1; expansion to full FSQ leaves (~1,800 total) deferred to Plan 5.
2. The `bonzai_genai/.gitignore` was initially too broad (`data/` matched `src/bonzai_genai/data/`). Fixed to `/data/` `/shards/` (root-anchored).
3. **Editable-install path issue** (see Key environmental facts). The conftest + sys.path workarounds work, but root cause should be diagnosed in a Plan 2 cleanup task.
4. Typer single-command behaviour: when only one `@app.command(...)` is registered, the subcommand name is optional and the bare options are taken as the default. Once Task 15 adds `overture-region` as a second command, the `synthetic` subcommand name becomes required at the CLI.

## Three countries for v1 de-risking (Phase 0a target)

| Country | Köppen | Geofabrik URL | Centroid bbox |
|---|---|---|---|
| Sweden | Cfb / Dfb | `https://download.geofabrik.de/europe/sweden-latest.osm.pbf` | `55.0,10.5 → 69.5,24.5` (full country) |
| Singapore | Af | `https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf` (extract) | `1.20,103.60 → 1.48,104.05` (full island) |
| Sri Lanka | Af / Aw | `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf` | `5.85,79.55 → 9.90,81.95` (full country) |

Singapore's Geofabrik bundle includes Malaysia + Brunei; we crop to Singapore's bbox during data prep.

## Phase 0a deliverable (recap)

A working `bonzai_genai/` Python package + tile shards on Leonardo `$WORK/bonzai-tiles/{sweden,singapore,sri_lanka}/`, validated via round-trip tests. **Zero GPU billing consumed.**

## Blockers / open questions

None. Awaiting Plan Task 14 execution.

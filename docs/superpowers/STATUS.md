# Bonzai-OSM — Live Status

> **Last updated:** 2026-05-03 (this session — Plan Tasks 1–18 complete; only Leonardo deployment + completion summary remain)
>
> If you are a new agent starting a session: read this file first, then [`PROJECT.md`](../../PROJECT.md), then the [spec](specs/2026-05-03-genai-city-infrastructure-design.md), then the [active plan](plans/2026-05-03-phase-0a-data-prep-pipeline.md). Pick up from the next unchecked task.

---

## Current state

| | |
|---|---|
| Branch | `genai-city-model` (long-lived dev branch — **never merge to main**) |
| Active phase | **Phase 0a — Data Prep Pipeline + 3-country tile dataset** |
| Active plan | [`plans/2026-05-03-phase-0a-data-prep-pipeline.md`](plans/2026-05-03-phase-0a-data-prep-pipeline.md) (20 tasks) |
| Last completed plan task | **Plan Task 18: Run the full test suite + lint** (commit `18416d4`) |
| Next action | **Plan Task 19: Deploy to Leonardo and run all three country jobs** *(requires user's active SSH cert)* |
| Tests passing | **50 / 50** |
| Ruff lint | **Clean** |
| GPU-h burned this session | 0 |
| GPU-h burned cumulative on project | 14 (from prior sessions, pre-v1) |

## Recent commit history (most recent first)

```
18416d4 feat: package README + ruff/black-clean + Slurm template (Plan Tasks 16-18)
daf4147 feat(cli): add overture-region command for real OSM tile generation
40b58e8 feat(data): add OSM PBF tile sampler (osmium-backed)
8890275 docs: mark Plan Tasks 6-13 complete; refresh STATUS.md
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

- [x] Task 1: Project scaffolding
- [x] Task 2: Install dev dependencies
- [x] Task 3: Global config module
- [x] Task 4: Token type definitions
- [x] Task 5: Attribute vocabulary
- [x] Task 6: Tokeniser encode/decode
- [x] Task 7: Tokeniser round-trip property test
- [x] Task 8: 9-channel rasteriser
- [x] Task 9: TileBundle dataclass + serialisation
- [x] Task 10: WebDataset shard writer + reader
- [x] Task 11: Synthetic procedural city generator
- [x] Task 12: End-to-end synthetic round-trip test
- [x] Task 13: Smoke test CLI (synthetic command)
- [x] Task 14: Real-data tile sampler (osmium)
- [x] Task 15: overture-region CLI + first Singapore tiles
- [x] Task 16: Slurm template for Leonardo
- [x] Task 17: Package README
- [x] Task 18: Full test suite + ruff
- [ ] **Task 19: Deploy to Leonardo and run all three country jobs ← NEXT (needs your SSH cert)**
- [ ] Task 20: Plan completion summary

## What was built (final tally for Phase 0a code)

```
bonzai_genai/
├── pyproject.toml                     ✅
├── README.md                          ✅
├── .gitignore                          ✅
├── conftest.py                         ✅ (sys.path workaround for editable-install)
├── .venv/                              ✅ Python 3.12, all deps installed
├── src/bonzai_genai/
│   ├── config.py                      ✅
│   ├── vocab/{tokens,attributes,tokeniser}.py    ✅
│   ├── data/{rasteriser,tile_bundle,shard_writer,sampling}.py    ✅
│   ├── synth/procedural.py            ✅
│   └── cli/prepare_tiles.py           ✅ (synthetic + overture-region commands)
├── configs/attributes_v1.yaml         ✅ (290 attributes, will expand to ~1,800 in Plan 5)
├── scripts/
│   ├── prepare_tiles_local.py         ✅
│   ├── leonardo_data_prep.sbatch      ✅
│   └── README.md                      ✅
├── data/                               ⏸ (gitignored; Singapore PBF here for local dev)
└── tests/                              ✅ 50/50 pass, ruff-clean
    ├── test_config.py (5)
    ├── test_tokens.py (7)
    ├── test_attributes.py (7)
    ├── test_tokeniser.py (8)
    ├── test_rasteriser.py (8)
    ├── test_tile_bundle.py (5)
    ├── test_shard_writer.py (4)
    ├── test_synth.py (3)
    ├── test_sampling.py (2 — including real Marina Bay extract)
    └── test_round_trip.py (1)
```

## Real-data smoke run (already done locally)

**Western Singapore, 50 tile centres attempted, 4 successfully encoded:**

```
SG-000018 (1.357,103.650): tokens=2129, raster_sum=68878
SG-000027 (1.375,103.650): tokens=2000, raster_sum=76728
SG-000036 (1.394,103.650): tokens=1862, raster_sum=25708
SG-000039 (1.394,103.705): tokens=7195, raster_sum=65149
```

The other 46 tiles overflowed the 512 road-node cap. **This is a known limit of the current encoding** — the tokeniser temporarily packs node references into the x-coord token namespace (capped at COORD_BINS=512). Plan 4 (Stage B) introduces dedicated node-ref tokens to lift this cap. For Phase 0a, the CLI handles overflow by `skip-and-continue`, which is correct behaviour.

**This also confirms Singapore as the "ultra-dense urban" extreme of our country triple** — every Singapore tile pushes the encoding limit.

## Notes for the next agent / next session

### How to resume execution (Plan Task 19)

This is the Leonardo deployment task. **Requires the PI's active Leonardo SSH cert** (`step ssh login 'aslamumar16@gmail.com' --provisioner cineca-hpc`).

Execution path:

1. From the Mac side: `rsync -az --partial --exclude=.venv --exclude=.pytest_cache --exclude=__pycache__ bonzai_genai/ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/bonzai_genai/`.
2. SSH to Leonardo, `cd "$WORK/bonzai_genai"`, set up venv (`module load python/3.11.7; python -m venv .venv; .venv/bin/pip install -e ".[dev]"`).
3. Download three PBFs via the datamover (`malaysia-singapore-brunei`, `sri-lanka`, `sweden`) — see plan Task 19 step 3 for exact `wget` commands.
4. Submit Singapore job (`sbatch scripts/leonardo_data_prep.sbatch` with env-vars set).
5. Wait, verify output, then submit Sri Lanka and Sweden jobs in parallel.
6. Pull manifests back to the repo. `git add bonzai_genai/results/ && git commit`.

**Expected skip rate:** Singapore will skip a lot of tiles (node-cap overflow). Sri Lanka and Sweden should keep most tiles.

**Then Plan Task 20:** write `bonzai_genai/results/PHASE_0A_COMPLETE.md` and commit. Phase 0a done.

### Key environmental facts (carried from earlier)

- **Python:** Mac has `python3.12`, not `python3.11`. Pyproject allows ≥3.11. Leonardo has `python/3.11.7` module.
- **Bash session loses cwd between calls.** Always `cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai && ...`.
- **Editable-install gotcha** (still unfixed): `pip install -e .` writes a `.pth` file at `.venv/lib/python3.12/site-packages/__editable__.bonzai_genai-0.1.0.pth` containing `src/`'s absolute path, but Python's site processing isn't picking it up. Workarounds in place:
  - `bonzai_genai/conftest.py` inserts `src/` on `sys.path` for pytest.
  - `bonzai_genai/scripts/prepare_tiles_local.py` prepends `src/` to `sys.path` before importing.
  - **On Leonardo this may also bite** — if it does, the same conftest + script-side workaround should still work.

### Open follow-ups discovered during execution

1. **Attribute vocab YAML ships ~290 tokens.** Expansion to full FSQ leaves (~1,800) deferred to Plan 5.
2. **Editable-install path issue** — root cause unknown; conftest + sys.path-prepend in scripts works around it.
3. **Singapore node-cap.** The current tokeniser caps road nodes per tile at COORD_BINS=512 because it re-uses x-coord token space for node references. Plan 4 (Stage B sampling decoder) introduces dedicated `ROAD_NODE_REF_*` tokens to lift this cap. For Phase 0a we skip-and-continue.
4. **`bonzai_genai/.gitignore`** initially over-broad (`data/` matched `src/bonzai_genai/data/`). Fixed to root-anchored `/data/` `/shards/`.
5. **Typer single-command behaviour:** when only one `@app.command(...)` is registered, the subcommand name is optional. Once `overture-region` was added (Task 15), `synthetic` became a required subcommand name at the CLI.

## Three countries for v1 de-risking (Phase 0a target)

| Country | Köppen | Geofabrik URL | Centroid bbox |
|---|---|---|---|
| Sweden | Cfb / Dfb | `https://download.geofabrik.de/europe/sweden-latest.osm.pbf` | `55.0,10.5 → 69.5,24.5` (full country) |
| Singapore | Af | `https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf` (extract) | `1.20,103.60 → 1.48,104.05` (full island) |
| Sri Lanka | Af / Aw | `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf` | `5.85,79.55 → 9.90,81.95` (full country) |

## Phase 0a deliverable (recap)

A working `bonzai_genai/` Python package + tile shards on Leonardo `$WORK/bonzai-tiles/{sweden,singapore,sri_lanka}/`, validated via round-trip tests. **Zero GPU billing consumed.** The package side is done; only the Leonardo run remains.

## Blockers / open questions

**One blocker for Task 19:** the PI must have an active Leonardo SSH cert (12 h validity from `step ssh login`). The next session begins by checking this; if expired, `step ssh login 'aslamumar16@gmail.com' --provisioner cineca-hpc` to refresh.

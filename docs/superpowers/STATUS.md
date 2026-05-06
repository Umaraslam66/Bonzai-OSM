# Bonzai-OSM — Live Status

> **Last updated:** 2026-05-06 (Plan 3a Sweden-only ran end-to-end on Leonardo; mixed result; Plan 3b is next).
>
> If you are a new agent starting a session: read this file first, then [`PROJECT.md`](../../PROJECT.md), then the [global design spec](specs/2026-05-03-genai-city-infrastructure-design.md), then the most recent hand-off doc — [`bonzai_genai/results/PLAN_3_REPORT.md`](../../bonzai_genai/results/PLAN_3_REPORT.md). Plan 3b (fresh multi-country VAE + constrained eval + country conditioning) is the next thing to draft.

---

## Current state

| | |
|---|---|
| Branch | `genai-city-model` (long-lived dev branch — **never merge to main**) |
| Active phase | **Phase 1 — De-risking experiments 1–4** |
| Last completed plan task | **Plan 3a Sweden-only complete** (commits `c0e674e` data_module repeat fix, `97c691c` KV-cached Inker sampler). 64 PNGs + 64 GeoJSONs eval'd; report at [`bonzai_genai/results/PLAN_3_REPORT.md`](../../bonzai_genai/results/PLAN_3_REPORT.md). |
| Plan 3a verdict | Architecture extracts gradient signal ✅ (Painter loss 52.4→0.26, Writer 6.25→0.0006) but end-to-end pipeline does **not** produce valid samples (0/64 GeoJSONs decoded) — synth-trained VAE + Writer memorisation + unconstrained decoding. |
| Next action | **Draft Plan 3b** — fresh VAE on Sweden+Singapore+Sri Lanka real corpus; enable mandatory-subset constrained decoding in eval; add country conditioning; retrain Painter+Writer. ~40 node-h estimate. |
| Tests passing | **129 / 129** (added 2 KV-cache sampler tests + 2 data_module repeat tests) |
| Ruff lint | clean (last verified pre-Plan-3a; not re-checked after KV-cache commit) |
| GPU-h burned this session (Plan 3a) | ~9.5 node-h (vs Plan 3 spec's 50) |
| GPU-h burned cumulative on project | ~25 (15 pre-3a + 9.5 Plan 3a + ~0.5 misc) |
| Tile shards on Leonardo | `$WORK/bonzai-tiles/{singapore,sri_lanka,sweden,synth}/` — 1,888 real + 5,000 synth |
| Trained checkpoints (Plan 3a, Sweden-only) | `$WORK/bonzai-plan3a-sweden/{stage_a,stage_b}/lightning_logs/.../checkpoints/epoch=49-step=10000.ckpt` (Painter 1.85 GB, Writer 3.3 GB) |
| Sample artefacts | local: [`bonzai_genai/results/plan3a-sweden-samples/`](../../bonzai_genai/results/plan3a-sweden-samples/) — 64 PNGs + 64 GeoJSONs |

## Recent commit history (most recent first)

```
ada9096 fix(eval): make fid_lite memory-safe (per-channel mean+std, not full covariance)
834bf2b docs(scripts): record Leonardo cu130 -> cu121 torch install workaround
abb48fa chore(lint): silence N812 (standard PyTorch/Lightning import idioms); auto-fixes from ruff
d868129 feat(slurm): Experiment 0 driver + eval driver scripts
1612076 feat(slurm): GPU training sbatch templates + shared Lightning trainer driver
c9541f9 feat(eval): end-to-end channel IoU + §8.2 baselines
d650f21 feat(eval): Stage B metrics (Chamfer, road graph, validity, POI placement)
38216e2 feat(eval): Stage A metrics (channel IoU, FID-lite, conditioning ablation stub)
6346ccb feat(training): Lightning Stage B + greedy Inker sampler with constrained-decoding hook
fbf2f0b feat(models): constrained decoding logit masks (mandatory subset)
cbf1267 feat(models): Inker token embed + RoPE + full transformer (self+cross attn)
c1959c1 feat(models): strided CNN raster encoder for Stage B cross-attention
6d1d3bd feat(training): EDM noise + DPM-Solver++ sampler + Lightning Stage A
10a642f feat(models): full DiT forward with conditioning paths + unpatchify
e1104e4 feat(models): DiT block with AdaLN-Zero conditioning
ad50404 feat(models): DiT patch embed + sinusoidal time embed
832992a feat(training): WebDataset LightningDataModule for raster + tokens
0f9d2b2 feat(cli): add synth-corpus subcommand for Experiment 0 dataset prep
c394523 feat(synth): extend procedural generator with density modes + diagonal roads
9710dab feat(training): Lightning VAE training module
f9b37f2 feat(models): VAE decoder + reparam + channel-aware loss
a26bd25 feat(models): VAE encoder (9-channel raster -> 64x64x4 latent)
1521969 feat(models): add config dataclasses with tiny/production presets
037c4f7 feat(deps): add torch + lightning + einops + torchmetrics + networkx; scaffold models/training/eval
d4664bd plan: phase 0b modeling layer + eval harness + experiment 0
73ad640 spec: phase 0b modeling layer + eval harness + experiment 0
78c71c6 docs: mark Phase 0a complete in STATUS.md + PROJECT.md
d4f1ff3 docs(phase-0a): record Sweden + Singapore + Sri Lanka tile manifests; mark Phase 0a complete
8e0b753 fix(data): shuffle iter_tile_centres so max_tiles samples uniformly across bbox
e103447 feat(data): refactor sampling.py to pure pyosmium with bucketed spatial index
c700101 feat(vocab): add NODE_REF token family; lift road-node cap from 512 to 8192
b30ae93 plan: insert Task 18.5 (lift road-node cap with NODE_REF tokens)
22198be docs: mark Plan Tasks 14-18 complete; refresh STATUS.md
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

## Plan progress (Phase 0b — 26 tasks, all complete)

- [x] T1: Add deps + scaffold model/training/eval dirs
- [x] T2: Model configs (TinyConfig + ProductionConfig)
- [x] T3-5: VAE encoder/decoder + LightningModule
- [x] T6-7: Extended synth procedural + synth-corpus CLI
- [x] T8: WebDataset LightningDataModule
- [x] T9-12: DiT (patch embed + AdaLN-Zero block + full module + EDM/DPM-Solver++ + LightningStageA)
- [x] T13: Strided CNN raster encoder
- [x] T14-17: Inker (token embed + RoPE + transformer + constrained decode + LightningStageB + greedy sampler)
- [x] T18-21: Eval harness (Stage A + Stage B + end-to-end + baselines)
- [x] T22-23: Slurm GPU scripts + Experiment 0 driver
- [x] T24: Local CPU dry-run (102/102 tests pass; ruff clean)
- [x] T25: Run Experiment 0 on Leonardo — go signal MET
- [x] T26: Wrap Phase 0b — `EXPERIMENT_0_REPORT.md` + STATUS / PROJECT updates committed

## Plan progress (Phase 0a — 20 + 2 bonus tasks; all complete)

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
- [x] Task 18.5: Lift road-node cap with NODE_REF tokens (commit `c700101`; cap landed at 8192)
- [x] Task 18.6: pyosmium refactor + bucketed spatial index (commit `e103447`; bonus task — original osmium-tool subprocess approach didn't work on Leonardo)
- [x] Task 19: Deploy to Leonardo and run all three country jobs (Singapore + Sri Lanka in original; Sweden after sampler-shuffle fix `8e0b753`)
- [x] Task 20: Plan completion summary (`bonzai_genai/results/PHASE_0A_COMPLETE.md`, commit `d4f1ff3`)

## What was built (final tally for Phase 0a code)

```
bonzai_genai/
├── pyproject.toml                     ✅ (deps incl. osmium>=4.3 / pyosmium)
├── README.md                          ✅
├── .gitignore                          ✅
├── conftest.py                         ✅ (sys.path workaround for editable-install)
├── .venv/                              ✅ Python 3.12 (Mac) / 3.11.7 (Leonardo)
├── src/bonzai_genai/
│   ├── config.py                      ✅
│   ├── vocab/{tokens,attributes,tokeniser}.py    ✅ (NODE_REF token family added; cap 8192)
│   ├── data/{rasteriser,tile_bundle,shard_writer,sampling}.py    ✅ (sampling: pyosmium + bucket index + shuffled centres)
│   ├── synth/procedural.py            ✅
│   └── cli/prepare_tiles.py           ✅ (synthetic + overture-region; pre-loads PBF features once)
├── configs/attributes_v1.yaml         ✅ (290 attributes, will expand to ~1,800 in Plan 5)
├── scripts/
│   ├── prepare_tiles_local.py         ✅
│   ├── leonardo_data_prep.sbatch      ✅
│   └── README.md                      ✅
├── results/                            ✅ Phase 0a manifests + PHASE_0A_COMPLETE.md
├── data/                               ⏸ (gitignored; Singapore PBF here for local dev)
└── tests/                              ✅ 55/55 pass, ruff-clean
    ├── test_config.py (5)
    ├── test_tokens.py (11)             — +4 NODE_REF tests
    ├── test_attributes.py (7)
    ├── test_tokeniser.py (9)           — +1 dense-tile test
    ├── test_rasteriser.py (8)
    ├── test_tile_bundle.py (5)
    ├── test_shard_writer.py (4)
    ├── test_synth.py (3)
    ├── test_sampling.py (2 — incl. real Marina Bay extract via pyosmium)
    └── test_round_trip.py (1)
```

## Phase 0a deliverable (on Leonardo)

| Country | Köppen | Tiles kept | Tiles attempted | Shards | Disk on `$WORK` | Slurm wall |
|---|---|---:|---:|---:|---:|---:|
| Singapore | Af | 203 | 300 | 3 | 1.9 GB | 06:17 |
| Sri Lanka | Aw | 384 | 2,000 | 4 | 3.4 GB | 04:15 |
| Sweden | Cfb | 1,301 | 5,000 | 14 | 12 GB | 11:50 |
| **Total** | | **1,888** | **7,300** | **21** | **~17.3 GB** | |

All ran on `lrd_all_serial` (budget-free). **Zero GPU-h consumed for Phase 0a.**

Bbox / source PBF for each country (Geofabrik):
- Sweden: `55.0,10.5 → 69.5,24.5` (full country) — `https://download.geofabrik.de/europe/sweden-latest.osm.pbf`
- Singapore: `1.20,103.60 → 1.48,104.05` (full island) — `https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf`
- Sri Lanka: `5.85,79.55 → 9.90,81.95` (full country) — `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf`

Singapore's 97 skips are all NODE_REF overflow (Marina Bay multi-level interchanges have >8192 unique nodes per 2 km² tile — genuinely extreme). Sri Lanka / Sweden skips are mostly the "near-empty tile" rule (`<5 features`) hitting ocean / forest / mountain.

Full hand-off: [`bonzai_genai/results/PHASE_0A_COMPLETE.md`](../../bonzai_genai/results/PHASE_0A_COMPLETE.md).

## Notes for the next agent / next session

**Phase 0a is complete.** Phase 0b begins by writing **Plan 2: synthetic smoke harness for Experiment 0** — a 1-tile overfit on a Stage A DiT to confirm the rasteriser → diffusion → sample → token-decode loop closes end-to-end before any real-data training. Plan 2 should also draft Stage A architecture (DiT, ~400M params) at code-only level (no GPU yet).

### Key environmental facts (still relevant)

- **Python:** Mac has `python3.12`, Leonardo has `python/3.11.7` (spack module). Pyproject allows ≥3.11.
- **Bash session loses cwd between calls.** Always `cd /Users/umaraslam/Documents/dynamo/Bonzai-OSM/bonzai_genai && ...`.
- **Editable-install gotcha** (still unfixed): `pip install -e .` writes a `.pth` file but Python's site processing isn't picking it up on either Mac or Leonardo. Workarounds in place: `bonzai_genai/conftest.py` (for pytest) and `bonzai_genai/scripts/prepare_tiles_local.py` (sys.path-prepend before import).
- **rsync exclude is unanchored by default.** `--exclude=data` strips both root `data/` (intentional) AND `src/bonzai_genai/data/` (broken). Use `--exclude='/data/'` (root-anchored).
- **`module load python/3.11.7` is NOT needed in the Slurm batch script** — `source .venv/bin/activate` resolves correctly. The 06:17 / 04:15 / 11:50 jobs all completed without a module load.
- **Leonardo datamover (`data.leonardo.cineca.it`)** accepts a restricted-shell command set (wget / rsync / scp / curl); arbitrary commands like `echo` are rejected. Pattern: `ssh ... wget -O <path> <url>`.

### Open follow-ups discovered during execution

1. **Attribute vocab YAML ships ~290 tokens.** Expansion to full FSQ leaves (~1,800) deferred to Plan 5.
2. **Editable-install path issue** — root cause unknown; conftest + sys.path-prepend in scripts works around it.
3. **Marina Bay still overflows the 8192 NODE_REF cap.** Either bump to 16384 (vocab cost) or add a tile-level pre-crop that splits >8k-node tiles in two. Decide before Plan 2.
4. **Multipolygon-relation buildings / landuse** are not picked up by the pyosmium handler — only simple closed ways. Add `osmium.area.AreaManager` support so big malls / lakes / parks land in the dataset (currently we lose the tag on these).
5. **`max_tiles` is a hard cap.** Sweden's 5,000 yielded 1,301 kept; ~74% skipped as near-empty. Bump or convert to a "keep until N successful" loop.
6. **Stratified sampling** deferred to Plan 5: Phase 0a uses bbox-uniform shuffled-random.
7. **`bonzai_genai/.gitignore`** initially over-broad (`data/` matched `src/bonzai_genai/data/`). Fixed to root-anchored `/data/` `/shards/`.

## Three countries for v1 de-risking (Phase 0a — done)

| Country | Köppen | Geofabrik URL | Bbox |
|---|---|---|---|
| Sweden | Cfb / Dfb | `https://download.geofabrik.de/europe/sweden-latest.osm.pbf` | `55.0,10.5 → 69.5,24.5` (full country) |
| Singapore | Af | `https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf` (extract) | `1.20,103.60 → 1.48,104.05` (full island) |
| Sri Lanka | Af / Aw | `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf` | `5.85,79.55 → 9.90,81.95` (full country) |

## Blockers / open questions

None blocking Phase 0b. Plan 2 can be drafted at any time.

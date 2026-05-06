# Bonzai-OSM — Project Log

> Generative city model trained on world-scale Overture / OSM / Foursquare data.
> Long-lived dev branch: `genai-city-model`. Never merged to `main`.

**PI:** Umar Aslam — Leonardo username `uaslam00`, project email `aslamumar16@gmail.com`, personal `aslamumar012@gmail.com`.
**Project account on Leonardo:** `AIFAC_P02_222` (EuroHPC allocation).
**Doc owner:** Claude + Umar. Every material decision lands here or in `docs/superpowers/`.

---

## 1. Goal (one paragraph)

Train a two-stage generative model — a "Sketcher" (latent diffusion on rasterised tiles) and an "Inker" (autoregressive transformer on tile-local vector tokens) — that produces novel, geometrically valid, plausibly-real **vector** city layouts (roads, building footprints, POIs, land use) from a text prompt. Primary user persona: simulation engineer at an AV / robotics / game studio who wants `generate_city(prompt, area, conditioning) → GeoJSON`. Secondary: academic publication; mapping-infill via inpainting.

## 2. Architecture in one paragraph

A 9-channel raster (roads × 4 classes + buildings × 2 + water + green + urban) is the structural skeleton; coordinates inside a tile are **always tile-local**, quantised to 512 bins per axis, never global lat/lon. Stage A is a ~400 M-param DiT diffusion model with EasyControl-style LoRA conditioning on text + region tags + (optional) terrain / coastline / style anchor / constraint mask. Stage B is a ~750 M-param decoder-only transformer with cross-attention to a CNN encoder of the raster, ~2,884-token vocabulary, 16 k context, structured constrained decoding for valid GeoJSON output. Stages trained independently; domain gap closed by training-time noise augmentation in Stage B.

Full design: [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md).
Full brainstorm: [`docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md`](docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md).
Live progress: [`docs/superpowers/STATUS.md`](docs/superpowers/STATUS.md).

## 3. Allocation

| Item | Value |
|---|---|
| Cluster | Leonardo Booster (CINECA, Bologna) |
| Account | `AIFAC_P02_222` |
| Initial budget | 40,000 local core-h = ~1,250 Booster GPU-node-h = ~5,000 single-A100-h |
| Project window | 2026-03-11 → 2026-06-11 (3 months) |
| Extension policy | Email `superc@cineca.it` with justification; PI confirms strong v1 unlocks fundraising for additional compute |

Spend rate:
- `boost_usr_prod`: 1 node × 1 h = 32 core-h (4× A100s).
- `dcgp_usr_prod`: 1 node × 1 h = 112 core-h (CPU-only; **NOT enabled** for this project).
- `lrd_all_serial`: 4 cores / 30 GB / 4 h walltime, **budget-free**.

**Spend rule:** Booster billing is reserved for actual model training. All preprocessing, data prep, eval-metric computation runs free on `lrd_all_serial`.

## 4. Storage

| Area | Path / var | Default quota | Scope | Retention |
|---|---|---|---|---|
| `$HOME` | per-user | 50 GB | per user | permanent |
| `$WORK` | `/leonardo_work/AIFAC_P02_222` | 1 TB | per project | 6 months past project end |
| `$FAST` | NVMe | 1 TB | per project | 6 months past project end |
| `$CINECA_SCRATCH` | `/leonardo_scratch/large/userexternal/uaslam00` | ~20 TB practical | per user | 40-day rolling auto-delete |

**Verified state on 2026-04-15:**
- `$WORK`: ~1.4 MB (clean).
- `$FAST`: 0 KB.
- `$CINECA_SCRATCH/osm/raw/planet-latest.osm.pbf`: ~92 GB, checksum verified.

## 5. Phased plan

| Phase | What | Compute | Wall time |
|---|---|---|---|
| 0 — Code scaffolding | bonzai_genai package, data prep, model defs, eval, Slurm | 0 GPU-h | ~1.5 weeks |
| **0a (in progress)** | **Data prep pipeline + Sweden / Singapore / Sri Lanka tile datasets on `$WORK`** | **0 GPU-h** | **~1 week** |
| 0b | Synthetic smoke harness (Experiment 0) + Stage A code + Stage B code | ~10 GPU-h | ~2 weeks |
| 1 | De-risking experiments 0–4 on the three-country dataset | ~250 GPU-h | ~2 weeks |
| 2 | Wave 1 production data prep (Western Europe, ~150k tiles) | 0 GPU-h | ~3 days |
| 3 | Production VAE training | ~50 GPU-h | ~2 days |
| 4 | Production Stage A training (Sketcher) | ~800 GPU-h | ~2-3 weeks |
| 5 | Production Stage B training (Inker) | ~2,500 GPU-h | ~3-4 weeks |
| 6 | Eval + ablations + paper | ~500 GPU-h | ~2 weeks |
| 7 | Hand-off / publication / pilot conversations | 0 GPU-h | — |

**Total estimate:** ~4,800 GPU-h. Initial 1,250 covers Phases 0–4. Extension request planned after Phase 1 de-risking shows green.

## 6. De-risking countries (Phase 0a)

Three countries chosen for climatic and morphological contrast:

| Country | Köppen | Urban form | Geofabrik size | Why valuable |
|---|---|---|---|---|
| Sweden | Cfb / Dfb | Northern European low-rise + sparse rural | ~700 MB | Cold-temperate baseline; lots of water/green; tests sparsity |
| Singapore | Af | Tropical ultra-dense gridded high-rise | ~30 MB | Maximum urban-density contrast; pure built-up |
| Sri Lanka | Af / Aw | Tropical mixed urban + rural, dense low-rise organic | ~150 MB | South Asian mid-density; tests rural variety |

**Why three (not two):** PI requested "different areas" plural for diverse de-risking signal. The trio covers three Köppen zones (Cfb/Dfb, Af, Af/Aw) and three urban morphologies.

## 7. Open action items

| # | Action | Owner | Status |
|---|---|---|---|
| A1 | Phase 0a: build `bonzai_genai` data prep package | Claude + Umar | **done** (2026-05-04) |
| A2 | Generate Sweden / Singapore / Sri Lanka tile datasets on `$WORK` | Claude + Umar | **done** (1,888 tiles / 17.3 GB; see `bonzai_genai/results/PHASE_0A_COMPLETE.md`) |
| A3 | Write Plan 2 (synthetic smoke harness + Stage A code) + run Experiment 0 | Claude + Umar | **done** (2026-05-04; 26 tasks landed, smoke ran green; see `bonzai_genai/results/EXPERIMENT_0_REPORT.md`) |
| A4 | Write Plan 3 + run Experiments 1+2 on Sweden (de-risked subset) | Claude + Umar | **done** (2026-05-06; 12-task plan landed, Sweden-only Plan 3a ran end-to-end; architecture verified, end-to-end pipeline not green; see `bonzai_genai/results/PLAN_3_REPORT.md`) |
| A5 | Write Plan 3b — fresh multi-country VAE + constrained eval + country conditioning + extend to Singapore + Sri Lanka | Claude | **next** |
| A6 | Run Experiments 3–4 (cross-domain transfer; full Phase 1) | Claude + Umar | pending |
| A7 | Email superc@cineca.it for compute extension after Phase 1 green | Umar | later |
| A8 | Submit paper (NeurIPS / CVPR / ICLR) | Claude + Umar | later |

## 8. Change log

- **2026-04-15** — initial cleanup; planet PBF download verified; Luxembourg test pipeline validated on `lrd_all_serial`.
- **2026-04-29** — Overture ∪ Foursquare dedup completed (127 M unified POIs); FINDINGS.md report committed on `overture-map` branch.
- **2026-05-02** — design brainstorm completed; spec drafted (Sketcher + Inker, two-stage hybrid, 2 km² tiles, ~2,884-token vocab, EasyControl conditioning).
- **2026-05-02** — primary v1 user persona locked: AV / robotics simulation engineer.
- **2026-05-03** — design philosophy updated: quality over thrift (1,250 GPU-h is initial allowance, not cap).
- **2026-05-03** — design spec finalised at `docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`; committed.
- **2026-05-03** — Plan 1 (Phase 0a data prep) written: 20-task TDD plan; committed.
- **2026-05-03** — branched off `overture-map` to long-lived `genai-city-model`. Removed obsolete test files (Luxembourg test SLURM jobs, early exploration scripts, prior overture-map work). Started executing Plan 1.
- **2026-05-03** — de-risking country triple updated: Sweden + Singapore + Sri Lanka (replacing Luxembourg + Iceland) for greater geographic and climatic diversity.
- **2026-05-04** — Phase 0a **complete**: 1,888 tile shards live on Leonardo (`$WORK/bonzai-tiles/{singapore,sri_lanka,sweden}/`, 17.3 GB total) with 0 GPU-h consumed. Bonus task 18.5 lifted the road-node cap to 8,192; bonus task 18.6 swapped `osmium-tool` subprocess for pure pyosmium with a bucketed spatial index (Leonardo doesn't ship `osmium-tool`). Sampler shuffle fix (`8e0b753`) made `iter_tile_centres` deterministically shuffle so `max_tiles` samples uniformly across big bboxes. Hand-off: `bonzai_genai/results/PHASE_0A_COMPLETE.md`.
- **2026-05-04** — Phase 0b **complete**: Plan 2 (26 TDD tasks, ~5k LoC + ~1.5k LoC tests) landed the modeling layer (custom-from-scratch VAE / DiT / Inker / raster CNN encoder), three PyTorch Lightning training loops, the full §8 eval suite, and Slurm GPU job templates. Experiment 0 ran end-to-end on Leonardo `boost_usr_prod` (~1 GPU-h, way under the 12-30 budget) — VAE / Stage A / Stage B all trained without divergence; eval suite produced clean numbers; validity rate **1.00**. Two bugs surfaced and were fixed inline: torch wheel CUDA-version mismatch (cu130 → cu121) and `fid_lite` memory blow-up. Hand-off: `bonzai_genai/results/EXPERIMENT_0_REPORT.md`.
- **2026-05-06** — Plan 3a **(Sweden-only)** ran end-to-end on Leonardo `boost_usr_prod` 4×A100 nodes. Painter (153 M DiT) trained 50 epochs in 4:58 wall, train/loss 52.4 → 0.26. Writer (271 M Inker + 3 M raster encoder) trained 50 epochs in 2:23 wall, train/loss 6.25 → 0.0006 (full memorisation of the 1,301-tile corpus). Two infrastructure fixes shipped: `data_module.py` `.repeat()` on the train stream so `BONZAI_LIMIT_TRAIN_BATCHES` cleanly bounds DDP epochs (`c0e674e`); KV-cached greedy Inker sampler that collapsed 8k-context decode from O(N³) to O(N) and turned a 1-h-timeout eval into a 30-min run (`97c691c`). Sample-from-ckpt eval produced 64 PNGs + 64 GeoJSONs. **Architecture verdict: ✅** loss curves are clean and monotone-down. **End-to-end verdict: ❌** Painter samples reproduce only the Sweden marginal distribution (sparse, faint, no coherent road graphs); 0/64 GeoJSONs decode (Writer emits invalid token grammars when conditioned on Painter rasters). Three compounding causes identified: synth-trained VAE reused as latent decoder, Writer over-memorisation on the small Sweden corpus, `constrained=False` in eval. Total burn ~9.5 node-h (vs Plan 3 spec's 50). Hand-off: `bonzai_genai/results/PLAN_3_REPORT.md`. Next: draft Plan 3b — train fresh VAE on Sweden+Singapore+Sri Lanka, enable mandatory-subset constrained decoding, add country conditioning, retrain Painter+Writer.

## 9. Where everything lives

| Topic | File |
|---|---|
| Live status / handoff | [`docs/superpowers/STATUS.md`](docs/superpowers/STATUS.md) |
| Design rationale | [`docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md`](docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md) |
| Design spec | [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md) |
| Phase 0a plan | [`docs/superpowers/plans/2026-05-03-phase-0a-data-prep-pipeline.md`](docs/superpowers/plans/2026-05-03-phase-0a-data-prep-pipeline.md) |
| Production code (Phase 0a) | `bonzai_genai/` |
| Leonardo command reference | [`commands.md`](commands.md) |
| Planet download helper | [`scripts/leonardo_download_planet.sh`](scripts/leonardo_download_planet.sh) |

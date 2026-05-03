# Bonzai-OSM Generative City Model — Design Spec

> **Date:** 2026-05-03
> **Author:** Umar Aslam (PI), with Claude assistance
> **Status:** Design phase complete; awaiting implementation plan
> **Branch:** `overture-map`
> **Source brainstorm log:** [`docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md`](../brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md)

---

## 1. Goal

Train a generative model that, given a text prompt and optional spatial conditioning, produces **novel, geometrically valid, plausibly-real city layouts as vector geometry** — roads, building footprints, points of interest, land use polygons — directly consumable by simulation, GIS, and game engines.

The headline interaction:

```
generate_city(
    prompt = "dense European commercial district, mid-rise, coastal",
    area  = 4.2 km²,
    conditioning = { terrain_DEM, coastline_mask, style_anchor, constraint_mask }  # all optional
) → GeoJSON FeatureCollection
```

The output is real vector data — not pixels, not voxels, not a NeRF. Each road has a class label and a polyline; each building has a footprint polygon, a class, and a height; each POI is a labelled point; each land-use region is a typed polygon.

## 2. Target user & business value

### Primary v1 user persona

The **simulation engineer at an autonomous-vehicle (AV), robotics, or game studio**. They call a function, get GeoJSON, and drop it into Unity / Unreal / CARLA / Foretellix / their own renderer.

### Why this persona

| Attribute | Why it matters |
|---|---|
| Funded buyer | AV companies and game studios have real procurement budgets and a track record of paying for sim-data |
| Aligned quality bar | Both want plausibility, not pixel-perfect real-world fidelity — exactly what generative models produce well |
| Vector-native output | GeoJSON / OpenDRIVE / glTF are all vector-native; our output decodes directly |
| Batch latency tolerated | No real-time constraint; we can afford 50–250 diffusion steps and beam search for quality headroom |
| No regulatory hurdles | Unlike urban planning (zoning approvals) or defence (export control), sim is commercial-friendly |
| Academic publication path is parallel | Open weights + benchmark = sim-data product *and* publishable artifact |

### Pitches at three audience levels

**Layperson / dinner-table:**
> *"We're teaching a computer what real cities look like — by showing it millions of them — so it can invent new ones. You type 'a dense European commercial district,' and it draws you a plausible map: streets in the right places, building footprints, shops, parks. Not a picture of a map, but the actual data a video game or a self-driving simulator can use."*

**Technical / non-ML (CTO, journalist, investor) — the canonical pitch:**
> *"Imagine Stable Diffusion, but for **maps** instead of images. SD generates pictures from a prompt; ours generates the underlying vector data of cities — roads, buildings, businesses — from a prompt. Self-driving companies pay around $5,000 per kilometre to map real cities and they only have a few; we generate unlimited plausible ones on demand. Game studios pay artists for months to handcraft one city; we generate one in seconds."*

**ML reviewer (NeurIPS / CVPR / ICLR):**
> *"We jointly train (i) a small multi-channel latent diffusion model on rasterised OSM/Overture tiles, and (ii) an autoregressive transformer that vectorises the diffusion's output into typed primitives (roads, buildings, POIs, land polygons) with tile-local quantised coordinates. The contribution is a tractable raster-to-vector decoding scheme for **semantically labelled** geometry at city scale, with EasyControl-style spatial conditioning on terrain, region embedding, and constraint masks."*

### Confidence calibration

- **~70%** confidence this produces a publishable research artifact within the project window.
- **~40%** confidence v1 alone gets paid integration with an AV / game studio (real procurement is slow).
- **~90%** confidence some version of this becomes a real product within 3–5 years.

Treated as a research-grade bet with a strong commercial tail. Academic deliverable is the safer first milestone; commercial validation comes after.

### Secondary buyers (zero extra cost to keep in scope)

- **Academic publication.** Paper writes itself in parallel.
- **Mapping / GIS infill.** Same model used in inpainting mode (mask the part to fill, condition on the rest).
- **Geospatial data augmentation.** Just publish the raw outputs.

### Buyers explicitly deferred to v2+

- Urban planning (regulatory plausibility raises the quality bar significantly).
- Defence / national security (contractual complexity).
- Real estate proptech (niche; geographic-pinning quality dependent).

## 3. System architecture overview

The system has **four units**. Three are runtime components, one is offline tooling. Each unit is independently buildable, trainable, and testable.

```
[1] DATA PREP            [2] STAGE A              [3] STAGE B            [4] EVAL
    (offline,                (Sketcher: latent        (Inker: AR             (offline,
     free CPU)                diffusion)               vectoriser)            free CPU)

OSM / Overture / FSQ ─►  Multi-channel raster ─►  Vector geometry ─►   Metrics + samples
                          (the "skeleton")         (roads, buildings,
                                                    POIs, land)
```

### The "Sketcher and Inker" mental model

Two artists work in sequence:

- **The Sketcher** (Stage A): a small diffusion model that paints a rough coloured sketch of a 2 km × 2 km tile on a 9-channel canvas. Different colours for roads, buildings, water, parks. Quick, fuzzy, shows *where* things go, not *what* they are exactly.

- **The Inker** (Stage B): a small autoregressive transformer that reads the Sketcher's coloured rough and carefully traces every shape with clean vector lines and semantic labels. Building blobs become precise polygons with `class=residential`. Road blobs become a graph with class labels. POI dots become labelled points with categories. Output: clean GeoJSON.

### Three load-bearing splits

1. **Local coordinates inside every tile.** Always 0–2048 m, quantised to 512 bins. The model never sees a global lat/lon. Vocabulary stays bounded forever. "Where on Earth" is handled by Stage A's prompt conditioning, not by the geometry vocabulary. **This is the single most important design decision.**

2. **Raster vs vector responsibility.** Stage A handles structural skeleton (presence/density of roads, buildings, water, green, urban). Stage B handles fine-grained semantics (POI categories, building subtypes, exact polygon vertices). Keeps Stage A small (~400 M params) and concentrates the rich vocab (~1,800 attribute tokens) in Stage B where it matters.

3. **Stages trained independently in v1.** Stage A on (raster, prompt) pairs. Stage B on (ground-truth raster, vector tokens) pairs. They only meet at sample time. Joint fine-tuning is a v2 lever if quality demands it. Domain-gap risk (clean ground-truth vs noisy sampled raster) is mitigated by training-time noise augmentation in Stage B (see §7).

## 4. Data sources & data prep pipeline

### 4.1 Three open vector data sources

| Source | What it's best at | Status |
|---|---|---|
| **OpenStreetMap (OSM)** | Raw global coverage, rich tags. ~85 GB planet PBF. | Already downloaded |
| **Overture Maps 2026-04-15.0** | Cleaned, license-clean, harmonised schema. 75 M places, 2.5 B buildings, 344 M road segments. | Already accessible |
| **Foursquare OS Places** | Deep POI labels with hierarchical categories. ~100 M places. | Already downloaded |

**Primary skeleton:** Overture (roads, buildings, land use).
**POI semantics:** Foursquare hierarchical categories (11 / 150 / 700 cascade).
**Tag fallback:** OSM via Overture Bridge Files (recovers labels on the ~6% of buildings that have OSM tags).

### 4.2 What one training example looks like

A 2.048 km × 2.048 km square anywhere on Earth becomes one bundle:

1. **Pull vector ground truth** — query Overture/OSM for everything in the square: every road, every building, every river, every shop.
2. **Convert to tile-local coordinates** — quantise to 512 bins per axis. The model never sees global lat/lon.
3. **Paint a 512×512×9 raster** ourselves from the vector data — one colour channel per layer (3 road classes + binary roads, buildings + density, water, green, urban).
4. **Serialise the same data as a token sequence** in stable order: land polygons → roads → buildings → POIs. ~5,000–14,000 tokens per tile.
5. **Attach metadata** — country, climate zone, density bucket, primary land use — for prompt conditioning at training time.

Bundle on disk: `(raster.npz, tokens.txt, metadata.json)`.

### 4.3 The 9 raster channels

| # | Channel | Encoding |
|---|---|---|
| 1 | All roads (any class) | Binary mask, 1-px line |
| 2 | Major roads (motorway / trunk / primary) | Binary mask, 2-px line |
| 3 | Mid-tier roads (secondary / tertiary) | Binary mask |
| 4 | Minor roads (residential / service) | Binary mask |
| 5 | Building footprints | Binary mask |
| 6 | Building density | Continuous 0–1 (Gaussian-blurred footprints, σ ≈ 32 px) |
| 7 | Water (rivers / lakes / ocean) | Binary mask |
| 8 | Green (parks / forests / fields / agricultural) | Binary mask |
| 9 | Built urban land use (residential / commercial / industrial) | Binary mask |

POI categories are deliberately **not** in the raster — they live in Stage B's vocabulary instead.

### 4.4 Tile sampling — stratified, not uniform

Uniform sampling would over-weight US Midwest and ocean. We stratify by:

- **Country** — each contributes proportional to a fixed budget, not population.
- **Density bucket** — rural / suburban / urban / dense urban, equal-weight.
- **Climate zone** — Köppen classification, equal-weight.

**Skip rules:** drop tiles with <20 features, >10,000 features, or >50% water.

The whole stratification is a DuckDB SQL query running free on `lrd_all_serial`.

### 4.5 Training-set sizing

| Item | Estimate |
|---|---|
| Tiles for de-risking phase (Sweden + Singapore + Sri Lanka) | ~5,000 |
| Tiles for **Wave 1** (Western Europe — default production) | ~150,000 |
| Tiles for **Wave 2** (planet-scale — extension) | ~500,000 |
| Raster per tile | ~50–100 KB compressed |
| Vector tokens per tile | ~10–30 KB |
| Metadata per tile | ~0.5 KB |
| Total per tile | ~80–130 KB |
| **Wave 1 training corpus** | **~15–20 GB** |
| **Wave 2 training corpus** | **~50–65 GB** |

**Naming convention:** "Wave 1 / Wave 2" denotes the **geographic coverage tier** of the training corpus. "Phase 0 through 7" (in §9.2) denotes the **execution phase** of the rollout. The two are independent axes — Wave 1 (Western Europe) is the default for execution Phases 4–5; Wave 2 (planet) is a stretch only attempted with a budget extension.

Easily fits in `$WORK` (1 TB) or `$FAST` (1 TB NVMe).

### 4.6 On-disk format

WebDataset format (PyTorch-native shards):

```
$WORK/bonzai-tiles/
├── shard-00000.tar    ← 1k tiles, ~100 MB
├── shard-00001.tar
├── ...
├── shard-00499.tar
├── manifest.parquet   ← which-tile-where + stratification metadata
└── stats.json         ← global statistics for normalisation
```

The pipeline that **builds** these shards runs free on `lrd_all_serial`. Generating Phase 1's 150k tiles takes ~1–2 days of wall time across multiple parallel free jobs. **Zero billed compute spent on data prep.**

### 4.7 The building-label gap

In Overture, ~94% of building footprints have no class/subtype label. Two paths around it:

1. **Default for v1:** emit `BUILDING_OPEN class=UNKNOWN` for unlabelled cases. The model learns class only when it's there.
2. **Optional enrichment via OSM Bridge Files:** join Overture buildings to original OSM IDs, then to OSM tags (`building=*`, `shop=*`, `amenity=*`). Roughly doubles labelled-building count. Free join on `lrd_all_serial`.

For v1, do (1) first; add (2) once Stage B is training stably and we want richer building semantics.

## 5. Stage A — The Sketcher

### 5.1 Configuration

| Parameter | Value |
|---|---|
| Tile side | 2048 m |
| Image dimension | 512 × 512 px |
| Spatial resolution | 4 m/px |
| Tile area | 4.2 km² |
| Channels | 9 |
| Latent shape (after VAE) | 64 × 64 × 4 |
| DiT parameters | ~300–500 M |

### 5.2 Pipeline

```
9-ch raster (512×512×9) → VAE encoder → latent (64×64×4) → DiT denoises
                                                        ↓
                                                latent → VAE decoder → 9-ch raster
```

8× spatial compression. Memory and per-step compute drop ~64×.

### 5.3 VAE

A small (~10 M parameter) multi-channel VAE compresses 512×512×9 to 64×64×4. Trained first as a one-shot reconstruction task with channel-aware loss: binary cross-entropy on the binary masks (channels 1–5, 7–9), mean-squared-error on the continuous density channel (6), plus standard KL regularisation. ~1 day on a single A100 (~15 GPU-h). Frozen before DiT training begins.

### 5.4 DiT (Diffusion Transformer) backbone

| Parameter | Value |
|---|---|
| Parameters | ~300–500 M (target 400 M) |
| Layers | 24 |
| Hidden dim | 1024 |
| FFN expansion | 4× |
| Patch size | 2 (over 64×64 latent → 32×32 = 1,024 transformer tokens) |
| Conditioning style | AdaLN-Zero |
| Attention | FlashAttention 3 |
| Noise schedule | EDM |

This is "modest but solid 2026 SOTA" sizing — comparable to base SDXL.

### 5.5 Conditioning paths

Three conditioning types flow in parallel:

| Type | What | How |
|---|---|---|
| **Text** | "dense Asian commercial, 2 km², coastal" | CLIP-class text encoder (2026 model TBD) → 768-dim embedding → cross-attention |
| **Region tags** | country, climate zone, density bucket, primary land use | Learned lookup tables → summed conditioning token |
| **ControlNet (EasyControl LoRA)** | terrain DEM, coastline mask, style anchor, constraint mask | Per-input LoRA adapters injecting conditioning tokens; trainable independently |

EasyControl-style LoRA injection ([arXiv 2503.07027](https://arxiv.org/abs/2503.07027)) is the 2026-SOTA replacement for classic ControlNet on DiT — lighter, more flexible, can be added incrementally.

### 5.6 Training schedule

- Optimiser: AdamW, learning rate ~1e-4, cosine decay.
- Epochs: 50–100 over the Wave 1 production corpus (~150k tiles).
- Classifier-free guidance: 10% of training has conditioning dropped to null (enables CFG at sample time).
- **Phased schedule:**
  - Epochs 1–20: unconditional only (learn base city distribution).
  - Epochs 21–50: text + region tags conditioning.
  - Epochs 51–100: full ControlNet LoRA conditioning.
- Checkpoint every 30 minutes, save to `$WORK`.

### 5.7 Sampling

- 50 denoising steps with DPM-Solver++ (fast, high-quality).
- Classifier-free guidance scale ~7.5 (Stable Diffusion default; tune later).
- Output: 9-channel 512×512 raster in [0, 1] per channel.

## 6. Stage B — The Inker

### 6.1 Vocabulary (~2,884 tokens)

| Family | Count | Examples |
|---|---|---|
| **Structural** | ~30 | `BOS`, `EOS`, `LAYER_LAND`, `LAYER_ROADS`, `LAYER_BUILDINGS`, `LAYER_POIS`, `BUILDING_OPEN`, `BUILDING_CLOSE`, `LAND_POLY_OPEN`, `LAND_POLY_CLOSE`, `ROAD_NODE`, `ROAD_EDGE`, `ROAD_EDGE_END`, `POI` |
| **X-coordinates** | 512 | `x_0` … `x_511` (4 m bin) |
| **Y-coordinates** | 512 | `y_0` … `y_511` |
| **Attributes** | ~1,800 | `road_class=motorway`, `poi=cafe`, `building_class=residential`, `land_class=park`, `water_class=river`, etc. (carried over from empirical Overture/FSQ vocabulary analysis) |
| **Heights / numeric** | ~30 | `height_5m`, `height_10m`, …, `height_NA` (5 m quantisation, NA for unknown) |

Total: ~2,884 tokens. Tiny by LLM standards.

### 6.2 Reading order within a tile

```
BOS
LAYER_LAND
  LAND_POLY_OPEN class=water
    x_30 y_42  x_31 y_45  x_34 y_48  ...
  LAND_POLY_CLOSE
  LAND_POLY_OPEN class=park
    ...
  LAND_POLY_CLOSE
LAYER_ROADS
  ROAD_NODE x_50 y_80      (node id 0)
  ROAD_NODE x_95 y_82      (node id 1)
  ROAD_NODE x_120 y_82     (node id 2)
  ROAD_EDGE class=residential nodes=[0,1]
  ROAD_EDGE class=secondary nodes=[1,2]
  ...
LAYER_BUILDINGS
  BUILDING_OPEN class=residential height=12m
    x_110 y_140  x_118 y_140  x_118 y_148  x_110 y_148
  BUILDING_CLOSE
  ...
LAYER_POIS
  POI class=cafe x_115 y_145
  POI class=hardware_store x_205 y_88
  ...
EOS
```

**Rationale for the order:**
- Land first → biggest features, set global context.
- Roads next → structural skeleton other things attach to.
- Buildings → fill gaps between roads.
- POIs last → smallest, depend most on surrounding context.

**Within each layer:** primitives sorted by raster-scan order of centroid (top-to-bottom, left-to-right). Stable, deterministic, learnable.

For roads: declare all unique nodes first, then describe edges by node-index references. Much more compact than re-emitting coordinates per edge.

### 6.3 Architecture

| Parameter | Value |
|---|---|
| Parameters | 500 M – 1 B (target 750 M) |
| Layers | 24–32 |
| Hidden dim | 1024–1280 |
| FFN expansion | 4× |
| Context length | 16 k tokens (32 k stretch) |
| Positional encoding | RoPE |
| Attention | FlashAttention 3 |
| Cross-attention | To raster CNN encoder output (32×32×768 feature grid) |

A small CNN encoder (~30 M params, 4-layer strided conv) compresses the 9-channel 512×512 raster to a 32×32×768 feature grid. Inker cross-attends to that grid at every layer. Reads the raster directly, not the VAE latent — preserves pixel-level detail.

### 6.4 Constrained decoding

Naive autoregressive generation can produce invalid GeoJSON (open polygons, malformed coordinate pairs, layer ordering violations). We prevent that with logit masking during sampling:

| Constraint | Enforcement |
|---|---|
| Coordinate pairs | After `x_*`, the next token must be `y_*`. Other logits masked. |
| Polygon closure | After `BUILDING_OPEN`, only `BUILDING_CLOSE` ends; ≥ 3 coord pairs minimum. |
| Polygon non-self-intersection | At each candidate vertex, run quick crossing check against in-progress polygon; mask vertices that would cross. |
| Layer order | `LAYER_LAND` before `LAYER_ROADS` before `LAYER_BUILDINGS` before `LAYER_POIS`; structural masking enforces this. |
| Road edge node references | `ROAD_EDGE nodes=[i,j]` indices masked to nodes already declared in this tile. |
| Building fields | After `BUILDING_OPEN`, must emit class then height before opening polygon. |

These are pure decoder-time logit masks — no retraining.

### 6.5 Training

- Cross-entropy on next-token prediction, teacher-forcing.
- Optimiser: AdamW, learning rate ~3e-4, cosine decay.
- Epochs: 50–100.
- **Domain-gap mitigation:** add Gaussian noise (σ = 0.05–0.2) to the raster input each step. Closes the gap between training (clean ground-truth raster) and inference (noisier sampled raster from Stage A).
- Checkpoint every 30 minutes to `$WORK`.

### 6.6 Sampling

- Greedy decoding (default, fastest).
- Beam search (size 4) when quality matters.
- Constrained decoding active throughout.
- Typical 4 km² tile: 5–30 seconds end-to-end.

## 7. Bridging the stages

The two stages are trained independently, then composed at sample time:

```
TRAIN-TIME:
  Stage A: train on (prompt, ground-truth raster) pairs
  Stage B: train on (ground-truth raster + noise, ground-truth tokens) pairs

SAMPLE-TIME:
  prompt → Stage A → sampled raster → Stage B → vector tokens → GeoJSON
```

### 7.1 The domain gap

Stage B's training input is the **clean** ground-truth raster derived from real OSM data. Stage B's inference input is the **sampled** raster from Stage A — slightly noisy, sometimes ambiguous, occasionally hallucinating.

### 7.2 Mitigations (in order of escalation)

1. **Training-time noise augmentation** (default for v1). Each Stage B training step adds Gaussian noise (σ ∈ [0.05, 0.2] randomly sampled) to the input raster. The Inker learns to be robust to small perturbations. Cheap, no architecture changes.

2. **Schedule sampling** (if (1) is insufficient). Mid-training, occasionally swap the ground-truth raster for a Stage A-sampled raster on the same prompt. Forces the Inker to learn from the actual distribution it'll see at inference.

3. **Joint fine-tune at the end** (if (1) and (2) are insufficient). After both stages are independently converged, run a short joint fine-tune end-to-end with a combined loss. Increases training cost ~2× but closes the gap directly. Defer to v2 unless quality demands it.

For v1, start with (1) only. Escalate based on Experiment 3 (end-to-end domain gap test) results.

## 8. Evaluation harness

### 8.1 Three metric families

#### Stage A (Sketcher) metrics
- **Per-channel IoU vs ground-truth distribution** on a 1,000-sample held-out set.
  - Targets: roads ≥ 0.45, buildings ≥ 0.30, land-use ≥ 0.50.
- **FID-like score on VAE latents** (track over training, lower is better).
- **Conditioning fidelity** — generate 100 "dense urban" + 100 "rural" prompts, measure if building density differs in the right direction.

#### Stage B (Inker) metrics
- **Building Chamfer distance** (target: < 5 m average).
- **Building IoU** (polygon vs polygon, target: > 0.50).
- **Road graph topology:**
  - Junction degree distribution KL-divergence from real.
  - Connected components per km² match.
  - Average edge length match.
- **POI placement match** — distance from each predicted POI to nearest ground-truth POI of same category. Target: ≥ 60% within 50 m.
- **Geometric validity (headline metrics):**
  - ≥ 95% non-self-intersecting buildings
  - ≥ 90% road graphs single-component (where ground truth is)
  - < 2% overlapping polygons
  - ≥ 80% POIs inside an appropriate land-use zone

#### End-to-end metrics
- **Tile-statistics JS-divergence** — sampled vs real distributions of building count, footprint area, road length per km², road class proportions, POI count per category. Target: most stats < 0.1 JS distance.
- **Real-vs-fake CNN classifier** — train a small classifier to distinguish real from generated raster tiles; **fool rate** is our headline (target: ≥ 30% at v1).
- **LLM-as-judge sample quality** — present 200 (prompt, generated) + 200 (prompt, real) to a GPT-4-class judge; target fool rate ≥ 25%.
- **Conditional generation correctness** — humans/LLM rate fit-to-prompt 1–5 across 50 samples per prompt category. Target average > 3.5.

### 8.2 Baselines (compute once, pre-training)

| Baseline | Purpose |
|---|---|
| Real-vs-real JS divergence | Lower bound — natural noise floor |
| Real-vs-uniform-random | Upper bound — what bad looks like |
| Real-vs-Perlin-noise | "Structured but meaningless" reference |
| Real-vs-CityEngine samples | Procedural-rules-based competitor (if obtainable) |

After training, our metrics should fall **between real-vs-real and real-vs-uniform-random** — closer to real-vs-real means better.

### 8.3 Geometric validity sub-harness (per Stage B checkpoint)

After every Stage B training checkpoint:
1. Sample 500 tiles end-to-end.
2. Decode tokens to GeoJSON.
3. Run Shapely `is_valid` on every polygon; count failures.
4. Run NetworkX connectivity on every road graph; count failures.
5. Check no two buildings overlap (Shapely `intersects` with positive area).
6. Check no buildings inside water polygons (basic land-use sanity).

Output: `(total_tiles, %valid_buildings, %valid_roads, %overlap_free, %land_consistent)` dashboard row. Track over training. Any dip is an immediate signal to investigate.

### 8.4 Sample-quality protocol (per major checkpoint)

1. Generate 16 samples for 10 fixed prompts (representative: dense European, suburban American, tropical, Scandinavian small town, Asian commercial, etc.).
2. Render to map images with consistent style (lonboard / deck.gl).
3. Visual review by PI.
4. LLM judge run in batch.

Cheap to run, deeply informative.

## 9. Training infrastructure & phased plan

### 9.1 Compute & resource map

| Resource | Where | Cost |
|---|---|---|
| Code development | Local Mac | Free |
| Data prep & rasterisation | Leonardo `lrd_all_serial` (4 cores / 30 GB / 4 h walltime) | **Free** (budget-exempt) |
| VAE training | Leonardo `boost_usr_prod` (1× A100) | Billed |
| Stage A / Stage B training | Leonardo `boost_usr_prod` (4× A100) | Billed |
| Eval | `boost_usr_prod` (1× A100) for sample-heavy eval; `lrd_all_serial` for metric computation | Mostly free |
| Storage | `$WORK` (1 TB), `$FAST` (1 TB NVMe), `$CINECA_SCRATCH` (~20 TB, 40-day TTL) | Free |

### 9.2 Phased plan

#### Phase 0 — Local development & code scaffolding (1.5 weeks, 0 GPU-h)

Build on Mac and free Leonardo partition:
- PyTorch Lightning + WebDataset + diffusers / transformers libraries.
- Data prep pipeline: tile sampling, rasterisation, vector tokenisation.
- Model definitions: VAE, DiT, raster CNN encoder, AR transformer.
- Eval harness: all metric implementations, baseline computation.
- Slurm job templates: `data_prep.sbatch`, `vae_train.sbatch`, `stage_a_train.sbatch`, `stage_b_train.sbatch`, `eval.sbatch`.
- End-to-end test on 100 synthetic tiles.

#### Phase 0.5 — Real data prep on Leonardo (3 days, 0 GPU-h)

On `lrd_all_serial` (free):
- Generate Sweden + Singapore + Sri Lanka tile shards (~3,000-5,000 tiles).
- Validate raster encoding against ground-truth visualisations.
- Validate vector tokenisation by round-tripping decode → re-rasterise → IoU check.

#### Phase 1 — De-risking experiments (~250 GPU-h, 2 weeks wall)

Run Experiments 0–4 sequentially (see §10). Decision gate at end:
- All four green → proceed to production.
- Any yellow → diagnose, fix, re-run.
- Any red → pause, return to design.

#### Phase 2 — Production data prep (~3 days, 0 GPU-h)

On `lrd_all_serial` (free):
- Generate **Wave 1 corpus** (~150k Western Europe tiles).
- Build manifest, compute global stats.
- Quality-filter pass.
- (Wave 2 — planet-scale, ~500k tiles — is generated later, only after Phase 6 results justify scaling and a budget extension is granted.)

#### Phase 3 — Production VAE training (~50 GPU-h, 1–2 days)

Single-GPU job on `boost_usr_prod`. Reconstruction loss, channel-aware. Freeze before Stage A starts.

#### Phase 4 — Production Stage A training (~800 GPU-h, ~2–3 weeks wall)

4× A100 chained 24-h jobs on `boost_usr_prod`. Phased schedule (unconditional → text/tags → ControlNet). Checkpoint every 30 min.

#### Phase 5 — Production Stage B training (~2,500 GPU-h, ~3–4 weeks wall)

4× A100 chained 24-h jobs. Cross-attention to Stage A's raster encoder. Noise-augmented training throughout. Checkpoint every 30 min.

#### Phase 6 — Eval, ablations, paper (~500 GPU-h, ~2 weeks)

- Full eval suite on a 5,000-tile held-out set.
- Ablations: tile size (2 km² vs 1 km²), conditioning modes (text only / tags only / full), model size (300 M vs 500 M Sketcher).
- Demo sample generation for paper figures and customer pitches.
- Write paper.

#### Phase 7 — Hand-off / publication (no compute)

- Open weights to Hugging Face.
- Paper submission.
- Initial customer pilot conversations.

### 9.3 Slurm job patterns

#### Free CPU job (data prep, eval metrics)
```bash
#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

Chain via `--dependency=afterok` for runs >4 h.

#### GPU training job
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

Chain via `--dependency=afterok` for multi-day training. Save checkpoints to `$WORK` with a strict naming convention (`stage_a_v1_step_{N}.pt`); resume on preemption.

### 9.4 Compute budget tracking

| Phase | Estimated GPU-h | Cumulative |
|---|---:|---:|
| 0 — Code scaffolding | 0 | 0 |
| 0.5 — Data prep dry-run | 0 | 0 |
| 1 — De-risking experiments | 250 | 250 |
| 2 — Production data prep | 0 | 250 |
| 3 — VAE production | 50 | 300 |
| 4 — Stage A production | 800 | 1,100 |
| 5 — Stage B production | 2,500 | 3,600 |
| 6 — Eval + ablations | 500 | 4,100 |
| Buffer / retries | 700 | **4,800** |

**Initial allowance:** 1,250 GPU-h (covers Phases 0–4 with ~150 GPU-h headroom — i.e., reaches the end of Stage A production training on Wave 1 data).
**Production extension request to CINECA:** ~3,500 additional GPU-h after Phase 1 (de-risking) shows green and we want to complete Phase 5 (Stage B production) at full Wave 1 quality.
**Fallback if extension delayed/denied:** stop after Phase 4 with a Stage-A-only artifact (still demoable as raster generation), or run Phase 5 with reduced Stage B model size on Wave 1 data. **Never** drop Wave 1 to single-country only — that's research, not a v1 result.

### 9.5 Distribution

- **Open weights** to Hugging Face under a permissive research license.
- **Commercial license** layered on top for AV / game studio customers.
- **Paper** submitted to NeurIPS / CVPR / ICLR (whichever submission window best fits Phase 6 completion).

## 10. De-risking experiments

Four sequential experiments on the Sweden + Singapore + Sri Lanka tile dataset, total ~250 GPU-h, ~2 weeks wall time.

### Experiment 0 — Architecture smoke test on synthetic data
- **Risk addressed:** fundamental architecture / code bugs.
- **Setup:** ~5k synthetic tiles (procedural rectangles + grid roads), tiny Sketcher + Inker, 1 epoch each.
- **Cost:** ~10 GPU-h, ~2 days wall.
- **Go signal:** pipeline produces something coherent end-to-end.
- **No-go signal:** training diverges → code bug; fix and re-run.

### Experiment 1 — Sketcher on real data (Sweden + Singapore + Sri Lanka)
- **Risk addressed:** does Stage A learn coherent multi-channel structure from real OSM data, *across diverse climates and urban morphologies*?
- **Setup:** ~3,000 tiles drawn from all three countries (balanced), Stage A DiT ~200 M params (smaller than production for speed).
- **Cost:** ~80 GPU-h, ~4 days wall.
- **Go signals (all should pass):**
  - Per-channel IoU within target ranges.
  - Visual: 7/10 sampled tiles "city-shaped" to a human.
  - FID below random-crop baseline.
  - Conditional samples differ from unconditional in expected directions.
- **No-go signals:** disconnected road blobs, no learned road/building exclusion, conditioning ineffective.
- **Mitigation if fails:** simplify to 5-channel skeleton; redesign conditioning encoder.

### Experiment 2 — Inker on perfect input (Sweden + Singapore + Sri Lanka)
- **Risk addressed:** can Stage B output geometrically valid GeoJSON?
- **Setup:** ~3,000 ground-truth (raster, tokens) pairs from all three countries, Stage B ~300 M params, train on ground-truth raster only.
- **Cost:** ~120 GPU-h, ~5 days wall.
- **Go signals (all should pass):**
  - Building Chamfer < 5 m average.
  - ≥ 90% non-self-intersecting buildings.
  - ≥ 85% single-component road graphs.
  - POIs within reasonable proximity of ground-truth same-category POIs.
- **No-go signals:** self-intersection > 30%, topology random, POI placement uniform.
- **Mitigation if fails:** add explicit topology tokens; batched polygon decoder; structured constraints harder.

### Experiment 3 — End-to-end domain gap
- **Risk addressed:** how badly does quality drop with sampled (vs ground-truth) raster?
- **Setup:** combine Exp 1 + Exp 2 trained models, 200 samples.
- **Cost:** ~30 GPU-h, ~1 day wall.
- **Go signals:**
  - Validity drops ≤ 20% vs Exp 2.
  - End-to-end FID within 2× of Stage A solo.
- **No-go signals:** validity collapses; Inker hallucinates beyond Sketcher.
- **Mitigation if fails:** crank up training-time raster noise; schedule-sampling; short joint fine-tune.

### Experiment 4 — Tile stitching
- **Risk addressed:** do adjacent tiles connect coherently?
- **Setup:** generate 2 adjacent tiles same prompt, inspect cross-boundary continuity.
- **Cost:** ~10 GPU-h, ~1 day wall.
- **Go signals:** ≥ 60% road continuity across edge; seams not obvious to human.
- **No-go signals:** < 20% continuity; tiles internally coherent but unrelated.
- **Mitigation if fails:** add cross-tile context conditioning at sample time (the second tile sees the first as additional ControlNet input).

## 11. Risks & mitigations

### High-impact risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Geometric validity remains stubbornly low | Medium | High | Constrained decoding; explicit topology tokens; if stuck, post-hoc validity correction step |
| Distribution shift across regions ("Berlin great, Lagos broken") | High | Medium | Stratified sampling by country / climate / density; per-region eval breakdowns |
| Tile-stitching produces visible seams in larger generations | High | Medium | Outpainting at Stage A; cross-tile context conditioning; overlapping tile + vector merge at Stage B |
| Compute extension denied | Medium | Medium | Fallback: Phase 1 (Western Europe) only is publishable; scope down geographically not architecturally |
| Stage A learns to generate rasters Stage B can't trace | Low | High | Domain-gap mitigations (§7.2); if persistent, joint fine-tune as v1.1 |

### Lower-impact risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Building label gap (94% null) limits POI realism | Medium | Low-Medium | OSM Bridge File enrichment doubles labelled coverage |
| FlashAttention 3 unavailable on Leonardo's PyTorch build | Low | Medium | Fall back to FlashAttention 2 (still enables 16k context with margin) |
| WebDataset shard format issues at scale | Low | Low | Standard format; well-tested |

## 12. Success criteria

### Minimum viable (v1.0)

These are the **shippable** thresholds — lower than the eval-target numbers in §8.1 (which are aspirational). v1 ships if all of these clear:

- Stage A: produces visually plausible rasters with non-trivial conditioning fidelity (≥ 25% fool rate on real-vs-fake classifier).
- Stage B: produces valid GeoJSON with ≥ 90% non-self-intersecting buildings, ≥ 85% single-component road graphs (vs §8.1 aspirational targets of 95% and 90%).
- End-to-end: at least 50% of generated 4 km² tiles are "obviously plausible" to a human reviewer in side-by-side with real tiles.
- Reproducible, documented code, weights uploaded to HuggingFace.

### Stretch (v1.1)

- Cross-tile stitching produces 16 km² regions with visible coherence.
- Conditioning beyond text: terrain DEM, coastline mask, style anchor, constraint mask all functional.
- Per-region (country / climate) eval breakdowns within acceptable variance.
- Pilot conversation with at least one AV / game studio customer.

### Stretch (v2 — out of scope for this spec)

- Joint Stage A + B fine-tune.
- 3D extrusion to glTF / OpenDRIVE.
- Region-conditional generation that geographically pins to a real area.
- Lane-level road detail.
- Higher-resolution tiles (8 km²+).

## 13. Open questions

- **Text encoder choice.** Specific 2026 CLIP-class model (OpenCLIP-2026 vs newer alternatives) — defer to implementation.
- **CNN encoder for Stage B raster ingest.** Exact architecture (ResNet vs ViT-style patch encoder) — defer.
- **EasyControl LoRA rank.** Hyperparameter; sweep during Phase 1 ablations.
- **Hyperparameters in general.** Learning rates, batch sizes, augmentation σ ranges — sweep during Phase 1.
- **Tile-stitching strategy.** Outpainting in Stage A vs vector-merge at Stage B vs both — decided after Experiment 4.
- **Phase 2 geographic scope.** Western Europe vs planet-scale — decided after de-risking results + budget extension outcome.
- **Closed vs open POIs.** FSQ keeps 7.5 M historical closures; default v1 drops them, v1.1 might use them as a temporal-generative signal.

## 14. References & related work

### Methodologies we build on

- **DiT — Diffusion Transformers.** [Peebles & Xie, 2022; ICLR Blogposts 2026 evolution review](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)
- **EasyControl / OmniControl.** [arXiv 2503.07027](https://arxiv.org/abs/2503.07027) — DiT-native conditioning replacing classic ControlNet.
- **Pix2Poly.** [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Adimoolam_Pix2Poly_A_Sequence_Prediction_Method_for_End-to-End_Polygonal_Building_Footprint_WACV_2025_paper.pdf) — sequence prediction for polygonal buildings; our Stage B token scheme is a direct extension.
- **PolyGen.** Nash et al., 2020 — autoregressive vertex tokenisation for 3D meshes; conceptual ancestor of our Inker.
- **EDM noise schedule.** Karras et al., 2022 — current best practice for diffusion training.
- **WebDataset.** PyTorch-native shard format for large-scale streaming.

### Concurrent / adjacent work (differentiation)

- **MapDiffusion.** [arXiv 2507.21423](https://arxiv.org/abs/2507.21423) — vectorised HD-map *construction* from sensor data (online perception). We do offline novel-city *generation* from text.
- **Sat2City.** [arXiv 2507.04403](https://arxiv.org/abs/2507.04403) — cascaded latent diffusion → 3D voxel city from satellite. We take text → 2D vector.
- **CityDreamer4D.** [arXiv 2501.08983](https://arxiv.org/html/2501.08983v1) — compositional 3D city, NeRF-based. We emit 2D vector usable in GIS / sim tools.
- **SLEDGE.** [arXiv 2403.17933](https://arxiv.org/html/2403.17933) — driving environments via DiT + raster latents. We emit *vector* (directly editable, OpenDRIVE-compatible).
- **Generative AI for Urban Design.** [arXiv 2505.24260](https://arxiv.org/html/2505.24260) — stepwise ControlNet pipeline, human-in-the-loop. We aim for fully automatic generation for sim / games.

**Our defensible position:** the only system producing multi-layer, semantically labelled, 2-D vector cities from a text prompt, ready to drop into a simulator. Narrow enough for a paper; broad enough for a commercial tail.

## 15. Glossary

- **Tile.** A 2.048 km × 2.048 km square of map data; one training example.
- **Raster channels.** 9 layers — roads × 4 classes, buildings × 2, water, green, urban — each a 512 × 512 binary or continuous mask.
- **Vector tokens.** Discrete tokens encoding geometry primitives, coordinates, and attributes; the unit Stage B emits.
- **Tile-local coordinates.** Coordinates inside `[0, 2048)` m relative to tile origin, quantised to one of 512 discrete bins per axis.
- **Stage A / The Sketcher.** Latent diffusion model. Raster in, raster out.
- **Stage B / The Inker.** Autoregressive transformer. Raster in, vector tokens out.
- **DiT.** Diffusion Transformer; the transformer-based replacement for U-Net in latent diffusion.
- **EasyControl.** DiT-native LoRA-based conditioning, replacing classic ControlNet.
- **Constrained decoding.** Logit masking at sampling time that enforces structural validity (paired coordinates, closed polygons, layer order).
- **Domain gap.** The mismatch between Stage B's clean training input and noisy inference input from Stage A.
- **WebDataset.** PyTorch-native tar-shard format for streaming training data.

---

## Appendix A — Architectures considered and rejected

| Architecture | Rejected because |
|---|---|
| Pure autoregressive vector tokeniser (PolyGen-for-cities) | Sequence length 10 k–100 k tokens too aggressive for v1; long-context research risk |
| Layer-cascaded (graph → polygons → point process) | Three training pipelines triple the engineering surface; cascading errors |
| Pure raster diffusion + post-hoc vectorisation | Vector quality bounded by raster resolution; semantic labels hard to recover from pixels |
| Joint stage training end-to-end | Increases compute > 2×; harder to debug; deferred to v1.1 |
| Per-region specialised models | Loses the "global vocabulary" leverage; multiplies training cost |
| 1.024 km² tiles at 256×256 px | Too small for sim-customer demo; more stitching seams; smaller per-tile context |
| Putting POIs in Stage A's raster channels | Pushes vocab to ~20 channels; better as fine semantics in Stage B |

## Appendix B — Compute budget table

(See §9.4 for the same table integrated with the phased plan.)

| Phase | GPU-h |
|---|---:|
| 0 — Code scaffolding | 0 |
| 0.5 — Data prep dry-run | 0 |
| 1 — De-risking experiments | 250 |
| 2 — Production data prep | 0 |
| 3 — VAE production | 50 |
| 4 — Stage A production | 800 |
| 5 — Stage B production | 2,500 |
| 6 — Eval + ablations | 500 |
| Buffer / retries | 700 |
| **Total estimated** | **4,800** |

Initial allowance 1,250 covers Phases 0–4. Extension request to CINECA after Phase 1 de-risking shows green.

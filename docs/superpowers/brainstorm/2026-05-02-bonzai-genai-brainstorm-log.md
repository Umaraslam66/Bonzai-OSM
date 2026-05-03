# Bonzai-OSM generative city model — brainstorm log

> **Started:** 2026-05-02. Running log of the architecture brainstorming session.
> Every section reflects what was discussed in chat. Open questions are flagged.
> The eventual *spec* (a separate, polished document) will be written once the
> open questions close out.

---

## 1. The problem (verbatim from the kickoff)

> We want to build a Generative AI model capable of creating realistic, structurally sound, and novel city infrastructure (roads, buildings, POIs, land use).
>
> **Input:** A massive dataset (OpenStreetMap or Overture) containing the physical geometry (polygons, lines, points) and metadata (building type, road speed limit, business category) of existing global cities.
>
> **Desired output:** A trained generative model that can output completely new, logical map geometry and zoning from scratch. For example, prompted with "generate a dense European commercial district," it should output coordinates and attributes for logical road networks and building footprints that look and function like a real city.
>
> **Core challenge:** Standard generative AI models (LLMs) read and write 1-D discrete sequences from left to right. Maps are 2-D, continuous, and rely on absolute global coordinates. Feeding raw global coordinates into a standard AI explodes the vocabulary, breaches memory limits, and fails to learn spatial relationships.

The brainstorming question: how to bridge this gap — what methodologies, data
transformations, and architectural designs translate 2-D spatial map data
into a format that a generative AI can efficiently learn from.

## 2. Constraints and ambient context

These shape what's buildable, but not the architecture itself.

| Item | Value |
|---|---|
| Compute allocation | EuroHPC `AIFAC_P02_222` on CINECA Leonardo |
| Budget | 40,000 core-hours total = ~1,250 Booster GPU-node-h = ~5,000 single-A100-h |
| Window | 2026-03-11 → 2026-06-11 (3 months); ~57 days remaining as of 2026-05-02 |
| Free CPU partition | `lrd_all_serial` (4 cores / 30 GB / 4 h walltime, unbilled) |
| Billed GPU partition | `boost_usr_prod` (32 cores / 512 GB / 4× A100 / 24 h) |
| Storage | `$WORK` 1 TB, `$FAST` 1 TB NVMe, `$CINECA_SCRATCH` ~20 TB (40-day rolling auto-delete) |
| Data on hand | Planet OSM PBF, Overture 2026-04-15.0 (75.5 M places, 2.5 B buildings, 344 M road segments), Foursquare 99.9 M places |

**Decision rule we adopted:** preprocessing and data prep go to `lrd_all_serial`
(free) or local Mac. Booster billing is reserved for actual model training.

## 3. The three candidate architectures considered

### 3.1 Two-stage hybrid: raster diffusion → AR vectoriser  ★ CHOSEN

**Stage A** is a small (~80 M-param) latent diffusion model on a multi-channel
raster tile. Each channel encodes one map layer (road density by class, building
footprint mask, building density, water, green, urban). Outputs a coarse
"semantic skeleton" of the tile. Conditioning is text/region embedding plus
optional ControlNet-style spatial conditioning (terrain, coastline, partial
mask).

**Stage B** is a small (~150 M-param) decoder-only transformer that reads
Stage A's raster (cross-attention to a CNN encoder) and autoregressively
emits vector geometry as a token sequence: `(primitive_type, dx_quantised, dy_quantised, attribute_id)` tuples in deterministic raster-scan order.

**Viability: HIGH.** Both halves have huge precedent (latent diffusion +
ControlNet for stage A; raster-to-vector tracing — RoadTracer, Sat2Graph,
deep-vector-graphics work — for stage B). Compute fit: ~500–1,000 GPU-h
total. Fits inside 1,250 Booster node-h budget with retry headroom.

### 3.2 Pure autoregressive vector tokeniser (PolyGen-for-cities)

One transformer end-to-end. Each tile is a sequence of vector tokens in
Hilbert-curve order with quantised, tile-relative coordinates. PolyGen for
3-D meshes is the closest analogue.

**Viability: MEDIUM.** Cleanest model; sequence length is the killer
(10 k–100 k tokens for 1 km²). Long-context tricks needed (sparse attention,
hierarchical, byte-latent-style chunking). Higher research risk for v1.

### 3.3 Layer-cascaded: graph → polygons → point process

Three models. A graph transformer emits the road network. A polygon
diffusion model emits building footprints conditioned on the road graph.
A labelled point-process model emits POIs conditioned on both layers.

**Viability: MEDIUM.** Each model is small and well-targeted. Downside:
three training pipelines, three checkpointing stories, errors compound.

### Why #1 was chosen

- Vector output preserved.
- Conditioning is "free" (Stage A behaves like Stable Diffusion).
- Debuggable layer by layer.
- Failure modes are interpretable (raster errors visible immediately).
- Naturally supports the simpler v1 scope (roads-only) as a 1-channel
  ablation of Stage A.
- Compute fit is comfortable.

## 4. Top-level system architecture (Section 1 of 7)

Four units, each independently buildable and testable:

```
[1] DATA PREP            [2] STAGE A              [3] STAGE B            [4] EVAL
    (offline,                (latent                  (AR                    (offline,
     free CPU)                diffusion)               vectoriser)            free CPU)

OSM/Overture/FSQ  ─►  Multi-channel raster  ─►  Vector geometry  ─►  Metrics + samples
                       (the "skeleton")          (roads, buildings,
                                                  POIs, land)
```

**[1] Data prep.** Off-cluster (Mac) and `lrd_all_serial` (free). Slices the
world into 1.024 km² tiles (256 px × 4 m/px). Per tile, emits two artefacts:
- **Raster** — a 9-channel tensor (256×256×9), the "skeleton view."
- **Vector tokens** — sequence of `(primitive_type, dx_q, dy_q, attribute_id)`
  encoding the tile's exact geometry, quantised to 256 bins per axis.

**[2] Stage A — latent diffusion.** ~80 M-param U-Net. VAE encodes the
9-channel raster to a 64×64×4 latent. EDM noise schedule. Cross-attention
conditioning on text/region embeddings; optional ControlNet for spatial
conditioning (terrain, coastline, masks).

**[3] Stage B — autoregressive vectoriser.** ~150 M-param decoder-only
transformer. CNN encoder turns Stage A's raster into 16×16×D feature map
(cross-attention keys/values). Decoder emits ~1.5 k–7 k tokens per tile.

**[4] Eval harness.** Three metric families:
- Stage A: per-channel IoU, raster FID-like score.
- Stage B: Chamfer distance on polygons, road-graph topology stats,
  building IoU.
- End-to-end: tile-statistics JS-divergence, real-vs-fake CNN classifier
  (fool rate), geometric validity (planarity, self-intersection rate,
  road connectivity %).

### Three load-bearing splits

1. **Local coordinates inside every tile.** Always 0–1024 m, quantised to
   256 bins. The model never sees a global lat/lon. Vocabulary stays
   bounded. "Where on Earth" is handled by Stage A's prompt conditioning,
   not by the geometry tokens.

2. **Raster vs vector responsibility.** Stage A handles structural
   skeleton. Stage B handles fine-grained semantics (POI categories,
   building subtypes, exact polygon vertices). Keeps Stage A small and
   concentrates the rich vocab in Stage B.

3. **Stages trained independently in v1.** Joint fine-tuning is a v2
   move. Domain-gap risk (clean ground-truth raster vs noisy sampled
   raster at inference) mitigated by augmentation in Stage B training.

## 5. Business value and target user persona  *(added 2026-05-02)*

The user asked: "what is the business value of it and who would be the
user?" — because the persona shapes input format, output format, and
quality bar.

### Plausible buyer segments

| Persona | Pain | Quality bar | Output format | WTP |
|---|---|---|---|---|
| Game / VR studios (open-world, MMOs) | Hand-authoring city geometry takes huge artist hours; existing procedural tools (CityEngine, Houdini) need expert config | Stylistic plausibility; no real-world accuracy | FBX / glTF / GeoJSON → 3-D extrusion | Medium-high |
| AV / robotics simulation | HD maps cost ~$5 k/km; can't test rare layouts without owning them | Realistic road topology; building occlusion matters; POIs less critical | OpenDRIVE / ASAM road graph + 3-D city | **Very high** |
| Urban planners / architects | Slow design iteration; bespoke synthetic data | Structurally realistic; counterfactual scenarios | GeoJSON / Shapefile / IFC | High |
| Geospatial AI / data augmentation | Labelled training data scarce, esp. for rare regions | Distribution-faithful, label-correct | GeoTIFF + vector labels | Medium |
| Academic research | No 2-D-aware generative baseline | Novelty + benchmarks | Whatever | Zero ($, but prestige) |
| Mapping / GIS infill | Sparse coverage in developing countries | Must overlay cleanly on real basemaps | OSM / GeoJSON | Medium |
| Defence / national security | Restricted maps of contested regions; need synthetic plausible cities for war-gaming | Geometric realism + regional flavour | Controlled | Very high |
| Real estate / proptech | Bespoke scenario tools for "what if X is built" | Geographically pinned to real area | 3-D + GeoJSON | Medium |

### v1 decision (locked 2026-05-02): **AV / robotics simulation engineer** is the primary persona.

The PI accepted the recommendation as the easiest persona to visualise and the
most exciting starting point. Treated as a novel R&D bet — additional use
cases will emerge along the way. Output is GeoJSON-native; OpenDRIVE export
is a downstream conversion. Game studios are a close-adjacent secondary
buyer with the same quality bar.

### v1 recommendation reasoning **(was: AV / game simulation engineer)**

Reasoning:
1. **Funded buyers.** AV companies and game studios both have real
   procurement budgets and track record of paying for sim-data.
2. **Aligned quality bar.** Both want plausibility, not pixel-perfect
   real-world fidelity — exactly what a generative model produces well.
3. **Vector output is the natural fit.** GeoJSON / OpenDRIVE / glTF are all
   vector-native. Our Stage B output decodes directly to GeoJSON.
4. **Latency is batch.** No real-time constraint; we can afford 50–250
   diffusion steps and beam search at Stage B for quality headroom.
5. **No regulatory hurdles.** Unlike urban planning (zoning approvals)
   or defence (export control), the sim use-case is commercial-friendly.
6. **Academic paper writes itself in parallel.** No conflict between
   "sim-data product" and "open-source weights + benchmark" distribution
   — they reinforce each other.

### How this reshapes the design

1. **Output format pinned to GeoJSON-native** with optional 2.5-D height
   per building (Overture has heights). Stage B's token decoder produces
   GeoJSON FeatureCollections directly.
2. **Conditioning richer than text.** A sim engineer's user inputs:
   - Text prompt ("dense Asian commercial, 2 km²")
   - Optional terrain raster (DEM)
   - Optional coastline / boundary mask
   - Optional "style anchor" — point to a real region whose vibe to copy
   - Optional "constraint mask" — pin certain pixels (e.g., "this 100 m square must be a park")
   ControlNet handles all of these uniformly through Stage A.
3. **Tile stitching becomes first-class.** Sim users want >1 km² output
   (a whole town). Two options: outpainting in Stage A (well-precedented),
   or generating overlapping tiles + merging at the vector layer (Stage B
   dedupes overlapping primitives). Decision deferred.
4. **Latency is batch-friendly.** No interactive constraint. Quality
   headroom comes from beam search and multi-sample voting at sample time.
5. **Distribution path: open weights + commercial license.** Hugging Face
   for academic credibility; commercial API on top for AV / game customers.

### Secondary buyers worth keeping in scope

- Academic publication: zero-cost, parallel.
- Mapping/GIS infill: easy add-on (the "region completer" use-case is
  the same as inpainting in Stage A).
- Geospatial data augmentation: easy add-on (just ship the raw outputs).

### Buyers we explicitly defer

- Urban planning: regulatory plausibility raises the quality bar
  significantly — defer to v2 or v3.
- Defence: contractual complexity high; revisit only after commercial
  traction.
- Real estate proptech: niche; depends on geographic-pinning quality.

## 5b. Potential assessment and pitches  *(added 2026-05-02)*

### Honest confidence levels

- **~70 %** this produces a publishable research artifact within the
  remaining ~57-day, ~1,250 GPU-h budget (NeurIPS / CVPR / ICLR scope).
- **~40 %** that v1 alone gets paid integration with an AV / game studio.
  Real procurement is slow; pilots take months.
- **~90 %** that some version of this becomes a real product within 3–5
  years. The gap is real and the underlying tech is converging.

Framing: a research-grade bet with a strong commercial tail. The
academic deliverable is the safer first milestone; commercial validation
comes after the paper.

### Why it's defensible

1. **The gap is empty.** Nobody ships "type a description → GeoJSON of a
   plausible novel city." Stable Diffusion gives raster imagery;
   CityEngine and Houdini give artist-configured procedural rules;
   satellite extraction tools give geometry of real places only.
2. **The data didn't exist 24 months ago.** Overture's 2024–2026 releases
   are the first globally consistent, license-clean, joinable vector
   dataset at this scale.
3. **The recipe is converging.** Latent diffusion + ControlNet + vector
   tokenisation + long-context transformers all matured to the point
   where the engineering side is solid; the novelty is in raster-to-vector
   decoding for *labelled* primitives — exactly the publishable contribution.

### Three risks named honestly

1. **Geometric validity is unsolved.** Plan for a 5–15 % validity
   post-processing rate as part of the system, not a bug.
2. **Distribution shift across regions.** "Great in Berlin, broken in
   Lagos" is the failure mode customers won't tolerate. Per-region
   balancing during training is critical.
3. **Tile stitching for >1 km² coherence** is its own research thread.
   Outpainting is the standard answer but seams will need attention.

### Pitches at three audience levels

**Layperson (dinner-table version):**
> "We're teaching a computer what real cities look like — by showing it
> millions of them — so it can invent new ones. You type 'a dense
> European commercial district,' and it draws you a plausible map:
> streets in the right places, building footprints, shops, parks. Not a
> picture of a map, but the actual data a video game or a self-driving
> simulator can use."

**CTO / journalist (canonical pitch):**
> "Imagine Stable Diffusion, but for **maps** instead of images. SD
> generates pictures from a prompt; ours generates the underlying
> vector data of cities — roads, buildings, businesses — from a prompt.
> Self-driving companies pay around $5,000 per kilometre to map real
> cities and only have a few; we generate unlimited plausible ones on
> demand. Game studios pay artists for months to handcraft one city;
> we generate one in seconds."

**ML reviewer (NeurIPS / CVPR version):**
> "We jointly train (i) a small multi-channel latent diffusion model on
> rasterised OSM/Overture tiles, and (ii) an autoregressive transformer
> that vectorises the diffusion's output into typed primitives (roads,
> buildings, POIs, land polygons) with tile-local quantised coordinates.
> Contribution: a tractable raster-to-vector decoding scheme for
> semantically labelled geometry at city scale, with ControlNet-style
> spatial conditioning on terrain, region embedding, and constraint
> masks."

## 5c. 2026 state-of-the-art update  *(added 2026-05-02 after web search)*

The search surfaced three architectural updates and four pieces of
concurrent work to position against.

### Architectural updates

1. **Stage A: U-Net → DiT (Diffusion Transformer).** By 2026 DiT has
   largely replaced U-Net for new latent diffusion projects. Pure-transformer
   architecture, better scaling, simpler code, native long-range attention.
   Same parameter count, better quality. We use a small DiT (~80-120 M
   params) for Stage A.
   - Reference: [ICLR Blogposts 2026 — "From U-Nets to DiTs"](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)

2. **Stage A: ControlNet → EasyControl-style LoRA injection.** ControlNet's
   encoder-side bypass was designed around U-Net. For DiT, 2025-2026
   successors use lightweight LoRA modules that inject conditioning tokens
   without modifying base weights. Cheaper, more flexible.
   - Reference: [EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer (2503.07027)](https://arxiv.org/abs/2503.07027)

3. **Stage B: borrow Pix2Poly's tokenisation directly.** Pix2Poly (WACV
   2025) is doing exactly what we proposed at the single-building level —
   quantise vertex coordinates to discrete tokens, sequence-predict
   polygons from raster input. Validated technique. We extend it to
   multi-layer (roads + buildings + POIs + land) with shared tile-local
   vocabulary.
   - Reference: [Pix2Poly: A Sequence Prediction Method for End-to-end Polygonal Building Footprint Extraction (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Adimoolam_Pix2Poly_A_Sequence_Prediction_Method_for_End-to-End_Polygonal_Building_Footprint_WACV_2025_paper.pdf)

### Concurrent work and differentiation

| Work | What they do | How we differ |
|---|---|---|
| [MapDiffusion (2025)](https://arxiv.org/abs/2507.21423) | Vectorised HD-map construction from sensor data (online perception for AV) | We do offline novel-city *generation* from text prompt |
| [Sat2City (2025)](https://arxiv.org/abs/2507.04403) | Cascaded latent diffusion → 3D voxel city from satellite | We take text → 2D vector geometry |
| [CityDreamer4D (2025)](https://arxiv.org/html/2501.08983v1) | Compositional 3D city, NeRF-based, unbounded | We emit 2D vector that drops into GIS / sim tools |
| [SLEDGE (2024)](https://arxiv.org/html/2403.17933) | Driving env via DiT + raster latent | We emit *vector* (directly editable, OpenDRIVE-compatible) |
| [Generative AI for Urban Design (2025)](https://arxiv.org/html/2505.24260) | Stepwise ControlNet pipeline for urban form, human-in-the-loop | We aim for fully-automatic generation for sim/games |

**Our defensible position:** the only system producing multi-layer,
semantically labelled, 2-D vector cities from a text prompt, ready to
drop into a simulator. Narrow enough to ship a paper, broad enough to
have a commercial tail.

### Simple-words explanation: The Sketcher and the Inker

**Stage 1 — The Sketcher (~80M-param DiT).** Studied millions of real
cities. Given a prompt ("dense European commercial, 1 km²"), produces
a coarse 9-channel coloured sketch — different colours for roads,
buildings, water, parks. Quick, fuzzy, shows *where* things go, not
*what* they are exactly. Output: a 256×256 image with 9 layered colours.

**Stage 2 — The Inker (~150M-param transformer).** Reads the Sketcher's
rough and carefully traces every shape with clean vector lines and
labels. Building blobs become precise polygons with `class=residential`.
Road blobs become a graph with class labels. POI dots become labelled
points with categories. Output: clean GeoJSON ready for any GIS / sim
tool.

**The one trick:** the Inker only sees coordinates inside one 1 km² tile,
quantised to 256 bins per axis. The vocabulary stays small (~2,400 tokens)
instead of exploding to billions of global lat/lon. "Where on Earth" and
"what's in a tile" are solved by two separate models.

## 5d. Data — what we feed the Sketcher and the Inker  *(added 2026-05-02)*

Headline: **we don't need imagery**. The world is already mapped in clean
precise vector form. We use it twice — once as the answer key for the
Inker, once as a homemade coloured sketch for the Sketcher (we render
the sketch ourselves from the same vector data).

### Three free vector data sources

| Source | What it's best at | Status |
|---|---|---|
| **OpenStreetMap (OSM)** | Raw global coverage, rich tags. ~85 GB planet PBF. | Already downloaded |
| **Overture Maps 2026-04-15.0** | Cleaned, license-clean, harmonised schema. 75 M places, 2.5 B buildings, 344 M road segments. | Already accessible via S3 |
| **Foursquare OS Places** | Deeper POI labels with hierarchical categories. ~100 M places. | Already downloaded |

**Primary skeleton:** Overture (roads, buildings, land use).
**POI semantics:** Foursquare hierarchical categories.
**Tag fallback:** OSM via Overture Bridge Files.

### One training example, concretely

A 1.024 km × 1.024 km square anywhere on Earth becomes one bundle:

1. **Pull vector ground truth** — query Overture/OSM for everything in
   the square.
2. **Convert to tile-local coordinates** — quantise to 256 bins per
   axis. The model never sees global lat/lon.
3. **Paint a 256×256×9 raster** ourselves from the vector data — one
   colour channel per layer (3 road classes + binary roads,
   buildings + density, water, green, urban).
4. **Serialise the same data as a token sequence** in stable order:
   land polygons → roads → buildings → POIs. ~3,000–5,000 tokens per
   tile.
5. **Attach metadata** — country, climate zone, density bucket,
   primary land use — as the prompt conditioning at training time.

Bundle on disk: `(raster.npz, tokens.txt, metadata.json)`.

### Design philosophy — quality over thrift  *(added 2026-05-03)*

The PI clarified that the **1,250 Booster GPU-h is the initial
allowance, not a cap**. CINECA grants extensions on justification, and
a strong v1 unlocks fundraising for additional compute. Therefore:

**Order of priorities:** (1) v1 quality, (2) methodological rigor,
(3) resource fit. **If compute binds, shrink the focus area** (Western
Europe instead of planet) **rather than the model**.

This shifts several earlier decisions upward in scale (see updated
tile-size and model-size sections below).

### Tile size — UPDATED to 2 km² (was 1.024 km²)  *(updated 2026-05-03)*

Decided 2026-05-02 after explicit calculations on five pressures.

**Configuration (updated 2026-05-03):**
- Tile side: **2048 m** (was 1024 m)
- Image dimension: **512 × 512 px** (was 256 × 256)
- Spatial resolution: **4 m/px** (unchanged — buildings still ~4 px each)
- Tile area: **4.2 km²** (was 1.05 km²)

Bigger tiles win when compute is flexible: more spatial context per
tile, fewer stitching seams, better customer demo (one tile = a real
town, not a piece of one). The previous 1.024 km² remains as the
fallback if a real compute wall hits.

**Five pressures and how they push:**

| Pressure | Wants | Direction |
|---|---|---|
| Buildings visible (≤4 m/px for 4 px/bldg) | Smaller tiles | ← |
| Inker sequence length (fit 8 k context) | Smaller tiles | ← |
| Compute budget (Stage B ≤700 GPU-h) | Smaller tiles | ← |
| Stitching seams (fewer edges per region) | Larger tiles | → |
| Training variety per region | Smaller tiles | ← |

Four of five push smaller. Stitching is the only force pushing bigger,
and it's solvable at sample time (outpainting / overlap), so it doesn't
constrain training.

**Compute estimate at 2 km² (updated 2026-05-03):**
- De-risking phase (Sweden + Singapore + Sri Lanka): ~250 GPU-h
- Phase 1 production (Western Europe, ~150k tiles): ~2,500 GPU-h
- Phase 2 production (planet-scale, ~500k tiles): ~5,000 GPU-h total
- Ablations + retries: ~800 GPU-h
- Estimated total for full production: ~4,800–6,000 GPU-h
- **Initial allowance covers de-risking comfortably; production needs extension**

**Sequence length envelope at 2 km²:**
- Typical urban tile: ~10,000 tokens
- Dense urban (Manhattan-like) tile: ~14,000 tokens
- Target context: 16k tokens (32k stretch)

**Updated model sizes (2026-05-03):**
- Sketcher (Stage A DiT): **~300-500 M params** (was ~80 M)
- Inker (Stage B transformer): **~500 M – 1 B params** (was ~150 M)
- Training: ~50-100 epochs with hyperparameter sweep (was ~30)

**Fallback to 0.5 km² triggered if:**
- Stage B small-country validation runs >50 GPU-h on 100 tiles (>10 % of estimated cost per 0.5 % of data)
- Densest 5 % of tiles consistently overflow 8 k context
- Validation shows Stage B fails to learn long-range token dependencies

POIs locked into the Inker's vocabulary (not in the Sketcher's raster
channels). The Sketcher learns coarse structure (roads + buildings + land
+ water + green); the Inker learns the fine semantic labels.

### Tile sampling — stratified, not uniform

Uniform sampling would over-weight US Midwest and ocean. We stratify by:

- **Country** — each country contributes proportional to a fixed budget,
  not population.
- **Density bucket** — rural / suburban / urban / dense urban,
  equal-weight.
- **Climate zone** — Köppen classification, equal-weight.

**Skip rules:** drop tiles with <20 features, >5,000 features, or
>50 % water.

The whole stratification is a DuckDB SQL query running free on
`lrd_all_serial`.

### Training-set size

| Item | Estimate |
|---|---|
| Tiles | ~500,000 |
| Raster per tile | ~10-50 KB compressed |
| Vector tokens per tile | ~6 KB |
| Metadata per tile | ~0.5 KB |
| Total per tile | ~50 KB |
| **Total training set** | **~25 GB** |

Easily fits in `$WORK` (1 TB) or `$FAST` (1 TB NVMe). For context,
Stable Diffusion v1 used ~240 TB of image-text pairs; we use ~4
orders of magnitude less because map data is far more structured than
natural imagery.

### On-disk format

WebDataset format (PyTorch-native shards):

```
$WORK/bonzai-tiles/
├── shard-00000.tar   ← 1k tiles, ~50 MB
├── shard-00001.tar
├── ...
├── shard-00499.tar   ← 500 shards, ~25 GB total
├── manifest.parquet  ← which-tile-where + stratification metadata
└── stats.json        ← global stats for normalisation
```

The whole pipeline that **builds** these shards runs free on
`lrd_all_serial`. Generating 500k tiles from Overture takes ~1-2 days
of wall time across multiple parallel free jobs. **Zero billed compute
spent on data prep.**

### The building-label gap

In Overture, ~94 % of building footprints have no class/subtype label.
Not our problem to solve — it's a known gap in the source data. For v1:

- **Default:** emit `BUILDING_OPEN class=UNKNOWN` for unlabelled
  cases. The model learns class only when it's there.
- **Optional enrichment via OSM Bridge Files:** join Overture buildings
  to their original OSM IDs, then to OSM tags (`building=*`, `shop=*`,
  `amenity=*`). Roughly doubles labelled-building count. Free join on
  `lrd_all_serial`. Defer until after v1 trains.

## 5e2. Stage A details — The Sketcher  *(added 2026-05-03)*

### The 9 raster channels

| # | Channel | Encoding |
|---|---|---|
| 1 | All roads (any class) | Binary mask, 1-px line |
| 2 | Major roads (motorway/trunk/primary) | Binary mask, 2-px line |
| 3 | Mid-tier roads (secondary/tertiary) | Binary mask |
| 4 | Minor roads (residential/service) | Binary mask |
| 5 | Building footprints | Binary mask |
| 6 | Building density | Continuous 0-1 (Gaussian-blurred footprints) |
| 7 | Water | Binary mask |
| 8 | Green (parks/forests/fields) | Binary mask |
| 9 | Built urban land use | Binary mask |

### Pipeline

```
9-ch raster (512×512×9) → VAE encoder → latent (64×64×4) → DiT denoises
                                                        ↓
                                                latent → VAE decoder → 9-ch raster
```

8× spatial compression. VAE ~10 M params, trained first as a one-shot reconstruction job.

### DiT architecture
- ~300-500 M params. 24 transformer layers, hidden dim 1024, FFN ×4.
- Patch size 2 over 64×64 latent → ~1,024 transformer tokens.
- AdaLN-Zero conditioning. FlashAttention 3.

### Conditioning paths

| Type | What | How |
|---|---|---|
| Text | "dense Asian commercial, 2 km²" | CLIP text encoder → 768-dim → cross-attention |
| Region tags | country, climate, density-bucket, land-use | learned lookup table → summed conditioning token |
| ControlNet (EasyControl LoRA) | terrain DEM, coastline, style anchor, constraint mask | per-input LoRA adapters injecting conditioning tokens |

### Training and sampling
- EDM noise schedule, AdamW, cosine LR, 50-100 epochs.
- Classifier-free guidance: 10% null conditioning during training.
- Phased: 20 epochs unconditional → 30 epochs conditional → 20 epochs full ControlNet.
- Sampling: 50 DPM-Solver++ steps, CFG scale ~7.5.

## 5e3. Stage B details — The Inker  *(added 2026-05-03)*

### Vocabulary (~2,884 tokens)

| Family | Count | Examples |
|---|---|---|
| Structural | ~30 | `BOS`, `EOS`, `LAYER_LAND`, `BUILDING_OPEN`, `ROAD_EDGE`, ... |
| X-coordinates | 512 | `x_0` through `x_511` (4 m bin = 1 raster pixel) |
| Y-coordinates | 512 | `y_0` through `y_511` |
| Attributes | ~1,800 | `road_class=motorway`, `poi=cafe`, `building_class=residential`, ... |
| Heights / numeric | ~30 | `height_5m`, `height_10m`, ..., `height_NA` |

### Reading order within a tile
1. `LAYER_LAND` — water, parks, land-use polygons (largest features).
2. `LAYER_ROADS` — declare all road nodes, then describe edges by index.
3. `LAYER_BUILDINGS` — building polygons with class + height.
4. `LAYER_POIS` — labelled points conditional on everything above.

Within each layer: raster-scan order by primitive centroid.

### Cross-attention to the raster
- A small CNN encoder (~30 M params, 4-layer strided conv) compresses the 9-channel 512×512 raster to a 32×32×768 feature grid.
- Inker cross-attends to that grid at every layer. Reads the raster directly, not the VAE latent (preserves pixel-level detail).

### Constrained decoding (writes valid GeoJSON only)
- After an `x_*` token, the next must be a `y_*` token.
- After `BUILDING_OPEN`, only `BUILDING_CLOSE` ends the polygon, ≥3 coord pairs minimum.
- Self-intersection check on in-progress polygons; mask vertices that would create crossings.
- Layer order enforced (`LAYER_LAND` before `LAYER_ROADS` etc.).
- Road edge `nodes=[i,j]` indices masked to declared nodes only.

### Architecture
- 500 M – 1 B params. 24-32 layers, hidden 1024-1280, FFN ×4.
- Context: 16 k tokens (32 k stretch).
- RoPE, FlashAttention 3.
- Cross-entropy training, teacher-forcing.
- **Domain-gap mitigation:** add Gaussian noise (σ = 0.05–0.2) to the raster input during training to bridge clean (training) vs noisy (inference) inputs.

### Sampling
- Greedy default; beam search size 4 when quality matters.
- Constrained decoding active throughout.
- Typical 2 km² tile: 5–30 seconds end-to-end.

## 5e4. Eval harness  *(added 2026-05-03)*

### Three metric families

**Stage A:**
- Per-channel IoU vs ground-truth (target: roads ≥0.45, buildings ≥0.30, land-use ≥0.50).
- FID-like score on VAE latents.
- Conditioning fidelity (does "dense urban" generate denser tiles than "rural"?).

**Stage B:**
- Building Chamfer distance (target: <5 m).
- Building IoU (target: >0.50).
- Road graph: junction degree KL, connected-components match, edge-length match.
- POI placement: ≥60% within 50 m of a same-category ground-truth POI.
- **Geometric validity headline:**
  - ≥95% non-self-intersecting buildings
  - ≥90% road graphs single-component
  - <2% overlapping polygons
  - ≥80% POIs inside an appropriate land-use zone

**End-to-end:**
- Tile-statistics JS-divergence (target: most stats <0.1).
- Real-vs-fake CNN classifier — **fool rate** (target: ≥30% at v1).
- LLM-as-judge sample quality (target fool rate: ≥25%).
- Conditional generation correctness (human/LLM rates fit-to-prompt 1-5; target avg >3.5).

### Baselines (compute once, pre-training)
- Real-vs-real JS divergence (lower bound, the noise floor)
- Real-vs-uniform-random (upper bound, what bad looks like)
- Real-vs-Perlin-noise (structured-but-meaningless reference)
- Real-vs-CityEngine samples if we can obtain them

### Geometric validity sub-harness (per checkpoint)
1. Sample 500 tiles end-to-end.
2. Decode to GeoJSON.
3. Shapely `is_valid` per polygon.
4. NetworkX connectivity per road graph.
5. Pairwise overlap check on buildings.
6. Land-use consistency check (no buildings in water, etc.).

Output: dashboard row `(total, %valid_buildings, %valid_roads, %overlap_free, %land_consistent)`. Track over training. Dip = immediate investigate.

### Sample-quality protocol (per major checkpoint)
1. Generate 16 samples for 10 fixed prompts.
2. Render to map images (lonboard / deck.gl style).
3. Visual review by PI.
4. LLM judge run.

## 5e5. De-risking experiments  *(added 2026-05-03)*

Four sequential experiments on the Sweden + Singapore + Sri Lanka
de-risking tile dataset, each with a specific risk and a clear go/no-go
signal. **Total ~250 GPU-h, ~2 weeks wall time.** All run in the
de-risking phase before any production-scale compute.

### Experiment 0 — Smoke test on synthetic data
- **Risk:** fundamental architecture / code bugs
- **Setup:** ~5k synthetic tiles, tiny Sketcher + Inker, 1 epoch each
- **Cost:** ~10 GPU-h, ~2 days
- **Go:** pipeline produces something coherent end-to-end
- **No-go:** training diverges → code bug

### Experiment 1 — Sketcher on real data (Sweden + Singapore + Sri Lanka)
- **Risk:** does Stage A learn coherent multi-channel structure across diverse climates and urban morphologies?
- **Setup:** ~3k tiles balanced across all three countries, Stage A DiT ~200 M params
- **Cost:** ~80 GPU-h, ~4 days
- **Go:** roads in linear/branching patterns; conditional samples
  differ from unconditional; FID below random-crop baseline
- **No-go:** disconnected road blobs, no learned exclusion of
  buildings from roads, conditioning ineffective

### Experiment 2 — Inker on perfect input (Sweden + Singapore + Sri Lanka)
- **Risk:** can Stage B output geometrically valid GeoJSON?
- **Setup:** ~3k ground-truth (raster, tokens) pairs from all three countries,
  Stage B ~300 M params, train on ground-truth raster only
- **Cost:** ~120 GPU-h, ~5 days
- **Go:** building Chamfer dist <5 m; ≥90 % non-self-intersecting;
  ≥85 % single-connected road components; POIs in plausible clusters
- **No-go:** self-intersection >30 %; topology ignored; POI random

### Experiment 3 — End-to-end domain gap
- **Risk:** how badly does quality drop with sampled (vs ground-truth)
  raster as Inker input?
- **Setup:** combine Exp 1 + Exp 2 trained models, 200 samples
- **Cost:** ~30 GPU-h, ~1 day
- **Go:** validity drops ≤20 % vs Exp 2; FID within 2× of Stage A solo
- **No-go:** validity collapses; Inker hallucinates beyond Sketcher

### Experiment 4 — Tile stitching
- **Risk:** do adjacent tiles connect across boundaries?
- **Setup:** generate 2 adjacent tiles, inspect cross-boundary
- **Cost:** ~10 GPU-h, ~1 day
- **Go:** ≥60 % road continuity across edge; seams not obvious to
  human reviewer
- **No-go:** <20 % continuity; tiles unrelated to neighbours

### Mitigations if any experiment fails

| Experiment | If it fails |
|---|---|
| 0 | Code-level bug — fix and re-run |
| 1 | Channel layout wrong (try 5-channel) or conditioning encoder |
| 2 | Vector vocabulary needs topology tokens or batched polygon decoder |
| 3 | Add training-time raster noise; schedule-sampling; joint fine-tune |
| 4 | Add tile-context conditioning at sample time (cross-tile ControlNet input) |

## 6. Open questions (deferred decisions)

These are real choices we'll come back to. None of them block starting
the design — they're parameters of the chosen architecture.

- **Stage A channel layout.** 9 channels for the structural skeleton
  (roads × 4 classes, buildings × 2, water, green, urban) — locked.
- ~~**Tile size.**~~ **Resolved 2026-05-02:** 1.024 km² (256 px × 4 m/px),
  fallback to 0.5 km² if compute or sequence length forces it.
- **Conditioning richness.** Text only? Text + ControlNet inputs? Style
  anchor (CLIP-of-real-region embedding)?
- **Tile stitching strategy.** Outpainting at Stage A vs vector-merge at
  Stage B vs both.
- **Vector token vocabulary detail.** Coordinate quantisation (128 vs 256
  bins). Primitive type set (do we need a separate `BUILDING_OPEN` /
  `BUILDING_CLOSE` or just `POLYGON_BEGIN`?). How to encode multi-polygon
  buildings (donut holes, courtyards).
- **Sequence ordering inside a tile.** Land first → roads → buildings →
  POIs is the proposed order. Within layer: raster-scan of centroid?
  Hilbert? z-order?
- **Domain-gap mitigation between stages.** Augmentation in Stage B
  training (noise the ground-truth raster), schedule-sampling, or joint
  fine-tune at the end — pick one.
- **Eval baselines.** Real-tile vs real-tile JS divergence as the lower
  bound; real-tile vs random-tile as the upper bound. Need to compute
  these once before training.

## 7. What's left to design (the 7 sections)

1. ~~**Top-level system architecture**~~ ✅ — Section 4.
2. ~~**Stage A details**~~ ✅ — Section 5e2.
3. ~~**Stage B details**~~ ✅ — Section 5e3.
4. ~~**Data prep pipeline**~~ ✅ — Section 5d.
5. **Bridging the stages** — handling the train-time clean /
   inference-time noisy raster gap. *Mostly captured in Stage B's
   noise-augmentation training; could expand if needed.*
6. ~~**Evaluation harness**~~ ✅ — Section 5e4.
7. **Training infrastructure & phased plan** — Slurm jobs, checkpointing,
   compute budget allocation, rollout milestones. *Next.*

## 8. Suggestions / recommendations made so far

- **Architecture: two-stage hybrid.** Latent diffusion → autoregressive
  vectoriser (Section 3.1).
- **Tile size: 1.024 km²** as the default, revisable.
- **Coordinates: tile-local, quantised to 256 bins** per axis. Never
  global lat/lon.
- **9 raster channels** for the structural skeleton.
- **Vector vocabulary: ~2,400 tokens** total (~1,800 attribute tokens
  from the existing vocab work + ~256 coord tokens × 2 axes + ~30 type
  tokens).
- **Sequence length budget: ~5 k tokens per tile** (8 k context with
  FlashAttention).
- **Compute split: Stage A ~200–400 GPU-h, Stage B ~200–400 GPU-h,
  scale-up + ablations ~300 GPU-h. Total ~700–1,000 GPU-h.**
- **Train Stage A and Stage B independently in v1.** Joint fine-tune is
  a v2 lever.
- **Output format: GeoJSON-native.** With 2.5-D height per building.
- **Primary user persona: AV / game simulation engineer.** Funded,
  vector-friendly, batch-latency-tolerant.
- **Distribution: open weights + commercial license.**

## Appendix A — Architectures considered and rejected

(For reference if we want to revisit.)

| Architecture | Rejected because |
|---|---|
| Pure AR vector tokeniser (3.2) | Sequence length 10 k–100 k tokens too aggressive for v1 budget; long-context research risk |
| Layer-cascaded (3.3) | Three training pipelines triple the engineering surface; cascading errors |
| Pure raster diffusion + post-hoc vectorisation | Vector quality bounded by raster resolution; semantic labels hard to recover from pixels |
| Joint stage training end-to-end | Increases compute > 2×; harder to debug; defer to v2 |
| Per-region specialised models | Loses the "global vocabulary" leverage; multiplies training cost |

## Appendix B — Glossary

- **Tile**: a 1.024 km × 1.024 km square of map data (a single training
  example).
- **Raster channels (9)**: roads (4 classes), buildings (footprint +
  density), water, green, urban.
- **Vector tokens**: discrete tokens encoding geometry primitives,
  coordinates, and attributes. The unit Stage B emits.
- **Primitive type**: structural marker (`BUILDING_OPEN`, `ROAD_EDGE`,
  `POI`, `LAND_POLYGON`, etc.).
- **Stage A**: latent diffusion. Raster in, raster out.
- **Stage B**: autoregressive transformer. Raster in, vector tokens out.
- **Tile-local coordinates**: coordinates inside `[0, 1024)` m relative
  to tile origin, quantised to one of 256 discrete bins per axis.

---

*Update this file as the brainstorm progresses. Final spec lives in*
*`docs/superpowers/specs/2026-05-XX-genai-city-infrastructure-design.md`*
*once the open questions in §6 close out.*

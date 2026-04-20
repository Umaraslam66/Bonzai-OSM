# PBF to Modal City Model Pipeline

## Goal

Build a smoke-test pipeline that starts from a real OpenStreetMap `.osm.pbf`, trains a model on a single A100 on Modal, and evaluates whether the model learns **city structure**, not just generic graph statistics.

Long term, the target is not a normal graph model.

The target is a **geographic generative model** that can do for cities what an LLM does for language:

- continue partial structure coherently
- reconstruct missing urban context
- generate plausible new local neighborhoods
- eventually synthesize whole cities or city fragments under constraints

## What Was Checked

### Local data

The repo contains a real Luxembourg PBF:

- path: `data/luxembourg-260419.osm.pbf`
- size: `46,505,653` bytes
- modified: `2026-04-20 18:16`

This is enough to make the smoke test concrete.

### Known parseability from repo history

The repo already documents that a Luxembourg `.osm.pbf` of this type was successfully opened with GDAL's OSM driver on Leonardo and exposed the expected layers:

- `points`
- `lines`
- `multipolygons`

The project log also records prior Luxembourg summary counts from a verified run:

- `245,917` point features
- `227,434` line features
- `368,748` multipolygon features
- `138,491` road-like features
- `218,772` building-like features
- `62,698` POI-like features

Important:

Those counts come from an earlier Luxembourg extract recorded in the repo docs, not from a fresh parse of the exact local `260419` file in this shell session. The local file still needs a proper parsing run for exact current counts.

## Correct Framing

We are not trying to learn "graphs" in the abstract.

We are trying to learn **structured geography**:

- roads as transport skeleton
- buildings as built mass
- POIs as functional semantics
- landuse as urban context
- spatial relations as city logic

That means the smoke test should reward:

- connectivity
- accessibility
- neighborhood composition
- realistic spatial co-occurrence
- local continuity

It should not only reward:

- edge reconstruction on a toy graph benchmark
- generic graph embedding quality detached from place-making

## Recommendation

Use a **two-stage plan**:

1. First smoke test: graph-native **masked reconstruction** on local city subgraphs
2. Second smoke test: **conditional city construction** on the same graph representation

Do **not** begin with a full unconditional city generator.

That would make it too hard to tell whether failures come from:

- bad graph extraction
- weak representation learning
- poor ordering for autoregressive generation
- insufficient training scale

## Dataset Representation

Represent the city as a **heterogeneous attributed graph** built from OSM.

### Node types

Version 1 should use:

1. `ROAD_JUNCTION`
2. `ROAD_SEGMENT`
3. `BUILDING`
4. `POI`
5. `LANDUSE`

### Edge types

Version 1 should use:

1. `SEGMENT_CONNECTS_JUNCTION`
2. `JUNCTION_ADJACENT_JUNCTION`
3. `BUILDING_NEAR_SEGMENT`
4. `POI_NEAR_SEGMENT`
5. `BUILDING_INSIDE_LANDUSE`
6. `POI_INSIDE_LANDUSE`

This is enough to represent both:

- movement structure
- urban semantics

without turning the first pass into ontology sprawl.

### Node attributes

#### `ROAD_JUNCTION`

- degree bucket
- local coordinates

#### `ROAD_SEGMENT`

- `highway=*` bucket
- length bucket
- oneway
- lanes bucket
- surface bucket
- speed bucket
- bridge flag
- tunnel flag
- geometry summary bucket

#### `BUILDING`

- `building=*` bucket
- area bucket
- levels bucket
- height bucket
- shape bucket
- local centroid

#### `POI`

- primary key bucket
- primary value bucket
- local centroid

#### `LANDUSE`

- primary key bucket
- primary value bucket
- area bucket
- local centroid

## Spatial Unit for Training

Do not train on the whole country graph as one sample.

Use **overlapping local graph chunks**.

### Recommended chunk

Start with:

- projected square tiles
- tile size: `1 km`
- overlap: `128 m` to `256 m`

Each training sample is a local city fragment.

That is much closer to the long-term generation target:

- a neighborhood
- a block cluster
- a district fragment

rather than an entire country graph.

## Task Definition

The smoke test should answer:

**Can the model reconstruct or extend local urban structure from partial evidence?**

That is closer to "city language modeling" than standard graph SSL.

## Phase 1: City Graph Reconstruction

Train a graph-native encoder with self-supervised reconstruction losses.

### Inputs

A local heterogeneous graph chunk with some information hidden.

### Corruptions

1. mask node attributes
2. drop some typed edges
3. hide some node types entirely in a local region
4. optionally mask a contiguous spatial patch inside the tile

### Objectives

1. node attribute reconstruction
2. edge existence reconstruction
3. edge type reconstruction
4. neighborhood summary prediction
5. optional spatial patch completion

### Why this is the right first test

Because it measures whether the model has learned:

- what tends to go near what
- how roads organize buildings
- what a plausible local urban fragment looks like

without yet forcing the hardest part, which is free-form generation.

## Phase 2: City Construction

Only after phase 1 works, move to **constructive generation**.

The right framing is not "generate a graph from nothing" first.

The right framing is:

- given a seed road scaffold, add plausible buildings and POIs
- given a partial neighborhood, complete the missing structures
- given a landuse context, generate a coherent local street-and-building pattern

This is much closer to the intended long-term product.

### Recommended phase-2 tasks

1. **Neighborhood completion**
   Input: a chunk with a masked spatial patch
   Output: reconstruct the missing nodes and edges in that patch

2. **Road-conditioned construction**
   Input: road graph only
   Output: add buildings, POIs, and area context

3. **Landuse-conditioned construction**
   Input: coarse landuse plus main roads
   Output: build a plausible local urban pattern

These tasks are much more meaningful than abstract graph generation metrics.

## Model Recommendation

## First model

Use a **small heterogeneous graph transformer encoder**.

Not a pretrained text encoder.

Reason:

- the architecture family transfers
- the text-domain vocabulary and pretraining objective do not

### Suggested size

- hidden size: `256`
- layers: `4-6`
- attention heads: `4-8`
- typed edge embeddings
- learned node-type embeddings

This is appropriate for Luxembourg local chunks on a single A100.

## Second model

After reconstruction is working, add a **graph decoder** or action decoder for local construction.

Good options:

1. transformer decoder over graph actions
2. autoregressive graph decoder over a fixed local traversal

But this is phase 2, not the first thing to implement.

## Modal Training Pipeline

The training stack should be designed for a **single-node A100 Modal job**.

### Why Modal fits this

According to Modal's current docs:

- a function can target `gpu="A100"`, `gpu="A100-40GB"`, or `gpu="A100-80GB"`
- `gpu="A100"` may be automatically upgraded to an 80 GB A100 at the same cost
- Modal Volumes are intended for write-once, read-many training data and checkpoints
- long training jobs should be made resumable with checkpoints and retries

### Storage layout on Modal

Use two persistent stores:

1. **dataset volume**
   For parsed graph chunks and train/val/test manifests

2. **checkpoint volume**
   For model checkpoints, logs, and evaluation outputs

Optional:

3. **cloud bucket mount**
   If you later want training artifacts mirrored to S3/R2/GCS

### Modal app structure

Create one Modal app with three function groups:

1. `prepare_dataset`
   CPU-heavy or mixed preprocessing
   Input: local Luxembourg PBF or uploaded PBF
   Output: graph chunks written to the dataset volume

2. `train_model`
   GPU function on `A100`
   Input: graph chunks from the dataset volume
   Output: checkpoints and metrics to the checkpoint volume

3. `evaluate_model`
   GPU or CPU function depending on metric cost
   Input: checkpoint + validation/test chunks
   Output: evaluation JSON, plots, and sample reconstructions

### Retry and resume policy

Training on Modal should be resumable:

- save checkpoints periodically
- look for latest checkpoint on startup
- use retries
- keep the function timeout below Modal's max and rely on resume if needed

## Parsing Pipeline

### Stage 1: PBF extraction

Input:

- `data/luxembourg-260419.osm.pbf`

Output:

- points table
- lines table
- multipolygons table

### Stage 2: semantic filtering

Derive:

- roads from `lines`
- buildings from `multipolygons`
- POIs from `points` and selected tags
- landuse from `multipolygons`

### Stage 3: graph construction

Derive:

- road junctions
- road segments between junctions
- building-to-road relations
- poi-to-road relations
- containment edges for areas

### Stage 4: chunking

Tile into local overlapping city subgraphs.

### Stage 5: serialization

Save each chunk as:

- node table
- edge table
- graph metadata

Parquet is a good default for this stage.

## Evaluation Plan

Evaluation should match the long-term city-generation goal.

## Phase 1 metrics

1. masked node attribute accuracy
2. edge existence F1
3. edge type accuracy
4. local neighborhood summary accuracy
5. building-road proximity distribution match
6. junction degree distribution match

## City-structure metrics

These matter more than generic graph metrics:

1. connectivity realism
2. access realism
3. density realism
4. semantic co-occurrence realism

Examples:

- are residential buildings near residential/local roads more often than motorways?
- do POIs cluster near accessible segments?
- do buildings respect plausible local density?

## Phase 2 metrics

Once city construction starts:

1. masked-patch completion quality
2. road-conditioned building placement quality
3. nearest-real-neighborhood similarity
4. diversity under fixed conditioning
5. validity of generated relation types

## Success Criteria

The smoke test is successful if:

1. the PBF can be turned into stable local graph chunks
2. the graph model trains on a single Modal A100 without special tricks
3. reconstruction metrics beat shallow baselines
4. structural metrics look like real city fragments, not just arbitrary graphs

Only then should we invest in full city construction or unconditional generation.

## Concrete First Build

The first implemented pipeline should be:

1. parse Luxembourg PBF
2. build local heterogeneous city graph chunks
3. train a small graph transformer encoder on masked reconstruction
4. evaluate on held-out spatial chunks
5. visualize a few reconstructed neighborhoods

That is the shortest path from "real PBF data" to "does this help us model cities?"

## What This Means Conceptually

Yes, the intended model is analogous to an LLM.

But the analog of a "sentence" here is not a token sequence.

It is a **local city fragment**:

- roads
- buildings
- functions
- context

The analog of "next-token prediction" is not literally next token.

It is:

- complete the missing urban patch
- predict the missing structure conditioned on the visible neighborhood
- extend the current city fragment coherently

That is the right way to translate the LLM idea into geography.

## Sources

Official Modal docs used for the current infrastructure assumptions:

- GPU acceleration: https://modal.com/docs/guide/gpu
- Volumes: https://modal.com/docs/guide/volumes
- Long, resumable training: https://modal.com/docs/examples/long-training
- Cloud bucket mounts: https://modal.com/docs/guide/cloud-bucket-mounts

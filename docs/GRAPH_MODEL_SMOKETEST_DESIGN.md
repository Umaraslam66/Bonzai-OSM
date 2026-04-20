# Graph-Native Smoke Test Design

## Purpose

This document defines a graph-native alternative to the current token-sequence GPT-2 baseline in `stockholm_poc/`.

The goal is not to replace the existing tokenizer immediately. The goal is to run a small, defensible smoke test on a regional OSM extract and answer one question:

**Does a graph-native self-supervised model learn map structure better than a sequence-only baseline at the same small scale?**

## Recommendation

Use a **graph transformer encoder** with **masked graph reconstruction** as the first smoke test.

Do **not** start with:

- a pretrained text encoder used as-is
- a full graph autoregressive decoder
- end-to-end unconditional graph generation

Those are reasonable phase-2 or phase-3 directions, but they are the wrong first experiment.

## Why This Direction

The current branch is strong on serialization and simple LM training, but it flattens spatial structure into a token stream. That loses explicit graph relations such as:

- road-road connectivity
- segment-junction incidence
- building-road proximity
- polygon containment
- typed neighborhood structure

A graph-native model can preserve those relations directly.

At the same time, reusing a state-of-the-art **text** encoder is not the main win here. The useful transfer is mostly from the **architecture family** rather than the pretrained lexical weights. OSM graphs are not natural language, and tokens like road classes, geometry buckets, and typed spatial edges do not align with text pretraining.

## Main Design Decision

### Chosen first experiment

Train a **heterogeneous graph encoder** on a small OSM region using:

- masked node-attribute reconstruction
- masked edge-type prediction
- local neighborhood reconstruction

This is the lowest-risk graph-native test that still exercises topology.

### Deferred experiment

After the encoder smoke test succeeds, add a **graph decoder** and train an autoregressive graph model over a fixed graph ordering or filtration.

## Smoke Test Dataset

### Primary choice

Use **Luxembourg** as the smoke test region.

Why:

- it is already validated in this repo
- it is small enough for fast iteration
- it has enough roads, buildings, POIs, and landuse to make the task meaningful
- the existing docs and jobs already support Luxembourg extraction and inspection

### Optional secondary check

Use a **tiny Stockholm slice** only after Luxembourg works, to check whether the model survives a denser urban graph.

## Graph Construction

Represent each region as a **heterogeneous attributed graph**.

### Node types

Use these node families in the first version:

1. `ROAD_JUNCTION`
   Meaning: a road intersection or road endpoint

2. `ROAD_SEGMENT`
   Meaning: a road polyline between two adjacent junctions

3. `BUILDING`
   Meaning: one building polygon

4. `POI`
   Meaning: one tagged point-like feature

5. `LANDUSE`
   Meaning: one polygonal landuse or natural area

Skip `WATERWAY` and `RAILWAY` in the first smoke test unless extraction is already clean. They can be added after the core graph works.

### Edge types

Use typed edges rather than one generic adjacency:

1. `SEGMENT_CONNECTS_JUNCTION`
   Between `ROAD_SEGMENT` and its endpoint `ROAD_JUNCTION`s

2. `JUNCTION_ADJACENT_JUNCTION`
   Optional shortcut edge for directly connected junctions

3. `BUILDING_NEAR_SEGMENT`
   Nearest road segment within a threshold radius

4. `POI_NEAR_SEGMENT`
   Nearest road segment within a threshold radius

5. `POI_INSIDE_LANDUSE`
   If point lies inside area polygon

6. `BUILDING_INSIDE_LANDUSE`
   If building centroid lies inside area polygon

7. `AREA_ADJACENT_AREA`
   If polygons touch or overlap within a small tolerance

The first version should keep the graph sparse. The minimum viable graph is:

- junction-segment incidence
- building-near-road
- poi-near-road

### Node attributes

Keep attributes compact and mostly categorical in version 1.

#### `ROAD_JUNCTION`

- degree bucket
- traffic-control flag if available
- normalized coordinates or tile-local coordinates

#### `ROAD_SEGMENT`

- `highway=*` bucket
- length bucket
- oneway
- lanes bucket
- surface bucket
- speed bucket
- bridge flag
- tunnel flag
- simple geometry summary such as heading histogram or first-order shape code

#### `BUILDING`

- `building=*` bucket
- area bucket
- levels bucket
- height bucket
- compactness or rectangularity bucket
- tile-local centroid

#### `POI`

- main tag key
- main tag value bucket
- tile-local centroid

#### `LANDUSE`

- main tag key
- main tag value bucket
- area bucket
- tile-local centroid

### Coordinate handling

Do not feed raw global lat/lon directly.

Use:

- region-local projected coordinates
- normalized coordinates inside a tile or chunk
- optional coarse spatial bucket embeddings

This preserves spatial signal without making the model memorize absolute geography.

## Chunking Strategy

Do not train on one giant regional graph first.

Instead, partition the region into **overlapping local subgraphs**.

### Recommended chunk definition

Use square tiles in projected metric space, for example:

- 512 m
- 1 km
- 2 km

Start with **1 km tiles with overlap**.

Each training sample becomes one local heterogeneous graph:

- nodes from the tile interior
- context nodes from an overlap band
- supervision applied mainly to the interior

This gives:

- bounded memory
- better batching
- data augmentation through many local neighborhoods
- an easier path to scaling later

## Model Architecture

## Phase 1 model

Use a **heterogeneous graph transformer encoder**.

The practical shape should be close to:

- typed node embeddings
- typed edge embeddings
- local message passing
- global attention or sparse transformer mixing

The closest design family is a **GraphGPS-style hybrid**:

- local graph message passing for real edges
- transformer-style global mixing
- explicit structural encodings

That is a better fit than dropping a text-only encoder into the problem.

### Minimal encoder block

Per layer:

1. node-type + attribute embedding
2. relation-aware local message passing
3. sparse or global attention over nodes in the chunk
4. residual + norm + MLP

### Structural encodings

Add:

- node type embedding
- edge relation embedding
- degree embedding
- shortest-path or hop-distance bucket for selected pairs
- local coordinate embedding

## Why not pretrained text encoder weights first

If we use a pretrained text model directly, we inherit:

- a wordpiece vocabulary that does not match OSM primitives
- a language-domain inductive bias that is only weakly relevant
- parameter allocation optimized for syntax and lexical composition rather than typed topology

Inference:

The benefit of reusing a text encoder would mostly be engineering convenience, not domain alignment.

If we want a transformer backbone, it is cleaner to:

- reuse the architecture pattern
- initialize from scratch
- keep the model small for the smoke test

## Training Objective

## Phase 1 objective

Use **masked graph modeling**, not generation.

### Loss terms

1. Masked node-attribute reconstruction
   Predict hidden categorical attributes such as road class, building class, area bucket, levels bucket, and POI class.

2. Masked edge-type prediction
   Remove a subset of typed edges and predict whether an edge exists and what relation type it has.

3. Neighborhood reconstruction
   Predict local structural summaries:
   - node degree bucket
   - number of incident road segments
   - number of nearby buildings or POIs

4. Optional geometry summary reconstruction
   Predict coarse geometry codes for road segments or building shape buckets.

### Masking policy

Start with:

- 30% masked node attributes
- 15% dropped typed edges
- harder masking on high-degree road nodes only after the base pipeline is stable

### Why this objective first

It is:

- fully unsupervised
- stable
- aligned with graph representation learning literature
- easier to debug than autoregressive graph generation

## Phase 2 objective

After the encoder shows value, add a **graph decoder** trained on a causal graph objective.

### Recommended formulation

Use an **autoregressive action sequence over a graph filtration**, not free-form graph decoding.

At each step the model predicts one action such as:

1. add node of type `T`
2. assign node attributes
3. connect node `u` to node `v` with relation `r`
4. stop current subgraph

The ordering should come from a deterministic local traversal, for example:

- BFS from a seed road junction
- degree-aware canonical order
- filtration based on radius growth from a seed node

This keeps generation well-defined and makes teacher forcing possible.

## Decoder Choice

For phase 2, use a lightweight graph decoder, not a giant text LM head.

Two viable options:

1. Transformer decoder over graph actions
2. Graph autoregressive decoder over partial subgraphs

The second is more faithful to the structure, but the first is easier to implement.

For the smoke test, do **not** build this yet.

## Baselines

The smoke test should compare against at least two baselines.

### Baseline A

Current sequence pipeline on the same region, reduced to a small training run.

Question answered:

Does explicit graph structure beat token serialization?

### Baseline B

Graph-only shallow baseline:

- node2vec or random-walk embedding
- simple GNN with masked attribute prediction

Question answered:

Do we need the transformer component at all?

## Metrics

The smoke test should not be judged by loss alone.

### Representation metrics

1. Masked attribute accuracy
2. Edge existence F1
3. Edge type accuracy
4. Neighborhood summary accuracy

### Structural sanity metrics

1. Road connectivity recovery
2. Junction degree distribution match
3. Building-road nearest-distance distribution match
4. POI-road nearest-distance distribution match

### Transfer-style probes

Train tiny linear heads on frozen embeddings for:

1. road class prediction
2. building class prediction
3. POI category prediction
4. link prediction for held-out typed edges

### Phase 2 generation metrics

Only after a decoder exists:

1. valid edge-type ratio
2. valid node-type ratio
3. connected-component statistics
4. degree distribution distance
5. motif counts
6. graph edit distance to nearest real local graph

## Model Size for the Smoke Test

Keep it small.

Recommended starting point:

- hidden size: 256
- layers: 4 to 6
- heads: 4 to 8
- edge-type embeddings: small learned tables
- batch over many local subgraphs

This is enough to test the idea without confusing architecture risk with scale.

## Data Pipeline for the Smoke Test

### Input source

Use the validated Luxembourg regional extract path already documented in the repo.

### Preprocessing steps

1. parse OSM region
2. split roads into junction-segment graph
3. attach buildings to nearest segment
4. attach POIs to nearest segment or containing area
5. build landuse polygons as area nodes
6. tile into overlapping local subgraphs
7. write a compact graph dataset format

### Recommended saved format

For each local subgraph, persist:

- node table
- edge table
- graph-level metadata

A simple Parquet-based storage is fine for the smoke test. PyG or DGL sample serialization is also acceptable if the pipeline stays readable.

## Concrete Success Criteria

Call the smoke test successful if it meets all of these:

1. The graph encoder trains stably on Luxembourg local subgraphs.
2. Masked node and edge reconstruction beat a shallow graph baseline.
3. Frozen graph embeddings support useful probes for road, building, and POI types.
4. Structural metrics look plausible without hand-tuned post-processing.

Only then should we fund phase 2 work on graph generation.

## Risks

### Risk 1: graph schema bloat

If we add too many node and edge types early, the experiment turns into ontology engineering instead of modeling.

Mitigation:

- start with roads, buildings, POIs
- add landuse only if extraction is clean

### Risk 2: spatial leakage

If absolute coordinates dominate, the model may memorize region layout instead of learning structure.

Mitigation:

- use local coordinates
- use overlapping tiles
- hold out spatial regions during evaluation

### Risk 3: decoder complexity

Graph generation is much harder than graph representation learning.

Mitigation:

- separate encoder smoke test from decoder work
- do not conflate both experiments

### Risk 4: false confidence from tiny data

Small regional graphs can make many architectures look good.

Mitigation:

- use spatial holdout
- compare against the current token baseline
- run a second region after Luxembourg

## Proposed Execution Plan

## Step 1

Build a Luxembourg heterogeneous graph dataset with:

- `ROAD_JUNCTION`
- `ROAD_SEGMENT`
- `BUILDING`
- `POI`

## Step 2

Tile it into overlapping local subgraphs.

## Step 3

Train a small graph transformer encoder with masked node and edge reconstruction.

## Step 4

Evaluate:

- reconstruction quality
- link prediction
- structural sanity
- frozen-embedding probes

## Step 5

Compare against the sequence baseline on the same smoke test region.

## Step 6

Only if phase 1 wins, design the graph decoder and causal graph prediction experiment.

## Decision Summary

The recommended next model is:

**a small graph-transformer encoder trained with masked graph reconstruction on Luxembourg local subgraphs**

The recommended next experiment is **not**:

- a pretrained text encoder dropped into OSM directly
- a full graph generator as the first smoke test

Those remain valid later directions, but they should come after the graph-native encoder proves that explicit topology helps.

## References

- Graphormer: https://openreview.net/forum?id=OeWooOxFwDa
- GraphGPS: https://openreview.net/forum?id=lMMaNf6oxKM
- AutoGraph: https://openreview.net/forum?id=eszmES7j1F
- Autoregressive filtration modeling: https://openreview.net/forum?id=3Up81Zq728
- Masked graph autoencoder with bandwidth masking: https://openreview.net/forum?id=0iwNrRRIiZ
- ModernBERT docs: https://huggingface.co/docs/transformers/model_doc/modernbert

# Bonzai-OSM — Generative City Model

A two-stage generative model that produces novel, geometrically valid, plausibly-real city layouts as **vector geometry** (roads, building footprints, points of interest, land use) from a text prompt.

## What this is

**Stage A — the Sketcher** is a small DiT diffusion model that paints a 9-channel coloured raster of a 2 km² tile from a prompt.

**Stage B — the Inker** is an autoregressive transformer that traces the Sketcher's coloured raster into precise vector geometry: building polygons, road graphs with class labels, labelled POIs, land-use polygons.

Output: **GeoJSON**, dropped into Unity / Unreal / CARLA / Foretellix / any GIS tool.

The headline interaction:

```
generate_city(
    prompt = "dense European commercial district, mid-rise, coastal",
    area  = 4.2 km²,
    conditioning = { terrain_DEM, coastline_mask, style_anchor, constraint_mask }  # all optional
) → GeoJSON FeatureCollection
```

## Status

**Branch:** `genai-city-model` (long-lived development branch; never merged to `main`).
**Phase 0a — Data Prep Pipeline:** in progress.
See [`docs/superpowers/STATUS.md`](docs/superpowers/STATUS.md) for the live progress tracker.

## Where to read

The repository is organised so a new contributor (or new agent in a new session) can come up to speed in 30 minutes by reading:

1. [`docs/superpowers/STATUS.md`](docs/superpowers/STATUS.md) — current state, last commit, next action.
2. [`PROJECT.md`](PROJECT.md) — project context, allocation, ground-truth state.
3. [`docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md`](docs/superpowers/specs/2026-05-03-genai-city-infrastructure-design.md) — the design spec (architecture, data, eval, phases).
4. [`docs/superpowers/plans/2026-05-03-phase-0a-data-prep-pipeline.md`](docs/superpowers/plans/2026-05-03-phase-0a-data-prep-pipeline.md) — current implementation plan (20 tasks, checkboxes track progress).
5. [`docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md`](docs/superpowers/brainstorm/2026-05-02-bonzai-genai-brainstorm-log.md) — full design rationale and rejected alternatives.

## Repo layout

```
Bonzai-OSM/
├── README.md                      ← this file
├── PROJECT.md                     ← project log, allocation, Leonardo state
├── commands.md                    ← Leonardo command reference
├── cspell.json
├── .gitignore
├── docs/superpowers/              ← spec, plan, brainstorm, status
│   ├── STATUS.md
│   ├── specs/
│   ├── plans/
│   └── brainstorm/
├── bonzai_genai/                  ← production codebase (in progress)
│   └── …
└── scripts/
    └── leonardo_download_planet.sh
```

## Compute & data

- **HPC allocation:** `AIFAC_P02_222` on CINECA Leonardo (EuroHPC). 40,000 core-h initial allowance; extension requested after Phase 0a de-risking shows green.
- **Free CPU partition:** `lrd_all_serial` (4 cores / 30 GB / 4 h walltime) — used for all data prep.
- **Billed GPU partition:** `boost_usr_prod` (4× A100, 24 h walltime) — used only for model training.
- **Data sources:** Overture Maps `2026-04-15.0`, Foursquare OS Places `dt=2026-04-14`, OpenStreetMap planet PBF.

## v1 de-risking countries

Phase 0a produces tile datasets for three countries chosen for climatic and morphological contrast:

| Country | Köppen | Urban form | Geofabrik size |
|---|---|---|---|
| Sweden | Cfb / Dfb | Northern European low-rise + sparse rural | ~700 MB |
| Singapore | Af | Tropical ultra-dense gridded high-rise | ~30 MB |
| Sri Lanka | Af / Aw | Tropical mixed urban + rural, dense low-rise | ~150 MB |

## Pitches at three audience levels

**Layperson / dinner-table:**

> "We're teaching a computer what real cities look like — by showing it millions of them — so it can invent new ones. You type 'a dense European commercial district,' and it draws you a plausible map: streets in the right places, building footprints, shops, parks. Not a picture of a map, but the actual data a video game or a self-driving simulator can use."

**Technical / non-ML (CTO, journalist):**

> "Imagine Stable Diffusion, but for **maps** instead of images. SD generates pictures from a prompt; ours generates the underlying vector data of cities — roads, buildings, businesses — from a prompt. Self-driving companies pay around $5,000 per kilometre to map real cities and they only have a few; we generate unlimited plausible ones on demand."

**ML reviewer:**

> "We jointly train (i) a small multi-channel latent diffusion model on rasterised OSM/Overture tiles, and (ii) an autoregressive transformer that vectorises the diffusion's output into typed primitives (roads, buildings, POIs, land polygons) with tile-local quantised coordinates. Contribution: a tractable raster-to-vector decoding scheme for **semantically labelled** geometry at city scale."

## License

TBD — open weights + commercial license for the trained models; code license to be decided before paper submission.

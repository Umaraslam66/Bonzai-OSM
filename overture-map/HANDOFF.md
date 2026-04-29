# Handoff prompt — Bonzai-OSM / overture-map

> Paste this into the next Claude Code session to continue the work. Last
> updated 2026-04-29 after the Overture ∪ Foursquare dedup landed.

---

## Project context

We are building a **global vocabulary for a generative spatial model** that
will procedurally generate cities (roads, buildings, land use, POIs). The
vocab is empirically derived from frequency analysis of three open
datasets:

- **Overture Maps 2026-04-15.0** — buildings, transportation, places, base,
  divisions, addresses (75 M places, 2.5 B buildings, 344 M road segments).
- **Foursquare OS Places dt=2026-04-14** — 99.9 M places, hierarchical
  categories, strong in Asia/LatAm/Eastern Europe.
- **OpenAddresses 2025-09-14 archive** — 500–600 M addresses across 62
  countries (US/EU/AU heavy).

Working on branch `overture-map`. Project dir: `overture-map/` inside
`Bonzai-OSM`. CINECA Leonardo HPC project: `AIFAC_P02_222`
(40 000 core-hours/quarter, ~30 used as of 2026-04-29).

## What's done

1. **Per-column frequency passes** on the full Overture release — see
   `outputs/freq_*.csv` (27 files) and `docs/VOCAB_ANALYSIS.md`.
2. **Foursquare frequency passes** — `outputs/fsq_freq_*.csv`.
3. **OpenAddresses inventory** — `outputs/oa_country_summary.csv` (62
   countries, 4 850 source CSVs, 35 GB raw).
4. **Spatial dedup Overture ∪ Foursquare** (2026-04-29):
   - `scripts/08_dedup_places.py` — streaming grid hash-join (550 m bucket,
     3×3 expansion, ≤ 50 m + Jaro-Winkler ≥ 0.6).
   - Ran on `boost_usr_prod` with 200 GB / 32 cores in ~25 min wall.
   - `outputs/dedup_summary.csv` — headline counts.
   - `outputs/dedup_pairs.parquet` (26.7 GB, Leonardo only) — 1.44 B raw
     candidate pairs.
   - `outputs/dedup_fsq_decisions.parquet` (7.9 GB, Leonardo only) — best
     Overture match per FSQ place + tier.

### Dedup headline

| metric | value |
|---|---:|
| Overture places | 75,495,994 |
| Foursquare places | 99,904,729 |
| FSQ matched (≤50 m + name_sim ≥ 0.6) | **48,252,270** |
| FSQ unique | **51,652,459** |
| Overture matched | 32,559,028 |
| Overture unique | 42,936,966 |
| **Merged universe** | **127,148,453** |

## What's next (priority order)

1. **Build the unified POI master table.** Outer-join Overture +
   `dedup_fsq_decisions.parquet` to materialize one row per merged place:
   `(merged_id, ovt_id?, fsq_id?, lat, lon, ovt_primary_cat,
   fsq_primary_cat, fsq_hierarchical_labels, source_set, dedup_tier)`.
   ~127 M rows. Save as `outputs/poi_master.parquet`. Most natural place
   for this is a new `scripts/09_build_master.py`.

2. **Re-run frequency passes on the merged universe.** FSQ's hierarchical
   taxonomy (`Dining and Drinking > Cafe, Coffee, and Tea House > Café`)
   gives a clean **top-11 / level-2 (~150) / level-3 (~700)** cascaded
   vocab. Compare against Overture's flat 690-keep tail. Update
   `docs/VOCAB_ANALYSIS.md`.

3. **Overture Bridge Files for OSM building labels.** Pull
   `s3://overturemaps-us-west-2/bridgefiles/2026-04-15.0/dataset=osm/` and
   join to OSM planet PBF tags (`building=*`, `shop=*`, `amenity=*`) to
   recover semantic labels for the 94 % null Overture buildings.

4. **(Optional) Per-region OpenAddresses integration.** Street-number /
   unit-level tokens for US/EU. `scripts/07_oa_inspect.py` already maps
   the global zip; CSV → Parquet conversion per country is a one-shot.

## Operating environment

### Local (Mac)

```bash
cd ~/Documents/dynamo/Bonzai-OSM/overture-map
.venv/bin/python ...   # Python 3.12 venv
```

### Leonardo

- ssh: `ssh uaslam00@login.leonardo.cineca.it`
- project work dir: `/leonardo_work/AIFAC_P02_222/overture-map`
- scratch: `$CINECA_SCRATCH/bonzai-data/` (40-day TTL)
- partitions enabled for `AIFAC_P02_222`:
  - `lrd_all_serial` (4 cores / 30 GB / 4 h, free) — fine for streaming
    S3 scans, NOT enough RAM for the dedup window function.
  - `boost_usr_prod` (32 cores / 512 GB / GPU optional, charges budget) —
    use this for any heavy in-memory work; ~30 core-hours per dedup run.
- `dcgp_usr_prod` is **not** enabled for this project.

### Login-node gotchas (learned the hard way)

- Cgroup cap is ~33 GB; **`SET memory_limit='10GB'` is not enough** for
  big DuckDB sorts because mmap'd parquet pages and OS page-cache live
  *outside* DuckDB's budget and push total RSS past the kernel limit.
- Anything sorting > ~100 M rows or hash-joining > ~1 B intermediate
  rows: submit to `boost_usr_prod` (200 GB sbatch, 32 threads, no spill).
- The dedup_prod sbatch template is `leonardo_dedup_prod.sbatch` (only on
  Leonardo currently — not in repo; recreate from `leonardo_dedup.sbatch`
  with: partition → `boost_usr_prod`, cpus → 32, mem → 200G, add
  `--qos=normal` and `--gres=gpu:0`).

## Repo state

- Branch: `overture-map` (not yet merged to `main`).
- Last commit: `Overture ∪ Foursquare spatial dedup — 127M unified POIs`.
- Untracked at repo root: `city_graph_review.html`, `image*.png` — local
  scratch, ignore.

## Recommended first move

Read `docs/MULTI_SOURCE_ANALYSIS.md` end-to-end (especially the
"Dedup outcome" and "Next concrete actions" sections). Then start on
Step 1 (`09_build_master.py`) — submit it to `boost_usr_prod` from the
beginning; don't try the login node for joins of this size.

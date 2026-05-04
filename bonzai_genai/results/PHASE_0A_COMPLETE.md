# Phase 0a Completion -- Data Prep Pipeline + Sweden + Singapore + Sri Lanka Tiles

**Completed:** 2026-05-04
**Branch:** `genai-city-model`
**Cluster:** Leonardo Booster (CINECA), `lrd_all_serial` partition (budget-free)
**GPU-h consumed this phase:** 0

## Deliverables shipped

- Package `bonzai_genai/`, fully tested (55 / 55 pass, ruff-clean) on Mac (Py 3.12) and Leonardo (Py 3.11.7)
- Token id space, attribute vocabulary, tokeniser (encode + decode)
- Vector -> 9-channel rasteriser with road class hierarchy
- TileBundle dataclass + WebDataset shard I/O
- Procedural smoke-test generator
- End-to-end synthetic round-trip test
- Real-data tile sampler (pyosmium-backed; bucketed spatial index)
- Slurm job template for Leonardo data prep
- Sweden tile dataset: `$WORK/bonzai-tiles/sweden/`
- Singapore tile dataset: `$WORK/bonzai-tiles/singapore/`
- Sri Lanka tile dataset: `$WORK/bonzai-tiles/sri_lanka/`

## Counts (from on-cluster manifests)

| Country | Tiles kept | Tiles attempted | Shards | Disk on `$WORK` | Slurm wall |
|---|---:|---:|---:|---:|---:|
| Singapore (SG, Af) | 203 | 300 | 3 | 1.9 GB | 06:17 |
| Sri Lanka (LK, Aw) | 384 | 2,000 | 4 | 3.4 GB | 04:15 |
| Sweden (SE, Cfb) | 1,301 | 5,000 | 14 | 12 GB | 11:50 |
| **Total** | **1,888** | **7,300** | **21** | **~17.3 GB** | **22:22** |

Skip reasons:
- **Singapore:** 97 skips, all NODE_REF overflow (>8192 unique road nodes per 2 km^2 tile -- central Marina Bay multi-level interchanges). Genuinely extreme density.
- **Sri Lanka:** 1,616 skips, almost entirely the "near-empty tile" rule (`<5 features`). The bbox covers a lot of ocean (south of the island).
- **Sweden:** 3,699 skips, "near-empty" rule. Sweden's 14.5deg lat x 14deg lon bbox includes large stretches of forest, mountain, and coastal water.

## What landed during execution beyond the original 20-task plan

- **Plan Task 18.5** (`c700101`) lifts the road-node cap from 512 -> 8192 by adding a dedicated `NODE_REF_*` token family (4 new tokens funcs + 5 new tests). Singapore Marina Bay tiles still overflow but the rest of the country fits.
- **Plan Task 18.6** (`e103447`) refactors `data/sampling.py` to pyosmium with a country-bbox load + 0.05deg bucketed spatial index. The original `osmium-tool` CLI subprocess approach didn't work on Leonardo (no `osmium` module, no conda). pyosmium ships manylinux wheels with bundled libosmium so dev-laptop and HPC use the same code path.
- **Sampler shuffle fix** (`8e0b753`) makes `iter_tile_centres` deterministically shuffle its centres list before yielding, so a downstream `max_tiles` cap takes a uniform sample rather than the SW-corner slice. Without this, Sweden's first run produced 0/5000 because the SW corner of the Sweden bbox is in Skagerrak / North Sea.

## Sample tile spot-check (3 bundles per country)

```
=== singapore ===
  SG-000000: tokens=343  raster_sum=23986 sw=(1.2000,103.6000)
  SG-000006: tokens=928  raster_sum=80368 sw=(1.2000,103.7104)
  SG-000007: tokens=227  raster_sum=4001  sw=(1.2000,103.7288)

=== sri_lanka ===
  LK-000445: tokens=345   raster_sum=5993  sw=(5.9052,80.5673)
  LK-000446: tokens=12112 raster_sum=31003 sw=(5.9052,80.5858)   <- dense urban
  LK-000569: tokens=3639  raster_sum=21207 sw=(5.9236,80.4563)

=== sweden ===
  SE-000007: tokens=1343 raster_sum=9112  sw=(63.8492,18.3476)   <- northern coast
  SE-000009: tokens=5704 raster_sum=7601  sw=(58.7715,14.7582)
  SE-000013: tokens=1691 raster_sum=43960 sw=(58.9187,17.3066)   <- Stockholm region
```

Every bundle has non-zero token count and non-zero raster sum. The shuffled Sweden sampler does reach Stockholm-region (lat 58.9) and far-northern (lat 63.8) tiles -- not just Skagerrak.

## Open follow-ups for Plan 2+

- [ ] Expand attribute vocab YAML to ~1,800 tokens (FSQ leaves) -- currently ~290.
- [ ] Multipolygon-relation buildings / landuse: pyosmium handler currently only ingests simple closed ways. Add `osmium.area.AreaManager` support so relation-based features (some big malls, lakes, parks) are picked up.
- [ ] Stratification logic: bbox-uniform random sampling is fine for de-risking; production runs (Plan 2) should stratify by population density / Köppen / urban-rural.
- [ ] Building-height extraction from OSM `building:height` / `building:levels` tags (currently all heights = `height=NA`).
- [ ] Overture Bridge-file enrichment for the ~6% of buildings whose OSM tags don't classify cleanly.
- [ ] Switch tile sampler from pyosmium to direct Overture parquet reads via DuckDB (Plan 5) for whole-planet scale.
- [ ] Marina Bay still overflows the 8192 NODE_REF cap. Either bump to 16384 (vocab cost) or add a tile-level pre-crop that splits >8k-node tiles in two. Decide before Plan 2.
- [ ] `max_tiles` is a hard cap; Sweden's 5,000 yields 1,301 kept. If we want O(10k) Sweden tiles we either bump max_tiles or add a "keep until N successful" loop instead of "try N centres then stop".

## Hand-off

Plan 2 (synthetic smoke harness for Experiment 0) and Plan 3 (Stage A / Sketcher code) can now begin. Both depend only on the tokeniser, rasteriser, and shard I/O implemented in Phase 0a -- all green.

Three real-country tile shards live on Leonardo `$WORK/bonzai-tiles/{singapore,sri_lanka,sweden}/`. They survive the 6-month-past-project retention window so Plan 2+ can read them directly.

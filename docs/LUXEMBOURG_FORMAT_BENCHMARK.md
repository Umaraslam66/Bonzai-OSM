# Luxembourg Format Benchmark

This document defines the first format comparison benchmark for a complete regional export from OpenStreetMap on Leonardo.

Date created: `2026-04-15`
Scope: full Luxembourg extract
Goal: compare practical intermediate formats for world-scale preprocessing

## Why This Benchmark Exists

The roads-only Luxembourg GeoJSON test proved that:

- GDAL can parse `.osm.pbf` on Leonardo
- `lrd_all_serial` is sufficient for budget-free regional extraction
- the workflow is operational

That is not enough to choose the right storage format for the full-world pipeline.

We now need a complete regional export across multiple formats so we can compare:

- runtime
- output size
- layer handling
- ease of downstream processing
- suitability for world-scale partitioning

## Formats Included

These are the formats we can benchmark directly with the currently verified Leonardo toolchain (`gdal/3.8.5--gcc--12.2.0`):

1. `GeoJSON`
   - one file per OSM layer
   - best for debugging
   - expected to be verbose and large

2. `GeoPackage`
   - one multi-layer file
   - strong prototype format
   - likely much better than GeoJSON for regional work

3. `Parquet` / GeoParquet
   - one file per OSM layer
   - best candidate for the world-scale intermediate store
   - compressed, columnar, partition-friendly

4. `GeoJSONSeq`
   - one file per OSM layer
   - newline-delimited features
   - good approximation of a streaming JSONL-style intermediate

## Formats Not Included in the Direct Extraction Benchmark

- `Arrow`
  - still recommended later for tokenized training shards
  - not the right direct `.osm.pbf` export target for this stage

- raw `OSM PBF`
  - already kept as the canonical source format
  - not compared as an extracted intermediate

## Layers To Export

Export all currently validated OSM layers from the Luxembourg `.osm.pbf`:

- `points`
- `lines`
- `multipolygons`

No semantic filtering in this benchmark. The purpose is full regional extraction, not theme-specific slicing.

## Outputs

Expected output roots under:

```text
/leonardo_scratch/large/userexternal/uaslam00/osm/benchmarks/luxembourg/
```

Planned layout:

```text
geojson/
  points.geojson
  lines.geojson
  multipolygons.geojson
  summary.txt

gpkg/
  luxembourg_full.gpkg
  summary.txt

parquet/
  points.parquet
  lines.parquet
  multipolygons.parquet
  summary.txt

geojsonseq/
  points.geojsons
  lines.geojsons
  multipolygons.geojsons
  summary.txt
```

## Metrics To Record

For each format:

- Slurm job id
- elapsed wall time
- per-layer output sizes
- total output size
- success/failure
- obvious usability notes

## Benchmark Interpretation

The Luxembourg benchmark will not linearly predict whole-world runtime, but it will tell us:

- how much each format inflates size relative to the source `.osm.pbf`
- how much per-layer overhead each export path has
- whether multi-layer packaging is useful
- whether columnar output is worth prioritizing early

## Working Hypothesis

Expected ranking for the main extracted world dataset:

1. `Parquet` / GeoParquet
2. `GeoPackage` for prototypes only
3. `GeoJSONSeq` for streaming/debug-oriented tasks
4. `GeoJSON` only for inspection and validation

## Result Table

Observed on 2026-04-15:

| Format | Job ID | State | Elapsed | Output Size | Notes |
| ------ | ------ | ----- | ------- | ----------- | ----- |
| GeoJSON | `39908595` | `COMPLETED` | `00:00:29` | `349M` total | separate files per layer |
| GeoPackage | `39908596` | `COMPLETED` | `00:00:11` | `280M` total | fastest successful run, single multi-layer file |
| Parquet | `39908598` | `FAILED` | `00:00:02` | n/a | GDAL build on Leonardo lacks `Parquet` driver |
| GeoJSONSeq | `39908600` | `COMPLETED` | `00:00:23` | `351M` total | separate newline-delimited files per layer |

## Measured Outputs

### GeoJSON

Summary:

```text
points          57M     2s
lines           94M     7s
multipolygons   193M    18s

349M    /leonardo_scratch/large/userexternal/uaslam00/osm/benchmarks/luxembourg/geojson
```

Interpretation:

- works reliably
- easy to inspect
- very verbose
- clearly not the right whole-world intermediate format

### GeoPackage

Summary:

```text
gpkg    280M    10s
```

Observed layer counts:

- `points`: `245,917`
- `lines`: `227,434`
- `multipolygons`: `368,748`

Total confirmed features:

- `842,099`

Interpretation:

- best successful format in this benchmark
- fastest runtime
- smallest successful output
- strong regional prototype format
- likely still not the final world-scale extracted store

### Parquet

Observed error:

```text
ERROR 1: Unable to find driver `Parquet'.
```

Interpretation:

- this is not a data problem
- this is a toolchain limitation on Leonardo's current GDAL module
- we cannot currently benchmark `Parquet` / GeoParquet directly with the available system GDAL build

### GeoJSONSeq

Summary:

```text
points          57M     1s
lines           90M     6s
multipolygons   195M    14s

351M    /leonardo_scratch/large/userexternal/uaslam00/osm/benchmarks/luxembourg/geojsonseq
```

Interpretation:

- operational and easy to stream
- size is almost identical to plain GeoJSON
- only modest runtime improvement over plain GeoJSON
- potentially useful for streaming/debug workflows, but not a clear winner for the world dataset

## Comparison Summary

Among the formats we can actually produce on Leonardo right now:

1. `GeoPackage` is the best prototype/export container
2. `GeoJSONSeq` is slightly better than `GeoJSON` for streaming-style inspection
3. `GeoJSON` remains useful only for debugging and validation
4. `Parquet` remains the likely best world-scale intermediate format in principle, but is blocked by the missing driver in the current GDAL build

Size inflation relative to the Luxembourg source file (`44.34M`):

- `GeoJSON`: about `7.9x`
- `GeoJSONSeq`: about `7.9x`
- `GeoPackage`: about `6.3x`

These ratios are useful for intuition but should **not** be linearly projected to the whole planet as if we were going to export one giant single-file world dataset. Whole-world processing should still be partitioned.

## Next Decision After Benchmark

After these exports complete, choose:

1. the intermediate world-scale extracted format
2. the first compact training-shard format
3. the partitioning scheme for whole-world extraction

## Current Decision After First Benchmark

Current recommendation based on observed reality:

1. Use `GeoPackage` for small and medium regional prototypes on Leonardo now
2. Do not use `GeoJSON` as the main extracted world format
3. Treat `GeoJSONSeq` as optional, mainly for streaming/debug use
4. If we want a proper world-scale intermediate store, we need one of:
   - a user-space GDAL build with `Parquet` enabled
   - another export/conversion path that produces `GeoParquet`
5. Training shards should still likely end up in `Arrow/Parquet` or `JSONL`, but that is a later tokenization-stage decision, not a direct `.osm.pbf` export decision

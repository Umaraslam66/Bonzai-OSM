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

Fill this after running the jobs.

| Format | Job ID | State | Elapsed | Output Size | Notes |
| ------ | ------ | ----- | ------- | ----------- | ----- |
| GeoJSON | TBD | TBD | TBD | TBD | |
| GeoPackage | TBD | TBD | TBD | TBD | |
| Parquet | TBD | TBD | TBD | TBD | |
| GeoJSONSeq | TBD | TBD | TBD | TBD | |

## Next Decision After Benchmark

After these exports complete, choose:

1. the intermediate world-scale extracted format
2. the first compact training-shard format
3. the partitioning scheme for whole-world extraction

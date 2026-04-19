# Stockholm PoC — Results

Evolution of the spatial tokenization pipeline on the BBBike Stockholm
extract, executed on Leonardo's budget-free `lrd_all_serial` partition.
Zero core-hours burnt across all three runs.

## V2.1 (current) — percentile bbox + full feature coverage

| Field            | Value                                  |
|------------------|----------------------------------------|
| SLURM job        | `40206301` on `lrd_all_serial`         |
| Wall time        | ~120 s                                 |
| Input PBF        | `Stockholm.osm.pbf` (54 MB, BBBike)    |
| Parquet output   | 18 MB, 69 row groups, zstd             |
| Objects          | **688,551** (6 feature kinds)          |
| **Total vocab**  | **666 tokens** (ceiling 1000)          |

### Objects by kind

| Kind      | Count   | GDAL reference | Match |
|-----------|---------|----------------|-------|
| BUILDING  | 369,723 | 369,782        | ✓     |
| ROAD      | 229,113 | 228,902        | ✓     |
| LANDUSE   | 44,344  | —              | new   |
| POI       | 35,810  | 50,411 poi-like | subset (amenity/shop only) |
| WATERWAY  | 4,869   | 4,863          | ✓     |
| RAILWAY   | 4,692   | 4,403          | ✓     |

### Vocabulary audit

| Family        | Unique | Notes                                     |
|---------------|-------:|-------------------------------------------|
| `X`           | 256    | full grid (percentile bbox)               |
| `Y`           | 256    | full grid                                 |
| `MOVE`        | 72     | 8 dirs × 9 distance buckets + edges       |
| `TAG`         | 60     | bucketed across 6 kinds                   |
| `LEVELS`      | 4      | 1-2, 3-5, 6-10, 11+                       |
| `SPEED`       | 3      | low <40, mid 40-70, high >70 kph          |
| `SURFACE`     | 2      | paved, unpaved                            |
| start/end × 6 | 12     | BUILDING/ROAD/POI/LANDUSE/WATERWAY/RAILWAY |
| `PART_SEP`    | 1      | MultiPolygon ring separator               |
| **Total**     | **666** | under 1000-token ceiling                |

### Sample sequences

```
BUILDING: <BUILDING_START> <TAG_YES> <X_0> <Y_0> <MOVE_SE_5M> <MOVE_SE_5M>
          <MOVE_NE_5M> <MOVE_NW_15M> <MOVE_SW_10M> <BUILDING_END>

ROAD:     <ROAD_START> <TAG_RESIDENTIAL> <SURFACE_PAVED> <X_0> <Y_0>
          <MOVE_W_25M> <MOVE_W_15M> <MOVE_W_5M> <MOVE_W_50M> <ROAD_END>

POI:      <POI_START> <TAG_PUBLIC_AMENITY> <X_0> <Y_0> <POI_END>

LANDUSE:  <LANDUSE_START> <TAG_RESIDENTIAL> <X_0> <Y_0>
          <MOVE_S_15M> ... <LANDUSE_END>

WATERWAY: <WATERWAY_START> <TAG_STREAM> <X_0> <Y_1>
          <MOVE_SE_50M> ... <WATERWAY_END>

RAILWAY:  <RAILWAY_START> <TAG_RAIL> <X_2> <Y_0>
          <MOVE_NE_100M> ... <RAILWAY_END>
```

## Version history

| Version | Anchor strategy        | Vocab   | Kinds | Features tokenized |
|---------|------------------------|--------:|------:|-------------------:|
| V1      | H3 res-11              | 289,073 | 2     | 598,836            |
| V2      | 256x256 grid, min/max bbox | 251 | 6     | 688,551            |
| V2.1    | 256x256 grid, percentile bbox | 666 | 6 | 688,551            |

- V1 → V2: H3 vocab blew up with area. Swapped to chunk-local X/Y grid
  and added POI/LANDUSE/WATERWAY/RAILWAY + attribute tokens (LEVELS,
  SPEED, SURFACE).
- V2 → V2.1: min/max bbox was pulled huge by a few outlier waterway
  and landuse relations, wasting ~80% of the anchor grid. Switched to
  p0.5/p99.5 percentile bbox: outliers clamp to `<X_0>`/`<X_255>`
  explicit-edge sentinels, city core gets the full 256 cells. V2's
  251-token vocab was misleadingly low — it only used 44×53 cells.

## Artifact paths on Leonardo

```
$WORK/stockholm_poc/data/Stockholm.osm.pbf
$WORK/stockholm_poc/outputs/stockholm_tokens.parquet
$WORK/stockholm_poc/outputs/stockholm_vocab.json
```

## Scaling implications

- Per-chunk vocabulary is now capped and stable. For planet scale, only
  chunk-level MOVE/TAG/structural tokens need to merge; anchors stay
  fixed at 512 slots.
- Percentile bbox generalises to any region (rural, oceanic, mixed) —
  no dependence on buildings being the dominant class.
- 334 tokens of headroom under the 1000-token ceiling for future
  attribute additions (building materials, road lanes, etc.).
- Next decision: what defines a "chunk" globally (S2 level, H3 parent,
  admin region). To pick before full-planet extraction.

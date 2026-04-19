# Stockholm PoC — Results

First end-to-end run of the tokenization pipeline on the BBBike
Stockholm extract, executed on Leonardo's budget-free
`lrd_all_serial` partition.

## Run summary

| Field            | Value                                                 |
|------------------|-------------------------------------------------------|
| Date             | 2026-04-19                                            |
| Leonardo account | `AIFAC_P02_222`                                       |
| Partition        | `lrd_all_serial` (budget-free)                        |
| SLURM job IDs    | `40148107` (H3 anchor), `40148589` (X/Y grid anchor)  |
| Wall time        | 91 s → 77 s after H3 → X/Y swap                       |
| Peak RSS         | 1.16 GB                                               |
| Core-hours burnt | 0 (free partition)                                    |
| Input PBF        | `Stockholm.osm.pbf` (54 MB, BBBike)                   |
| Output parquet   | 15 MB (60 row groups, zstd)                           |
| Output vocab     | 9 KB                                                  |

## Vocabulary audit — the anchor pivot

Initial H3-res-11 anchor design exploded the vocabulary. Replaced with
a chunk-local 256x256 grid emitting `<X_ix>` + `<Y_iy>` tokens. Same
corpus, same row count:

| Family       | H3 anchor (before) | X/Y grid (after) |
|--------------|--------------------|------------------|
| Anchor       | 288,966 (H3)       | **482** (X+Y)    |
| MOVE         | 69                 | 69               |
| TAG          | 33                 | 33               |
| BUILDING/ROAD start+end | 4        | 4                |
| PART_SEP     | 1                  | 1                |
| **Total**    | **289,073**        | **589**          |

490x reduction in vocabulary with no information loss — moves still
carry the fine-grained shape; anchors only need to place the object
within its chunk.

Note: X/Y families show 238 and 244 (not the full 256) because the
chunk bounding box is the tight enclosing rectangle — the extreme
corner cells simply contain no anchor points.

## Corpus

- Tokenized objects: **598,836**
- First row sample:

```
<ROAD_START> <TAG_UNCLASSIFIED> <X_31> <Y_11>
<MOVE_N_15M> <MOVE_N_5M> <MOVE_N_50M> <MOVE_N_25M> <MOVE_N_5M>
<MOVE_N_10M> <MOVE_N_10M> <MOVE_N_10M> <MOVE_NW_50M> <MOVE_NW_10M>
<MOVE_NW_10M> <MOVE_N_10M> <MOVE_N_15M> <MOVE_N_25M> <MOVE_N_10M>
<MOVE_NE_25M> ...
```

## Artifact paths on Leonardo

```
$WORK/stockholm_poc/data/Stockholm.osm.pbf
$WORK/stockholm_poc/outputs/stockholm_tokens.parquet
$WORK/stockholm_poc/outputs/stockholm_vocab.json
```

## Implications for scaling

- **Global vocabulary no longer scales with area.** Per-chunk anchor
  vocab is capped at `2 * GRID_SIZE` = 512 tokens regardless of chunk
  size. World-scale vocab grows only with the count of
  MOVE/TAG/structural tokens, which are bounded by design.
- **Chunk boundary is the new knob.** The grid resolution inside a
  chunk is `bbox_span / 256`. For Stockholm (~0.6 deg span),
  each cell is ~260 m wide — coarser than H3 res 11 by design, with
  the MOVE tokens carrying sub-cell detail.
- **Next decision:** what defines a "chunk" globally? Candidates are
  S2 cells at a chosen level, H3 parents at a coarse resolution, or
  administrative regions. To be decided before full-planet extraction.

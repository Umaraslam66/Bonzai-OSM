# overture-map

Global vocabulary design for the Bonzai generative map model, driven by empirical frequency analysis of the Overture Maps public dataset.

## What lives here

```
overture-map/
├── README.md                    — this file
├── leonardo.sbatch              — SLURM job for lrd_all_serial (freq passes)
├── leonardo_dedup.sbatch        — SLURM job for lrd_all_serial (dedup)
├── leonardo_dedup_prod.sbatch   — SLURM job for boost_usr_prod (dedup, fast)
├── requirements.txt             — duckdb, pandas, matplotlib, pyarrow, requests
│
├── docs/
│   ├── VOCAB_ANALYSIS.md        — per-field vocab recommendations (auto-regenerated)
│   ├── DATA_SOURCES.md          — external POI/building/address datasets we considered
│   └── MULTI_SOURCE_ANALYSIS.md — Overture + Foursquare + OpenAddresses + dedup outcome
│
├── scripts/
│   ├── common.py                — release pin, S3 config, DuckDB bootstrap
│   ├── 01_schema.py             — dump every Overture theme's Parquet schema
│   ├── 02_freq_pass.py          — per-column global GROUP BY on S3 (streaming, OOM-safe)
│   ├── 04_fsq_download.sh       — download Foursquare OS Places → $CINECA_SCRATCH
│   ├── 05_oa_download.sh        — download OpenAddresses global → $CINECA_SCRATCH
│   ├── 06_fsq_freq_pass.py     — frequency scan on local FSQ parquet
│   ├── 07_oa_inspect.py         — walk OA zip, per-country summary without extraction
│   ├── 08_dedup_places.py       — Overture ∪ Foursquare spatial+name dedup
│   ├── inventory.py             — pretty-print schema + freq inventory
│   └── analyze.py               — read CSVs, write VOCAB_ANALYSIS.md
│
├── schema/                      — DESCRIBE output for all 14 theme/type combos
│   └── *.txt
│
└── outputs/
    └── freq_*.csv               — per-field global distribution (27 files)
```

## Release pinned

`2026-04-15.0` — set in `scripts/common.py` as `RELEASE`.

## Running locally (Mac / any laptop)

```bash
cd overture-map
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
ulimit -n 8192
.venv/bin/python scripts/01_schema.py      # schema dump (fast, projection-only)
.venv/bin/python scripts/02_freq_pass.py   # frequency pass (long — S3 bandwidth bound)
.venv/bin/python scripts/analyze.py        # regenerate docs/VOCAB_ANALYSIS.md
```

`02_freq_pass.py` is idempotent — it skips per-column CSVs that already exist, so you can kill + restart freely.

## Running on Leonardo (CINECA, recommended)

Budget-free `lrd_all_serial` partition — 4 cores / 30.8 GB RAM / 4 h walltime / unlimited submissions.

```bash
# On Mac
rsync -az --partial --exclude=.venv overture-map/ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/overture-map/

# On Leonardo (after `step ssh login` if your 12 h cert expired)
cd /leonardo_work/AIFAC_P02_222/overture-map
module load python/3.11.7
python -m venv .venv
.venv/bin/pip install -r requirements.txt
sbatch leonardo.sbatch                     # queues when compute available
# OR for login-node interactive (I/O bound — survives if short):
nohup .venv/bin/python scripts/02_freq_pass.py > freq.log 2>&1 & disown
```

The script handles `http_retries=10` and streams per column, so it survives transient S3 hiccups and doesn't blow the login-node memory cgroup.

Pull outputs back:
```bash
rsync -az --partial uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/overture-map/outputs/ overture-map/outputs/
```

## Data sizes

Full Overture release `2026-04-15.0` is **608 GB across 972 Parquet files**:

| theme | files | compressed |
|---|---:|---:|
| buildings/building | 512 | 276 GB |
| base/land_cover | 128 | 109 GB |
| transportation/segment | 128 | 70 GB |
| base/land, water, land_use, infrastructure | 112 | 88 GB |
| addresses/address | 32 | 22 GB |
| transportation/connector | 32 | 24 GB |
| places/place | 16 | 10 GB |
| divisions/* | 10 | 6.6 GB |
| others | 2 | 0.6 GB |

We never download the full release. Column-projected scans pull only the column chunks needed — typical egress per freq pass is **1–5 GB per theme**, not the full partition.

## Key findings snapshot

See [`docs/VOCAB_ANALYSIS.md`](docs/VOCAB_ANALYSIS.md) for the full per-field breakdown. Top-line:

| category | distinct | null | recommendation |
|---|---:|---:|---|
| transportation.class | 24 | 0% | keep all — road hierarchy |
| places.basic_category | 267 | 8% | keep all — mid-level POI |
| places.primary / taxonomy | ~2 000 | 4% | trim tail, keep top ~690 |
| buildings.subtype | 13 | **94%** | 94% unlabeled — use context to infer |
| base.land_cover.subtype | 10 | 0% | keep all |
| roof/facade materials | 11–14 | **99.7%+** | drop |
| places.operating_status | 3 | — | drop (99.99% "open") |
| places.brand_wikidata | 3 040 | **98%** | collapse to `is_branded` bit |

Estimated final vocab size: **~1 200 tokens** for a balanced first-cut.

## Dedup outcome (2026-04-29)

`scripts/08_dedup_places.py` cross-matched Overture ∪ Foursquare worldwide on Leonardo `boost_usr_prod` (32 cores, 200 GB RAM). Headline:

| | Overture | Foursquare | Merged |
|---|---:|---:|---:|
| total places | 75.5 M | 99.9 M | **127.1 M** |
| matched to other source | 32.6 M (43%) | 48.3 M (48%) | — |
| unique to source | 42.9 M | 51.7 M | — |

Foursquare adds **+52 M unique POIs** (68 % growth over Overture-alone), concentrated in Asia / LatAm / Eastern Europe. See [`docs/MULTI_SOURCE_ANALYSIS.md § Dedup outcome`](docs/MULTI_SOURCE_ANALYSIS.md#dedup-outcome--overture--foursquare).

## Next steps

See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) and the [next-actions list](docs/MULTI_SOURCE_ANALYSIS.md#next-concrete-actions-priority-order):

1. Build the unified POI master table from `dedup_fsq_decisions.parquet` (~127 M rows).
2. Re-run frequency passes on the merged universe → re-derive POI vocab on the FSQ hierarchical taxonomy.
3. Overture Bridge Files → OSM tag join to recover building semantic labels (94 % null in Overture).
4. (Optional) Per-region OpenAddresses integration for street-number tokens.

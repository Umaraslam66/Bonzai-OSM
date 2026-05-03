# Scripts

## `prepare_tiles_local.py`

CLI entrypoint. Two subcommands:

- `synthetic` — generate procedural tiles for smoke tests.
- `overture-region` — generate real OSM tiles for a bbox.

Examples:

```bash
# 100 synthetic tiles
.venv/bin/python scripts/prepare_tiles_local.py synthetic -o /tmp/bonzai-syn -n 100

# Singapore (~150 real tiles, full island)
.venv/bin/python scripts/prepare_tiles_local.py overture-region \
    --pbf data/malaysia-singapore-brunei-latest.osm.pbf \
    --sw-lat 1.20 --sw-lon 103.60 \
    --ne-lat 1.48 --ne-lon 104.05 \
    -o /tmp/bonzai-sg \
    --country SG --koppen Af
```

Note: with two registered subcommands (synthetic + overture-region), the subcommand name is required at the CLI.

## `leonardo_data_prep.sbatch`

SLURM job template for the free `lrd_all_serial` partition. Set env-vars before submission. Examples for the three Phase 0a countries:

```bash
# Singapore (smallest, fastest — but expect heavy skip-rate due to node-cap)
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/malaysia-singapore-brunei-latest.osm.pbf
export BONZAI_SW_LAT=1.20 BONZAI_SW_LON=103.60
export BONZAI_NE_LAT=1.48 BONZAI_NE_LON=104.05
export BONZAI_COUNTRY=SG BONZAI_KOPPEN=Af
export BONZAI_OUT=$WORK/bonzai-tiles/singapore
export BONZAI_MAX_TILES=300
sbatch scripts/leonardo_data_prep.sbatch

# Sri Lanka
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/sri-lanka-latest.osm.pbf
export BONZAI_SW_LAT=5.85 BONZAI_SW_LON=79.55
export BONZAI_NE_LAT=9.90 BONZAI_NE_LON=81.95
export BONZAI_COUNTRY=LK BONZAI_KOPPEN=Aw
export BONZAI_OUT=$WORK/bonzai-tiles/sri_lanka
export BONZAI_MAX_TILES=2000
sbatch scripts/leonardo_data_prep.sbatch

# Sweden
export BONZAI_PBF=$CINECA_SCRATCH/osm/raw/sweden-latest.osm.pbf
export BONZAI_SW_LAT=55.0 BONZAI_SW_LON=10.5
export BONZAI_NE_LAT=69.5 BONZAI_NE_LON=24.5
export BONZAI_COUNTRY=SE BONZAI_KOPPEN=Cfb
export BONZAI_OUT=$WORK/bonzai-tiles/sweden
export BONZAI_MAX_TILES=5000
sbatch scripts/leonardo_data_prep.sbatch
```

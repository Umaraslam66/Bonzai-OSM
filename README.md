# Bonzai-OSM

World-scale OpenStreetMap ingestion and preprocessing for a generative "map LLM" workflow on CINECA Leonardo.

## Current Status

- Project/account on Leonardo: `AIFAC_P02_222`
- Budget on 2026-04-15: `14 / 40000` local core-hours consumed
- Cleanup completed on 2026-04-15:
  - `$WORK` reset to ~`1.4M`
  - `$FAST` reset to `0k`
- Planet download completed on 2026-04-15 via Leonardo datamover:
  - file: `/leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf`
  - upstream snapshot resolved to: `planet-260406.osm.pbf`
  - size: `92,239,256,545` bytes
  - checksum: `planet-latest.osm.pbf: OK`
- Available Leonardo modules currently confirmed:
  - `gdal/3.8.5--gcc--12.2.0`
  - `proj/9.2.1--gcc--12.2.0-spack0.22`
  - no `osmium` module found so far

## Repo Layout

- [PROJECT.md](./PROJECT.md): project log, decisions, verified Leonardo state
- [commands.md](./commands.md): copy-paste operational commands
- [scripts/leonardo_cleanup.sh](./scripts/leonardo_cleanup.sh): safe cleanup helper
- [scripts/leonardo_download_planet.sh](./scripts/leonardo_download_planet.sh): datamover-based planet download helper
- [jobs/luxembourg_probe.sbatch](./jobs/luxembourg_probe.sbatch): free `lrd_all_serial` GDAL probe job
- [jobs/leonardo_osm_extract.sbatch](./jobs/leonardo_osm_extract.sbatch): future `osmium` extract template once `osmium` is installed or built

## Operational Rules

- Use `data.leonardo.cineca.it` for large downloads.
- Keep raw `.osm.pbf` files on `$CINECA_SCRATCH`.
- Use `lrd_all_serial` for preprocessing and extraction.
- Do not use `dcgp_usr_prod` or `boost_usr_prod` for PBF download or light parsing.
- Move durable outputs to `$WORK` only after they are stable and worth keeping.

## Next Step

Prototype the parsing pipeline on a small Geofabrik extract such as Luxembourg or Iceland using GDAL on `lrd_all_serial`. Do not start full-planet custom extraction until the region probe succeeds or `osmium` is installed.

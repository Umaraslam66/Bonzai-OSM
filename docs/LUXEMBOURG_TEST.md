# Luxembourg Test Report

End-to-end validation of the first budget-free OSM parsing workflow on Leonardo.

Date: `2026-04-15`
Account: `AIFAC_P02_222`
User: `uaslam00`

## Purpose

Validate that we can:

1. keep the full planet on `$CINECA_SCRATCH`
2. avoid burning paid compute for preprocessing
3. parse `.osm.pbf` successfully on Leonardo with currently available modules
4. extract a first usable artifact from OSM on the free `lrd_all_serial` partition

## Starting State

Before the Luxembourg test:

- legacy project data was removed
- `$WORK` was reduced to about `1.4M`
- `$FAST` was reduced to `0k`
- world planet file was already downloaded and verified:
  - `/leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf`
  - upstream redirect resolved to `planet-260406.osm.pbf`
  - `md5sum -c planet-latest.osm.pbf.md5` returned `OK`

Relevant quota snapshot:

```text
/leonardo_scratch/large/userexternal/uaslam00  85.95G
/leonardo_work/AIFAC_P02_222                   1.438M
/leonardo_scratch/fast/AIFAC_P02_222           0k
```

## Key Finding

Leonardo currently provides a working GDAL OSM driver, but no confirmed `osmium` module for this account.

Verified module probe:

```bash
module load gdal/3.8.5--gcc--12.2.0
ogrinfo --formats | grep -i OSM
```

Observed output:

```text
OSM -vector- (rov): OpenStreetMap XML and PBF
```

This changed the short-term plan:

- use GDAL first for region-scale prototyping
- keep `osmium` as a future option if we install/build it later

## Step 1: Download Luxembourg Extract

Command used on Leonardo:

```bash
ssh -xt "$USER"@data.leonardo.cineca.it wget --continue -P /leonardo_scratch/large/userexternal/uaslam00/osm/raw https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf
```

Observed behavior:

- Geofabrik redirected to `luxembourg-260414.osm.pbf`
- download completed successfully
- final destination:
  `/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf`
- final size from `wget`: `46,493,508` bytes (`44.34M`)

## Step 2: Submit Free Probe Job

Job file:

```bash
#!/bin/bash
#SBATCH --job-name=lux-probe
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gdal/3.8.5--gcc--12.2.0

RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf

ogrinfo "$RAW" -so points
ogrinfo "$RAW" -so lines
ogrinfo "$RAW" -so multipolygons
```

Submission command:

```bash
sbatch luxembourg_probe.sbatch
```

Observed Slurm result:

```text
Submitted batch job 39908193
```

## Step 3: Probe Result

Result summary:

- job finished successfully
- `lux-probe-39908193.err` was empty
- GDAL opened the Luxembourg `.osm.pbf` successfully three times
- the following layers were confirmed:
  - `points`
  - `lines`
  - `multipolygons`

Important observed output:

```text
INFO: Open of `/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf'
      using driver `OSM' successful.
```

Schema confirmed:

- `points` includes fields such as `osm_id`, `name`, `highway`, `place`, `man_made`
- `lines` includes fields such as `osm_id`, `name`, `highway`, `waterway`, `railway`, `z_order`
- `multipolygons` includes fields such as `osm_id`, `building`, `landuse`, `natural`, `amenity`, `tourism`

This proves GDAL is sufficient for a first structured extraction workflow on Leonardo.

## Step 4: Extract Luxembourg Roads

Job file:

```bash
#!/bin/bash
#SBATCH --job-name=lux-roads
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gdal/3.8.5--gcc--12.2.0

RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf
OUT=/leonardo_scratch/large/userexternal/uaslam00/osm/extracts/luxembourg_roads.geojson

mkdir -p /leonardo_scratch/large/userexternal/uaslam00/osm/extracts

ogr2ogr -f GeoJSON "$OUT" "$RAW" lines -where "highway IS NOT NULL"

ls -lh "$OUT"
```

Final successful submission:

```text
Submitted batch job 39908360
```

Observed output:

```text
0...10...20...30...40...50...60...70...80...90...100 - done.
-rw-r--r--. 1 uaslam00 interactive 63M Apr 15 19:37 /leonardo_scratch/large/userexternal/uaslam00/osm/extracts/luxembourg_roads.geojson
```

Artifact created:

```text
/leonardo_scratch/large/userexternal/uaslam00/osm/extracts/luxembourg_roads.geojson
```

Artifact size:

```text
63M
```

## What Worked

- large raw planet download on the free Leonardo datamover path
- checksum verification on the login node
- Geofabrik regional extract download to `$CINECA_SCRATCH`
- GDAL OSM driver on Leonardo
- `lrd_all_serial` for free parsing and extraction
- first real GeoJSON artifact produced from OSM

## What Failed or Needed Correction

1. Datamover download commands were initially fragile when typed manually with split paths.
   Fix:
   use absolute `-O /full/path/file.osm.pbf` form when possible.

2. `osmium` is not currently available as a module.
   Fix:
   switch to GDAL-first region testing.

3. One early road-extraction job used a broken `OUT=` path with an accidental space.
   Fix:
   corrected script and resubmitted successfully as job `39908360`.

## Operational Conclusions

- The free preprocessing strategy is working.
- We do not need to touch `boost_usr_prod` or `dcgp_usr_prod` for this stage.
- The right next artifacts to extract are:
  - buildings from `multipolygons`
  - POIs from `points`
- GeoJSON is acceptable for validation, but not ideal for large-scale training pipelines.
- For the real dataset build, we should likely move toward:
  - GeoParquet
  - newline-delimited JSON
  - another compact structured intermediate format

## Recommended Next Commands

Download Iceland:

```bash
ssh -xt "$USER"@data.leonardo.cineca.it wget --continue -P /leonardo_scratch/large/userexternal/uaslam00/osm/raw https://download.geofabrik.de/europe/iceland-latest.osm.pbf
```

Extract buildings from Luxembourg:

```bash
printf '%s\n' '#!/bin/bash' '#SBATCH --job-name=lux-buildings' '#SBATCH --partition=lrd_all_serial' '#SBATCH --account=AIFAC_P02_222' '#SBATCH --time=01:00:00' '#SBATCH --cpus-per-task=1' '#SBATCH --mem=12G' '#SBATCH --output=%x-%j.out' '#SBATCH --error=%x-%j.err' '' 'set -euo pipefail' '' 'module load gdal/3.8.5--gcc--12.2.0' '' 'RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf' 'OUT=/leonardo_scratch/large/userexternal/uaslam00/osm/extracts/luxembourg_buildings.geojson' '' 'mkdir -p /leonardo_scratch/large/userexternal/uaslam00/osm/extracts' '' 'ogr2ogr -f GeoJSON "$OUT" "$RAW" multipolygons -where "building IS NOT NULL"' '' 'ls -lh "$OUT"' > luxembourg_buildings.sbatch
sbatch luxembourg_buildings.sbatch
```

Extract POIs from Luxembourg:

```bash
printf '%s\n' '#!/bin/bash' '#SBATCH --job-name=lux-pois' '#SBATCH --partition=lrd_all_serial' '#SBATCH --account=AIFAC_P02_222' '#SBATCH --time=01:00:00' '#SBATCH --cpus-per-task=1' '#SBATCH --mem=12G' '#SBATCH --output=%x-%j.out' '#SBATCH --error=%x-%j.err' '' 'set -euo pipefail' '' 'module load gdal/3.8.5--gcc--12.2.0' '' 'RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf' 'OUT=/leonardo_scratch/large/userexternal/uaslam00/osm/extracts/luxembourg_pois.geojson' '' 'mkdir -p /leonardo_scratch/large/userexternal/uaslam00/osm/extracts' '' 'ogr2ogr -f GeoJSON "$OUT" "$RAW" points -where "name IS NOT NULL OR amenity IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL"' '' 'ls -lh "$OUT"' > luxembourg_pois.sbatch
sbatch luxembourg_pois.sbatch
```

## Bottom Line

The Luxembourg test is a success. We now have a verified, budget-free OSM preprocessing path on Leonardo using:

- datamover for downloads
- `$CINECA_SCRATCH` for raw and intermediate storage
- GDAL for `.osm.pbf` parsing
- `lrd_all_serial` for free extraction jobs

This is enough to continue prototyping without touching paid compute.

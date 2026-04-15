#!/bin/bash
#SBATCH --job-name=lux-gpkg
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gdal/3.8.5--gcc--12.2.0

RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf
OUTDIR=/leonardo_scratch/large/userexternal/uaslam00/osm/benchmarks/luxembourg/gpkg
OUT="${OUTDIR}/luxembourg_full.gpkg"
SUMMARY="${OUTDIR}/summary.txt"

mkdir -p "${OUTDIR}"
rm -f "${OUT}" "${SUMMARY}"

start=$(date +%s)
ogr2ogr -f GPKG "${OUT}" "${RAW}" points
ogr2ogr -f GPKG -update -append "${OUT}" "${RAW}" lines -nln lines
ogr2ogr -f GPKG -update -append "${OUT}" "${RAW}" multipolygons -nln multipolygons
end=$(date +%s)

printf 'gpkg\t%s\t%ss\n' "$(du -h "${OUT}" | awk '{print $1}')" "$((end-start))" > "${SUMMARY}"
ogrinfo "${OUT}" points -so >> "${SUMMARY}"
ogrinfo "${OUT}" lines -so >> "${SUMMARY}"
ogrinfo "${OUT}" multipolygons -so >> "${SUMMARY}"
cat "${SUMMARY}"

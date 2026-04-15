#!/bin/bash
#SBATCH --job-name=lux-gjs
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
OUTDIR=/leonardo_scratch/large/userexternal/uaslam00/osm/benchmarks/luxembourg/geojsonseq
SUMMARY="${OUTDIR}/summary.txt"

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}/points.geojsons" "${OUTDIR}/lines.geojsons" "${OUTDIR}/multipolygons.geojsons" "${SUMMARY}"

run_export() {
  local layer="$1"
  local outfile="${OUTDIR}/${layer}.geojsons"
  local start end
  start=$(date +%s)
  ogr2ogr -f GeoJSONSeq "${outfile}" "${RAW}" "${layer}"
  end=$(date +%s)
  printf '%s\t%s\t%ss\n' "${layer}" "$(du -h "${outfile}" | awk '{print $1}')" "$((end-start))" >> "${SUMMARY}"
}

run_export points
run_export lines
run_export multipolygons

printf '\n' >> "${SUMMARY}"
du -sh "${OUTDIR}" >> "${SUMMARY}"
cat "${SUMMARY}"

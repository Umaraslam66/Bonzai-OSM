#!/bin/bash
#SBATCH --job-name=lux-bundle
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gdal/3.8.5--gcc--12.2.0

REPO_ROOT="${WORK:-/leonardo_work/AIFAC_P02_222}/bonzai-osm"
RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf
OUTDIR=/leonardo_scratch/large/userexternal/uaslam00/osm/summaries
JSON_OUT="${OUTDIR}/luxembourg_summary.json"
REPORT_OUT="${OUTDIR}/luxembourg_summary_report.md"

mkdir -p "${OUTDIR}"

python3 "${REPO_ROOT}/scripts/osm_layer_summary.py" \
  --input "${RAW}" \
  --output "${JSON_OUT}" \
  --sample-limit 5

python3 "${REPO_ROOT}/scripts/render_osm_summary_report.py" \
  --input "${JSON_OUT}" \
  --output "${REPORT_OUT}"

ls -lh "${JSON_OUT}" "${REPORT_OUT}"
head -80 "${REPORT_OUT}"

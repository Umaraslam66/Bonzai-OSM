#!/bin/bash
#SBATCH --job-name=lux-summary
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
OUT=/leonardo_scratch/large/userexternal/uaslam00/osm/summaries/luxembourg_summary.json

mkdir -p /leonardo_scratch/large/userexternal/uaslam00/osm/summaries

python3 "${REPO_ROOT}/scripts/osm_layer_summary.py" \
  --input "${RAW}" \
  --output "${OUT}" \
  --sample-limit 5

ls -lh "${OUT}"
head -120 "${OUT}"

#!/bin/bash
#SBATCH --job-name=osm-extract
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ACCOUNT="${ACCOUNT:-AIFAC_P02_222}"
WORK_DIR="${WORK:-/leonardo_work/${ACCOUNT}}"
FAST_DIR="${FAST:-/leonardo_scratch/fast/${ACCOUNT}}"
SCRATCH_DIR="${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}"
OSM_MODULE="${OSM_MODULE:-osmium-tool}"
RAW_PBF="${RAW_PBF:-${SCRATCH_DIR}/osm/raw/planet-latest.osm.pbf}"
REGION_CONFIG="${REGION_CONFIG:-${WORK_DIR}/bonzai-osm/jobs/regions.conf}"
OUT_DIR="${OUT_DIR:-${SCRATCH_DIR}/osm/extracts}"

module load "${OSM_MODULE}"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

osmium --version
osmium fileinfo "${RAW_PBF}"
osmium extract \
  --config "${REGION_CONFIG}" \
  --strategy=simple \
  --overwrite \
  "${RAW_PBF}"

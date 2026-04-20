#!/bin/bash
# slurm_eda_free.sh
#
# Budget-free variant of slurm_eda.sh. Runs the identical EDA pipeline
# on `lrd_all_serial` (4 cores, 30 GB RAM, 4 h walltime, unlimited
# submissions, excluded from the project's core-hour budget).
#
# Why this is the recommended path: global tag counting needs kilobytes
# of memory and scales with O(unique-values) ~ 10^5, not O(file size).
# The 30 GB cap is never a constraint for this job. pyosmium is single-
# threaded by design, so the 4-core limit costs nothing either.
#
# Submit:
#   sbatch stockholm_poc/slurm_eda_free.sh

#SBATCH --job-name=global-osm-eda-free
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} partition=lrd_all_serial"
echo "[slurm] start: $(date -u +%FT%TZ)"

: "${WORK:?WORK is not set}"
: "${CINECA_SCRATCH:?CINECA_SCRATCH is not set}"

WORKSPACE="${WORK}/stockholm_poc"
SCRIPT_DIR="${WORKSPACE}/scripts"
VENV="${WORKSPACE}/venv"

PBF="${CINECA_SCRATCH}/osm/raw/planet-latest.osm.pbf"
EDA_DIR="${WORK}/eda"
OUT_DIR="${EDA_DIR}/outputs"
INTER_DIR="${EDA_DIR}/intermediate"

if [[ ! -f "${PBF}" ]]; then
  echo "[slurm] ERROR: planet PBF not found at ${PBF}" >&2
  exit 2
fi
if [[ ! -d "${VENV}" ]]; then
  echo "[slurm] ERROR: missing venv at ${VENV}" >&2
  exit 2
fi

module purge || true
if module avail 2>&1 | grep -qE '^java/'; then
  module load "$(module avail 2>&1 | grep -oE 'java/[^ (]*' | head -n1)"
fi
if module avail 2>&1 | grep -qE 'python/3\.11'; then
  module load "$(module avail 2>&1 | grep -oE 'python/3\.11[^ (]*' | head -n1)"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python --version
java -version 2>&1 | head -n1 || true

mkdir -p "${OUT_DIR}" "${INTER_DIR}"

cd "${WORKSPACE}"

# Tighter driver-memory for the 30 GB partition. Stage 2 is genuinely
# modest — aggregating ~10^5 unique tag values per key.
python "${SCRIPT_DIR}/global_osm_eda.py" \
  --input "${PBF}" \
  --output-dir "${OUT_DIR}" \
  --intermediate-dir "${INTER_DIR}" \
  --spark-driver-memory 12g \
  --spark-shuffle-partitions 8 \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -lh "${OUT_DIR}"

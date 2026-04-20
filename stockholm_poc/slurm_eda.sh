#!/bin/bash
# slurm_eda.sh
#
# Run global_osm_eda.py on the CINECA Leonardo CPU partition
# (dcgp_usr_prod: 112 cores, 512 GB RAM, CPU-only). This follows the
# project spec of "1 standard compute node, maximum RAM and cores for
# the Spark application."
#
# BUDGET WARNING:
#   dcgp_usr_prod bills at 112 core-hours per node-hour. A 3 h run on
#   one node therefore burns ~336 core-hours (~0.85% of our 40,000
#   core-hour allocation). Most of that spend is wasted on this task
#   specifically, because stage 1 (pyosmium PBF scan) is single-
#   threaded by design — 110 of the 112 cores sit idle for the bulk
#   of the job.
#
#   If you want the same output for ZERO core-hours, submit the sibling
#   script `slurm_eda_free.sh`, which targets `lrd_all_serial` (4
#   cores, 30 GB RAM, 4 h walltime, budget-free, unlimited submissions).
#   Tag counting needs kilobytes of memory, not gigabytes, so the 30 GB
#   cap is never a constraint — the free path is strictly recommended
#   unless you have a specific reason to use the paid partition.
#
# Submit:
#   sbatch stockholm_poc/slurm_eda.sh

#SBATCH --job-name=global-osm-eda
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=490G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[slurm] cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "[slurm] start: $(date -u +%FT%TZ)"

: "${WORK:?WORK is not set}"
: "${CINECA_SCRATCH:?CINECA_SCRATCH is not set}"

WORKSPACE="${WORK}/stockholm_poc"
SCRIPT_DIR="${WORKSPACE}/scripts"
VENV="${WORKSPACE}/venv"

# Inputs / outputs.
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

# PySpark needs a JVM; pyosmium needs Python 3.11.
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

# Spark driver-memory roughly half of SBATCH --mem, leaving headroom
# for pyosmium's parse and OS caches. Stage 1 never goes near this.
python "${SCRIPT_DIR}/global_osm_eda.py" \
  --input "${PBF}" \
  --output-dir "${OUT_DIR}" \
  --intermediate-dir "${INTER_DIR}" \
  --spark-driver-memory 240g \
  --spark-shuffle-partitions 32 \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -lh "${OUT_DIR}"

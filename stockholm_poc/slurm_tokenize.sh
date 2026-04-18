#!/bin/bash
# slurm_tokenize.sh
#
# Run tokenize_stockholm.py on a single Leonardo CPU node with 128 GB RAM.
#
# Partition choice:
#   - instruct.md requires "1 CPU node" with "at least 128GB RAM", which
#     rules out `lrd_all_serial` (4 cores, 30.8 GB).
#   - We use `dcgp_usr_prod`: CPU-only, 112 cores/node, 512 GB/node,
#     24 h walltime. Cost: 112 core-hours per node-hour — this job is
#     short so the burn stays small (see the README for the math), but
#     be aware the budget *is* being touched here, unlike every other
#     free preprocessing job in this repo.
#
# For smoke-testing the pipeline on Stockholm, a much cheaper run is
# possible on the free `lrd_all_serial` partition: see the sibling
# script `slurm_tokenize_serial.sh` if you add one (Stockholm fits in
# 30 GB). This script exists to satisfy the spec as written.
#
# Submit:
#   sbatch stockholm_poc/slurm_tokenize.sh
#
# Inputs expected on disk before submit:
#   $WORK/stockholm_poc/data/Stockholm.osm.pbf
#   $WORK/stockholm_poc/venv/      (python venv with requirements.txt)

#SBATCH --job-name=stockholm-tokenize
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=dcgp_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "[slurm] start: $(date -u +%FT%TZ)"

: "${WORK:?WORK is not set}"

WORKSPACE="${WORK}/stockholm_poc"
SCRIPT_DIR="${WORKSPACE}/scripts"
DATA_PBF="${WORKSPACE}/data/Stockholm.osm.pbf"
OUT_DIR="${WORKSPACE}/outputs"
VENV="${WORKSPACE}/venv"

if [[ ! -f "${DATA_PBF}" ]]; then
  echo "[slurm] ERROR: missing input PBF at ${DATA_PBF}" >&2
  echo "[slurm] run setup_workspace.sh on the login node first" >&2
  exit 2
fi

if [[ ! -d "${VENV}" ]]; then
  echo "[slurm] ERROR: missing Python venv at ${VENV}" >&2
  echo "[slurm] see stockholm_poc/README.md for venv creation commands" >&2
  exit 2
fi

# The Python interpreter inside the venv has to find its shared libs.
# Loading the matching python module (if needed) keeps libc compat sane.
# Adjust the module name once we pin one on Leonardo — pyrosm's wheels
# are CPython 3.8 – 3.11 only, so 3.11 is the current sweet spot.
module purge || true
if module avail 2>&1 | grep -qE 'python/3\.11'; then
  module load "$(module avail 2>&1 | grep -oE 'python/3\.11[^ ]*' | head -n1)"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python --version

mkdir -p "${OUT_DIR}"

python "${SCRIPT_DIR}/tokenize_stockholm.py" \
  --input  "${DATA_PBF}" \
  --output-dir "${OUT_DIR}" \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -la "${OUT_DIR}"

#!/bin/bash
# slurm_tokenize_serial.sh
#
# Budget-free variant of slurm_tokenize.sh, targeting the lrd_all_serial
# partition (4 cores, 30.8 GB RAM, 4 h walltime, unlimited submissions,
# excluded from the project's core-hour budget).
#
# Stockholm has on the order of 1e5–1e6 objects and the tokenizer never
# materialises more than a few hundred MB in memory, so 30 GB is plenty
# for the PoC.
#
# Submit:
#   sbatch stockholm_poc/slurm_tokenize_serial.sh

#SBATCH --job-name=stockholm-tokenize-serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=28G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} partition=lrd_all_serial"
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

module purge || true
if module avail 2>&1 | grep -qE 'python/3\.11'; then
  module load "$(module avail 2>&1 | grep -oE 'python/3\.11[^ (]*' | head -n1)"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python --version

mkdir -p "${OUT_DIR}"

python "${SCRIPT_DIR}/tokenize_stockholm.py" \
  --input  "${DATA_PBF}" \
  --output-dir "${OUT_DIR}" \
  --region EUROPE \
  --climate TEMPERATE \
  --density URBAN \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -la "${OUT_DIR}"

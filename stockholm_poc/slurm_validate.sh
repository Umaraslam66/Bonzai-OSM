#!/bin/bash
# slurm_validate.sh
#
# Run validate_stockholm.py against the trained checkpoint.
# Uses the same GPU partition because generating thousands of sequences
# is fast with CUDA (a few minutes) vs painful on CPU (~1 h).
#
# Budget: typically finishes in ~5 minutes on 1 x A100, i.e. ~1 core-h.
#
# Submit:
#   sbatch stockholm_poc/slurm_validate.sh

#SBATCH --job-name=stockholm-validate
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[slurm] start: $(date -u +%FT%TZ)"

: "${WORK:?WORK is not set}"

WORKSPACE="${WORK}/stockholm_poc"
SCRIPT_DIR="${WORKSPACE}/scripts"
MODEL_DIR="${WORKSPACE}/checkpoints/stockholm_overfit/final"
PARQUET="${WORKSPACE}/outputs/stockholm_tokens.parquet"
REPORT_DIR="${WORKSPACE}/validation"
VENV="${WORKSPACE}/venv"

if [[ ! -f "${MODEL_DIR}/model.safetensors" ]]; then
  echo "[slurm] ERROR: missing trained model at ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_DIR}/token_to_id.json" ]]; then
  echo "[slurm] ERROR: missing token_to_id.json at ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -f "${PARQUET}" ]]; then
  echo "[slurm] ERROR: missing parquet at ${PARQUET}" >&2
  exit 2
fi

module purge || true
if module avail 2>&1 | grep -qE '^cuda/12'; then
  module load "$(module avail 2>&1 | grep -oE 'cuda/12[^ (]*' | head -n1)"
fi
if module avail 2>&1 | grep -qE 'python/3\.11'; then
  module load "$(module avail 2>&1 | grep -oE 'python/3\.11[^ (]*' | head -n1)"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python --version
python - <<'PY'
import torch
print(f"[slurm] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[slurm] device0={torch.cuda.get_device_name(0)}")
PY

mkdir -p "${REPORT_DIR}"

cd "${WORKSPACE}"

python "${SCRIPT_DIR}/validate_stockholm.py" \
  --model-dir "${MODEL_DIR}" \
  --parquet "${PARQUET}" \
  --output-dir "${REPORT_DIR}" \
  --n-per-kind 200 \
  --max-new-tokens 256 \
  --temperature 0.9 \
  --top-p 0.95 \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -lh "${REPORT_DIR}"

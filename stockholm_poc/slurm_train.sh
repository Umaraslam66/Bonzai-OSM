#!/bin/bash
# slurm_train.sh
#
# Overfit-test training run for train_stockholm.py on the Leonardo
# Booster GPU partition. One A100 GPU, 8 CPU cores, 64 GB RAM, 2 h
# walltime.
#
# Budget note:
#   boost_usr_prod bills 32 core-hours per node-hour, and a single GPU
#   is ~1/4 of a node. A 2 h run therefore burns roughly
#     2 h * 32 core-h/node-h * 0.25 node = 16 core-hours
#   i.e. ~0.04% of the 40,000 core-hour allocation. Not free, but tiny.
#
# Submit:
#   sbatch stockholm_poc/slurm_train.sh
#
# Pre-flight requirements (done on a login node):
#   1. Tokenization already run -> $WORK/stockholm_poc/outputs/*.parquet + *.json
#   2. Training deps installed into $WORK/stockholm_poc/venv:
#        source $WORK/stockholm_poc/venv/bin/activate
#        pip install -r $WORK/Bonzai-OSM/stockholm_poc/requirements_train.txt
#        deactivate

#SBATCH --job-name=stockholm-train
#SBATCH --account=AIFAC_P02_222
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[slurm] host=$(hostname) jobid=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[slurm] cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} gpus=${SLURM_GPUS_ON_NODE:-?}"
echo "[slurm] start: $(date -u +%FT%TZ)"

: "${WORK:?WORK is not set}"

WORKSPACE="${WORK}/stockholm_poc"
SCRIPT_DIR="${WORKSPACE}/scripts"
PARQUET="${WORKSPACE}/outputs/stockholm_tokens.parquet"
VOCAB="${WORKSPACE}/outputs/stockholm_vocab.json"
CKPT_DIR="${WORKSPACE}/checkpoints/stockholm_overfit"
VENV="${WORKSPACE}/venv"

if [[ ! -f "${PARQUET}" ]]; then
  echo "[slurm] ERROR: missing parquet at ${PARQUET}" >&2
  exit 2
fi
if [[ ! -f "${VOCAB}" ]]; then
  echo "[slurm] ERROR: missing vocab at ${VOCAB}" >&2
  exit 2
fi
if [[ ! -d "${VENV}" ]]; then
  echo "[slurm] ERROR: missing venv at ${VENV}" >&2
  exit 2
fi

# Load CUDA + Python. `module avail` on Leonardo shows cuda/12.3 and
# python/3.11.7 as the current default toolchain; both are bundled with
# the torch>=2.3 wheels we install in the venv.
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

mkdir -p "${CKPT_DIR}"

cd "${WORKSPACE}"

python "${SCRIPT_DIR}/train_stockholm.py" \
  --parquet "${PARQUET}" \
  --vocab "${VOCAB}" \
  --output-dir "${CKPT_DIR}" \
  --block-size 2048 \
  --n-layer 12 --n-head 12 --n-embd 768 \
  --batch-size 16 --eval-batch-size 16 \
  --epochs 40 \
  --val-fraction 0.1 \
  --learning-rate 3e-4 \
  --warmup-steps 200 \
  --logging-steps 50 \
  --bf16 \
  --log-level INFO

echo "[slurm] end: $(date -u +%FT%TZ)"
ls -lh "${CKPT_DIR}"

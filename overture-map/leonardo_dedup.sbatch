#!/bin/bash
#SBATCH --job-name=overture-dedup
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load profile/base
module load python/3.11.6--gcc--8.5.0 2>/dev/null \
  || module load python/3.11.7--gcc--12.2.0 2>/dev/null \
  || module load python/3.11 2>/dev/null \
  || module load python

python --version

if [ ! -d .venv ]; then
  python -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

ulimit -n 8192
.venv/bin/python scripts/08_dedup_places.py

echo "== done =="
ls -la outputs/dedup_*

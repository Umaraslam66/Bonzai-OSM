#!/bin/bash
#SBATCH --job-name=overture-master
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
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

ulimit -n 8192
.venv/bin/python scripts/09_build_master.py

echo "== done =="
ls -la outputs/poi_master.parquet

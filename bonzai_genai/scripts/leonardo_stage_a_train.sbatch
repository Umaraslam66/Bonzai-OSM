#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-stage-a
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"
export BONZAI_STAGE=stage_a
python scripts/_train_runner.py

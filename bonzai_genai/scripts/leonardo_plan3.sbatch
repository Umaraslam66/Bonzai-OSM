#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-plan3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Plan 3 driver -- 4-GPU DDP training for Painter or Writer.
# Required env-vars (set by the caller, e.g. scripts/run_plan3.py):
#   BONZAI_STAGE                  "stage_a" | "stage_b"
#   BONZAI_PRESET                 usually "plan3"
#   BONZAI_TRAIN_URL              WebDataset shard glob (all 3 countries)
#   BONZAI_VAL_URL                held-out shard glob
#   BONZAI_OUT                    checkpoint + log dir on $WORK
#   BONZAI_BATCH_SIZE             per-rank batch size (effective = 4x this)
#   BONZAI_MAX_EPOCHS             50 for full Plan 3; 1 for smoke
#   BONZAI_VAE_CKPT               (stage_a only) frozen VAE checkpoint path
#   BONZAI_COUNTRY_WEIGHTS_JSON   JSON dict country -> sampling weight

set -euo pipefail
mkdir -p logs
source "$WORK/bonzai_genai/.venv/bin/activate"
cd "$WORK/bonzai_genai"

# Force DDP across all 4 A100s -- boost_usr_prod bills per node, so we
# saturate the node we already pay for. See feedback_leonardo_full_node.md.
export BONZAI_DDP_DEVICES=4

srun python scripts/_train_runner.py

#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bonzai-data-prep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Required env-vars (set before sbatch):
#   BONZAI_PBF       — full path to the .osm.pbf
#   BONZAI_SW_LAT    — SW corner lat
#   BONZAI_SW_LON    — SW corner lon
#   BONZAI_NE_LAT    — NE corner lat
#   BONZAI_NE_LON    — NE corner lon
#   BONZAI_COUNTRY   — ISO country code (e.g. SE, SG, LK)
#   BONZAI_KOPPEN    — Köppen group (e.g. Cfb, Af, Aw)
#   BONZAI_OUT       — output dir on $WORK
#   BONZAI_MAX_TILES — optional; default 5000

set -euo pipefail

mkdir -p logs

# Activate venv on Leonardo
source "$WORK/bonzai_genai/.venv/bin/activate"

cd "$WORK/bonzai_genai"

mkdir -p "$BONZAI_OUT"

python scripts/prepare_tiles_local.py overture-region \
    --pbf "$BONZAI_PBF" \
    --sw-lat "$BONZAI_SW_LAT" --sw-lon "$BONZAI_SW_LON" \
    --ne-lat "$BONZAI_NE_LAT" --ne-lon "$BONZAI_NE_LON" \
    -o "$BONZAI_OUT" \
    --country "$BONZAI_COUNTRY" --koppen "$BONZAI_KOPPEN" \
    --shard-size 100 \
    --max-tiles "${BONZAI_MAX_TILES:-5000}"

echo "Job complete; manifest:"
cat "$BONZAI_OUT/manifest.json"

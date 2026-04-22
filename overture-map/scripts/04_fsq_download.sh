#!/usr/bin/env bash
# Download the latest Foursquare Open Source Places release to $CINECA_SCRATCH.
#
# Requires:
#   - venv with huggingface_hub installed (pip install -r requirements.txt)
#   - HF_TOKEN env var OR $HOME/.hf_token file with the token string
#
# Usage:
#   ./scripts/04_fsq_download.sh                      # downloads to default path
#   FSQ_RELEASE=dt=2026-04-14 ./scripts/04_fsq_download.sh

set -euo pipefail

: "${CINECA_SCRATCH:?CINECA_SCRATCH must be set (it auto-exports on Leonardo login)}"

FSQ_RELEASE="${FSQ_RELEASE:-dt=2026-04-14}"
DEST="${CINECA_SCRATCH}/bonzai-data/fsq/${FSQ_RELEASE}"

if [ -z "${HF_TOKEN:-}" ]; then
    if [ -r "${HOME}/.hf_token" ]; then
        HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
    else
        echo "ERROR: HF_TOKEN env var not set and ${HOME}/.hf_token not found" >&2
        echo "  create it with:  echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token" >&2
        exit 1
    fi
fi

export HF_TOKEN
mkdir -p "${DEST}"

echo "[fsq] repo       = foursquare/fsq-os-places"
echo "[fsq] release    = ${FSQ_RELEASE}"
echo "[fsq] dest       = ${DEST}"
echo "[fsq] start      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Use `hf download` (new CLI replacing the deprecated `huggingface-cli`).
# --local-dir puts files on scratch instead of the quota-limited ~/.cache.
# --include scopes to a single release folder.
.venv/bin/hf download \
    foursquare/fsq-os-places \
    --repo-type dataset \
    --include "release/${FSQ_RELEASE}/*" \
    --local-dir "${DEST}" \
    --token "${HF_TOKEN}"

echo "[fsq] done       = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
du -sh "${DEST}"

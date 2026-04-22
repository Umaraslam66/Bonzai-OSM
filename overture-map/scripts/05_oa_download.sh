#!/usr/bin/env bash
# Download the global OpenAddresses archive (~12 GB zip) to $CINECA_SCRATCH.
#
# NOTE on data source: the batch.openaddresses.io/api endpoints now require
# authentication (as of 2025). However, data.openaddresses.io still publishes
# public collected-region zip files at its static archive URLs, and these are
# kept reasonably fresh (last-modified 2025-09-14 as of this writing).
# If you need bleeding-edge fresh data later, sign up at batch.openaddresses.io
# and provision an OA_TOKEN.
#
# Usage:
#   ./scripts/05_oa_download.sh                    # global (~12 GB)
#   OA_REGION=europe ./scripts/05_oa_download.sh   # per-region variant

set -euo pipefail

: "${CINECA_SCRATCH:?CINECA_SCRATCH must be set (it auto-exports on Leonardo login)}"

OA_REGION="${OA_REGION:-global}"
URL="https://data.openaddresses.io/openaddr-collected-${OA_REGION}.zip"

DEST_DIR="${CINECA_SCRATCH}/bonzai-data/openaddresses"
mkdir -p "${DEST_DIR}"
OUT="${DEST_DIR}/openaddr-collected-${OA_REGION}.zip"

echo "[oa] region      = ${OA_REGION}"
echo "[oa] url         = ${URL}"
echo "[oa] output      = ${OUT}"
echo "[oa] start       = $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Peek at the file to verify reachability and print expected size (best-effort)
SIZE_BYTES="$( { curl -sLI -r 0-0 -D - -o /dev/null --max-time 30 "${URL}" \
                  | grep -i '^content-range:' \
                  | awk '{print $NF}' | cut -d/ -f2 | tr -d '\r\n'; } || true)"
if [ -n "${SIZE_BYTES}" ]; then
    SIZE_GB="$(awk -v b="${SIZE_BYTES}" 'BEGIN{printf "%.2f", b/1024/1024/1024}')"
    echo "[oa] remote size = ${SIZE_BYTES} bytes (${SIZE_GB} GB)"
fi

# --continue-at - resumes if a previous partial file exists.
# --retry handles transient TLS/connection errors.
curl -L --fail --retry 10 --retry-delay 15 --retry-connrefused \
     --continue-at - -o "${OUT}" \
     "${URL}"

echo "[oa] done        = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
du -sh "${OUT}"

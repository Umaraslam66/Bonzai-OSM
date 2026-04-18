#!/bin/bash
# setup_workspace.sh
#
# Create a Stockholm-PoC workspace on the Leonardo $WORK filesystem and
# download the BBBike Stockholm .osm.pbf extract into it.
#
# Run on the Leonardo *login* node. $WORK is auto-exported on login
# (it points at /leonardo_work/AIFAC_P02_222 for this allocation).
#
# Usage (on Leonardo):
#   bash setup_workspace.sh
#
# Why login node is fine for this transfer:
#   The BBBike Stockholm PBF is ~35 MB, well below the 10-minute login-node
#   CPU limit documented in the Leonardo user guide. For planet-scale
#   transfers we use data.leonardo.cineca.it — we keep that same helper
#   path here as a fallback if the direct download gets throttled.

set -euo pipefail

# ---------- config ----------------------------------------------------------

: "${WORK:?WORK is not set - are you on Leonardo with project env loaded?}"

WORKSPACE="${WORK}/stockholm_poc"
DATA_DIR="${WORKSPACE}/data"
OUT_DIR="${WORKSPACE}/outputs"
LOG_DIR="${WORKSPACE}/logs"
VENV_DIR="${WORKSPACE}/venv"

PBF_URL="https://download.bbbike.org/osm/bbbike/Stockholm/Stockholm.osm.pbf"
PBF_PATH="${DATA_DIR}/Stockholm.osm.pbf"

# ---------- layout ----------------------------------------------------------

echo "[setup] creating workspace at ${WORKSPACE}"
mkdir -p "${DATA_DIR}" "${OUT_DIR}" "${LOG_DIR}"

# ---------- download --------------------------------------------------------

if [[ -s "${PBF_PATH}" ]]; then
  echo "[setup] PBF already present at ${PBF_PATH} ($(du -h "${PBF_PATH}" | cut -f1))"
else
  echo "[setup] downloading Stockholm PBF from BBBike"
  # Try the login-node wget first (file is small). If it fails, fall back to
  # the Leonardo datamover which has no CPU-time limit.
  if ! wget --continue --progress=dot:giga -O "${PBF_PATH}" "${PBF_URL}"; then
    echo "[setup] login-node wget failed, retrying via data.leonardo.cineca.it"
    ssh -xt "${USER}@data.leonardo.cineca.it" \
      wget --continue --progress=dot:giga \
      -O "${PBF_PATH}" "${PBF_URL}"
  fi
fi

# ---------- report ----------------------------------------------------------

echo "[setup] workspace ready:"
echo "  WORKSPACE = ${WORKSPACE}"
echo "  DATA      = ${DATA_DIR}"
echo "  OUTPUTS   = ${OUT_DIR}"
echo "  LOGS      = ${LOG_DIR}"
echo "  VENV      = ${VENV_DIR} (not created yet - see README)"
ls -la "${DATA_DIR}"

echo
echo "[setup] next step: create the Python venv and install requirements."
echo "        See stockholm_poc/README.md for the exact commands."

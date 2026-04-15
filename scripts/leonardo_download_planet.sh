#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: leonardo_download_planet.sh [options]

Prepare the Leonardo scratch workspace and download the world OSM PBF through
the Leonardo datamover service. Run this from a Leonardo login node.

Options:
  --raw-dir PATH         Target raw-data directory. Default: $CINECA_SCRATCH/osm/raw
  --datamover HOST       Datamover host. Default: data.leonardo.cineca.it
  --skip-md5             Skip checksum download and verification
  --help                 Show this help

This keeps the long transfer off the login node and avoids burning the project
allocation. The download is network-bound; extraction should still run later on
the budget-free lrd_all_serial partition.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -d /leonardo_work ]] || die "run this on a Leonardo login node"

SCRATCH_ROOT="${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/${USER}}"
RAW_DIR="${SCRATCH_ROOT}/osm/raw"
EXTRACTS_DIR="${SCRATCH_ROOT}/osm/extracts"
LOGS_DIR="${SCRATCH_ROOT}/osm/logs"
DATA_MOVER="data.leonardo.cineca.it"
VERIFY_MD5="yes"
PLANET_URL="https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf"
PLANET_MD5_URL="${PLANET_URL}.md5"
PLANET_OUT="${RAW_DIR}/planet-latest.osm.pbf"
PLANET_MD5_OUT="${RAW_DIR}/planet-latest.osm.pbf.md5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-dir)
      [[ $# -ge 2 ]] || die "--raw-dir requires a value"
      RAW_DIR="$2"
      shift 2
      ;;
    --datamover)
      [[ $# -ge 2 ]] || die "--datamover requires a value"
      DATA_MOVER="$2"
      shift 2
      ;;
    --skip-md5)
      VERIFY_MD5="no"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

mkdir -p "$RAW_DIR" "$EXTRACTS_DIR" "$LOGS_DIR"
if [[ -n "${WORK:-}" && -d "${WORK:-}" ]]; then
  mkdir -p "$WORK/osm/jobs" "$WORK/osm/logs"
fi

printf 'Scratch root: %s\n' "$SCRATCH_ROOT"
printf 'Raw dir:      %s\n' "$RAW_DIR"
printf 'Datamover:    %s\n' "$DATA_MOVER"

ssh -xt "${USER}@${DATA_MOVER}" \
  wget --continue --progress=dot:giga -O "$PLANET_OUT" "$PLANET_URL"

if [[ "$VERIFY_MD5" == "yes" ]]; then
  ssh -xt "${USER}@${DATA_MOVER}" \
    wget --continue -O "$PLANET_MD5_OUT" "$PLANET_MD5_URL"

  (
    cd "$RAW_DIR"
    md5sum -c planet-latest.osm.pbf.md5
  )
fi

printf '\nReady:\n'
printf '  %s/planet-latest.osm.pbf\n' "$RAW_DIR"
if [[ "$VERIFY_MD5" == "yes" ]]; then
  printf '  checksum verified via %s/planet-latest.osm.pbf.md5\n' "$RAW_DIR"
fi

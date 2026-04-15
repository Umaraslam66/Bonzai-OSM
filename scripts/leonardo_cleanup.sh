#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: leonardo_cleanup.sh [options]

Safe Leonardo cleanup helper for the old project area.
It never deletes anything. It inventories usage and can move selected
directories into dated _trash_ folders.

Options:
  --account ACCOUNT       Project/account name. Default: $WORK basename or AIFAC_P02_222
  --inventory-only        Print inventory only (default)
  --stage-defaults        Move known stale dirs:
                          WORK: containers bonzai
                          FAST: bonzai_cache
  --stage-work NAME       Move one WORK dir into trash (repeatable)
  --stage-fast NAME       Move one FAST dir into trash (repeatable)
  --trash-date YYYYMMDD   Override trash suffix date
  --help                  Show this help

Examples:
  ./scripts/leonardo_cleanup.sh --inventory-only
  ./scripts/leonardo_cleanup.sh --stage-defaults
  ./scripts/leonardo_cleanup.sh --stage-work containers --stage-fast bonzai_cache
EOF
}

log() {
  printf '\n== %s ==\n' "$1"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

ACCOUNT_DEFAULT="${WORK:-}"
ACCOUNT_DEFAULT="${ACCOUNT_DEFAULT##*/}"
if [[ -z "${ACCOUNT_DEFAULT}" ]]; then
  ACCOUNT_DEFAULT="AIFAC_P02_222"
fi

ACCOUNT="${OLD_ACCOUNT:-$ACCOUNT_DEFAULT}"
MODE="inventory"
TRASH_DATE="$(date +%Y%m%d)"
declare -a WORK_DIRS=()
declare -a FAST_DIRS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      [[ $# -ge 2 ]] || die "--account requires a value"
      ACCOUNT="$2"
      shift 2
      ;;
    --inventory-only)
      MODE="inventory"
      shift
      ;;
    --stage-defaults)
      MODE="stage"
      WORK_DIRS+=("containers" "bonzai")
      FAST_DIRS+=("bonzai_cache")
      shift
      ;;
    --stage-work)
      [[ $# -ge 2 ]] || die "--stage-work requires a directory name"
      MODE="stage"
      WORK_DIRS+=("$2")
      shift 2
      ;;
    --stage-fast)
      [[ $# -ge 2 ]] || die "--stage-fast requires a directory name"
      MODE="stage"
      FAST_DIRS+=("$2")
      shift 2
      ;;
    --trash-date)
      [[ $# -ge 2 ]] || die "--trash-date requires YYYYMMDD"
      TRASH_DATE="$2"
      shift 2
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

OLD_WORK="${OLD_WORK:-/leonardo_work/${ACCOUNT}}"
OLD_FAST="${OLD_FAST:-/leonardo_scratch/fast/${ACCOUNT}}"
TRASH_WORK="${OLD_WORK}/_trash_${TRASH_DATE}"
TRASH_FAST="${OLD_FAST}/_trash_${TRASH_DATE}"

[[ -d /leonardo_work ]] || die "run this on Leonardo"
[[ -d "$OLD_WORK" ]] || die "missing WORK path: $OLD_WORK"
[[ -d "$OLD_FAST" ]] || printf 'warning: FAST path missing: %s\n' "$OLD_FAST" >&2

print_children_usage() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    printf 'missing: %s\n' "$root"
    return
  fi

  du -sh "$root" 2>/dev/null || true
  find "$root" -mindepth 1 -maxdepth 1 -exec du -sh {} + 2>/dev/null | sort -h || true
}

print_top_files() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return
  fi

  find "$root" -xdev -type f -printf '%s\t%p\n' 2>/dev/null \
    | sort -nr \
    | head -20 \
    | awk '{ printf "%10.2f GB\t%s\n", $1/1024/1024/1024, substr($0, index($0,$2)) }'
}

stage_dirs() {
  local src_root="$1"
  local trash_root="$2"
  shift 2
  local name src dst

  [[ -d "$src_root" ]] || return
  mkdir -p "$trash_root"

  for name in "$@"; do
    src="${src_root}/${name}"
    dst="${trash_root}/${name}"

    if [[ ! -e "$src" ]]; then
      printf 'skip missing: %s\n' "$src"
      continue
    fi

    if [[ -e "$dst" ]]; then
      printf 'skip existing trash target: %s\n' "$dst"
      continue
    fi

    printf 'move: %s -> %s\n' "$src" "$dst"
    mv "$src" "$dst"
  done
}

log "Account"
printf 'ACCOUNT=%s\nOLD_WORK=%s\nOLD_FAST=%s\n' "$ACCOUNT" "$OLD_WORK" "$OLD_FAST"

if command -v saldo >/dev/null 2>&1; then
  log "Budget"
  saldo -b || true
fi

if command -v cindata >/dev/null 2>&1; then
  log "Disk Usage"
  cindata || true
fi

if command -v cinQuota >/dev/null 2>&1; then
  log "Quota"
  cinQuota || true
fi

log "WORK Inventory"
print_children_usage "$OLD_WORK"

log "FAST Inventory"
print_children_usage "$OLD_FAST"

log "Largest WORK Files"
print_top_files "$OLD_WORK"

if [[ "$MODE" == "stage" ]]; then
  log "Staging WORK Trash"
  stage_dirs "$OLD_WORK" "$TRASH_WORK" "${WORK_DIRS[@]}"

  log "Staging FAST Trash"
  stage_dirs "$OLD_FAST" "$TRASH_FAST" "${FAST_DIRS[@]}"

  log "Trash Summary"
  printf 'WORK trash: %s\nFAST trash: %s\n' "$TRASH_WORK" "$TRASH_FAST"
fi

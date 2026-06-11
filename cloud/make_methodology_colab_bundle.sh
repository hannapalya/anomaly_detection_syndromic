#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
OUT="${1:-$ROOT/cloud_methodology_bundle.tgz}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

DEST="$TMPDIR/syndromic"
mkdir -p "$DEST"

rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'results' \
  --exclude 'score_cache' \
  --exclude 'cloud_out' \
  --exclude '*_inputs' \
  "$ROOT/" "$DEST/"

for mag in small medium large; do
  src="$DATA_ROOT/big_signal_datasets_$mag"
  if [[ ! -d "$src" ]]; then
    echo "Missing data directory: $src" >&2
    exit 1
  fi
  rsync -a "$src" "$DEST/"
done

tar -czf "$OUT" -C "$TMPDIR" syndromic
ls -lh "$OUT"

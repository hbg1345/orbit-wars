#!/usr/bin/env bash
# Download the latest N daily Orbit Wars episode datasets from Kaggle, keeping
# each as a compressed .zip (no --unzip) to save disk space.
#
# Usage:
#   ./download_latest.sh             # latest 5, 4 parallel downloads (defaults)
#   ./download_latest.sh 3           # latest 3, 4 parallel
#   ./download_latest.sh 10 6        # latest 10, 6 parallel
#   ./download_latest.sh 5 1         # latest 5, sequential (one at a time)
set -euo pipefail


MANIFEST="data/manifest.csv"
DEST="data/zips"
N="${1:-5}"          # how many of the latest datasets to fetch
JOBS="${2:-5}"       # max concurrent downloads (Kaggle rate-limits, so keep modest)

mkdir -p "$DEST"

echo "Downloading latest $N dataset(s) with up to $JOBS parallel job(s) -> $DEST"

# Latest N by date (col 1), skipping the header; emit just the slug (col 2),
# then fan out to `xargs -P` for bounded-concurrency downloads. No --unzip: each
# dataset is saved as <slug>.zip (already compressed).
tail -n +2 "$MANIFEST" \
  | sort -t, -k1,1 -r \
  | head -n "$N" \
  | cut -d, -f2 \
  | sed '/^$/d' \
  | xargs -P "$JOBS" -I {} kaggle datasets download -d "kaggle/{}" -p "$DEST"

echo
echo "Done. Downloaded zips:"
ls -lh "$DEST"/*.zip

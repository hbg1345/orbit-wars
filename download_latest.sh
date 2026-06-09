#!/usr/bin/env bash
# Download the latest N daily Orbit Wars episode datasets from Kaggle, one at a
# time, keeping each as a compressed .zip (no --unzip) to save disk space.
#
# Usage:
#   ./download_latest.sh           # latest 5 (default)
#   ./download_latest.sh 3         # latest 3
set -euo pipefail

MANIFEST="data/manifest.csv"
DEST="data/zips"
N="${1:-5}"

mkdir -p "$DEST"

# Latest N by date (col 1), skipping the header. Emit: date,slug,total_bytes
latest=$(tail -n +2 "$MANIFEST" | sort -t, -k1,1 -r | head -n "$N")

echo "Downloading latest $N dataset(s) -> $DEST"
while IFS=, read -r date slug url episode_count total_bytes rest; do
    [ -z "$slug" ] && continue
    gb=$(awk "BEGIN{printf \"%.2f\", $total_bytes/1073741824}")
    echo
    echo "=== $date  $slug  (~${gb} GB uncompressed) ==="

    # No --unzip: the dataset is saved as <slug>.zip (already compressed).
    # One at a time (sequential loop) so peak disk use stays at one dataset.
    kaggle datasets download -d "kaggle/$slug" -p "$DEST"
done <<< "$latest"

echo
echo "Done. Downloaded zips:"
ls -lh "$DEST"/*.zip

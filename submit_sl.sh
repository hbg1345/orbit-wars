#!/usr/bin/env bash
# Bundle and submit the SL agent (orbit_net.pt) to the Orbit Wars competition.
#
# The bundle is a multi-file tar.gz with main.py at the archive root, plus every
# module main.py pulls in (main -> physics/features/model, features -> forecast,
# forecast -> physics) and the SL weights.
#
# Usage:
#   ./submit_sl.sh                       # bundle + submit
#   ./submit_sl.sh "SL agent v3"         # custom submission message
#   ./submit_sl.sh "msg" --check         # also run one local sanity game first
#   ./submit_sl.sh --dry-run             # build the tar.gz only, do not submit
set -euo pipefail

# Activate the conda env so `python` and `kaggle` resolve to the orbit env.
# (`conda activate` needs conda.sh sourced first in a non-interactive shell.)

COMP="orbit-wars"
WEIGHTS="orbit_net.pt"
OUT="submission.tar.gz"
FILES=(main.py model.py features.py forecast.py physics.py "$WEIGHTS")

MSG="SL agent"
CHECK=0
DRYRUN=0
for arg in "$@"; do
  case "$arg" in
    --check)   CHECK=1 ;;
    --dry-run) DRYRUN=1 ;;
    *)         MSG="$arg" ;;
  esac
done

# 1) Verify every file to be bundled exists (fail loud — a missing weight or
#    module would silently produce a broken agent).
missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing file: $f" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || { echo "Aborting: bundle incomplete." >&2; exit 1; }

# 2) Build the bundle (main.py at the tar root).
echo "Bundling ${#FILES[@]} files -> $OUT"
tar -czf "$OUT" "${FILES[@]}"
echo "  $(ls -lh "$OUT" | awk '{print $5, $9}')"
echo "  weights: $WEIGHTS ($(date -r "$WEIGHTS" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?'))"

# 3) Optional local sanity game with the exact files about to be submitted (--check).
if [[ "$CHECK" -eq 1 ]]; then
  echo "Local sanity game: run_game.py main.py random ..."
  python run_game.py main.py random >/dev/null
  echo "  OK (one game ran without error)"
fi

# 4) Submit.
if [[ "$DRYRUN" -eq 1 ]]; then
  echo "--dry-run: built $OUT, skipping submit."
  exit 0
fi
echo "Submitting to '$COMP' with message: $MSG"
kaggle competitions submit "$COMP" -f "$OUT" -m "$MSG"

echo
echo "Done. Check status with:"
echo "  kaggle competitions submissions $COMP"

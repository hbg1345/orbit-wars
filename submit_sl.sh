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
WEIGHTS="orbit_net.pt"            # full training checkpoint (weights + optimizer)
WEIGHTS_SUBMIT="orbit_net_submit.pt"  # slim, weights-only (generated below)
OUT="submission.tar.gz"
STAGE="sl_submission"
CODE=(main.py model.py features.py forecast.py physics.py)
# What we need before we can build: the code modules + the full checkpoint.
FILES=("${CODE[@]}" "$WEIGHTS")

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

# 2) Strip the optimizer state out of the checkpoint -> slim weights for the
#    bundle (the full orbit_net.pt is ~146MB; the weights alone are ~50MB). The
#    full checkpoint is left untouched so training can still resume from it.
echo "Slimming weights: $WEIGHTS -> $WEIGHTS_SUBMIT"
python make_submit_weights.py --src "$WEIGHTS" --dst "$WEIGHTS_SUBMIT"

# 3) Stage code + slim weights renamed to orbit_net.pt (the name main.py loads),
#    so the agent code needs no change. Copied fresh so a stale stage can't
#    desync the weights from the code.
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "${CODE[@]}" "$STAGE"/
cp "$WEIGHTS_SUBMIT" "$STAGE/orbit_net.pt"

# 4) Build the bundle (main.py at the tar root).
echo "Bundling -> $OUT"
tar -czf "$OUT" -C "$STAGE" "${CODE[@]}" orbit_net.pt
echo "  $(ls -lh "$OUT" | awk '{print $5, $9}')"
echo "  weights: $WEIGHTS ($(date -r "$WEIGHTS" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?'))"

# 5) Optional local sanity game with the EXACT staged files about to be submitted
#    (--check) -- this also verifies the slim weights load and the agent attacks.
if [[ "$CHECK" -eq 1 ]]; then
  echo "Local sanity game: run_game.py $STAGE/main.py random ..."
  python run_game.py "$STAGE/main.py" random >/dev/null
  echo "  OK (one game ran without error)"
fi

# 6) Submit.
if [[ "$DRYRUN" -eq 1 ]]; then
  echo "--dry-run: built $OUT, skipping submit."
  exit 0
fi
echo "Submitting to '$COMP' with message: $MSG"
kaggle competitions submit "$COMP" -f "$OUT" -m "$MSG"

echo
echo "Done. Check status with:"
echo "  kaggle competitions submissions $COMP"

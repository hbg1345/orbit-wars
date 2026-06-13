#!/usr/bin/env bash
# Bundle and submit the RL agent to Kaggle (orbit-wars).
#
# The submission root file must be named main.py, but the RL entry point is
# main_rl.py with weights orbit_net_rl.pt. So we stage the current training code
# + weights into rl_submission/ (copied fresh every time so a stale model.py
# can't desync the weights), rename main_rl.py -> main.py, and tar from inside
# the folder (-C) so main.py lands at the tar root.
#
# Usage:
#   ./submit_rl.sh "RL agent v3"      # message as the first argument
#   ./submit_rl.sh                    # falls back to a timestamped message
set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-RL agent $(date +%Y-%m-%d_%H:%M)}"

STAGE="rl_submission"
BUNDLE="submission_rl.tar.gz"
# Files main_rl.py imports + the weights it loads. Add new modules here if needed.
FILES=(model.py features.py forecast.py physics.py orbit_net_rl.pt)

# Fail loud if anything we need is missing (don't ship a broken bundle).
for f in main_rl.py "${FILES[@]}"; do
    [ -f "$f" ] || { echo "error: required file '$f' not found" >&2; exit 1; }
done

mkdir -p "$STAGE"
cp "${FILES[@]}" "$STAGE"/
cp main_rl.py "$STAGE"/main.py      # entry point only; root main.py (SL) untouched

tar -czf "$BUNDLE" -C "$STAGE" main.py "${FILES[@]}"

echo "bundled -> $BUNDLE"
echo "submitting with message: $MSG"
kaggle competitions submit orbit-wars -f "$BUNDLE" -m "$MSG"

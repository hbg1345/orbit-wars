#!/usr/bin/env bash
# Bundle and submit the *latest* RL checkpoint to Kaggle (orbit-wars).
#
# orbit_net_rl_last.pt is a full training checkpoint
# ({model_state_dict, optimizer_state_dict, ...}, ~170MB) which blows the
# submission size limit. So we extract just model_state_dict into a bare
# state_dict and ship that as orbit_net_rl.pt (~57MB). The entry point is the
# normal main_rl.py, which already loads orbit_net_rl.pt as a plain state_dict,
# so no agent-code changes are needed.
#
# Requires torch on PATH (run inside the `orbit` conda env).
#
# Usage:
#   ./submit_rl_last.sh "RL agent (last) v3"   # message as the first argument
#   ./submit_rl_last.sh                        # falls back to a timestamped message
set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-RL agent (last) $(date +%Y-%m-%d_%H:%M)}"

STAGE="rl_submission_last"
BUNDLE="submission_rl_last.tar.gz"
SRC_CKPT="orbit_net_rl_last.pt"
# Files main_rl.py imports. The weights (orbit_net_rl.pt) are extracted below.
FILES=(model.py features.py forecast.py physics.py)

# Fail loud if anything we need is missing (don't ship a broken bundle).
for f in main_rl.py "$SRC_CKPT" "${FILES[@]}"; do
    [ -f "$f" ] || { echo "error: required file '$f' not found" >&2; exit 1; }
done

mkdir -p "$STAGE"
cp "${FILES[@]}" "$STAGE"/
cp main_rl.py "$STAGE"/main.py        # entry point only; root main.py (SL) untouched

# Extract only the model weights from the training checkpoint into a bare
# state_dict, so the bundle ships ~57MB instead of ~170MB.
python - "$SRC_CKPT" "$STAGE/orbit_net_rl.pt" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
if "model_state_dict" not in ck:
    sys.exit(f"error: '{src}' has no 'model_state_dict' key (keys: {list(ck)})")
torch.save(ck["model_state_dict"], dst)
print(f"extracted weights -> {dst}")
PY

tar -czf "$BUNDLE" -C "$STAGE" main.py orbit_net_rl.pt "${FILES[@]}"

echo "bundled -> $BUNDLE"
du -h "$BUNDLE" | cut -f1 | sed 's/^/bundle size: /'
echo "submitting with message: $MSG"
kaggle competitions submit orbit-wars -f "$BUNDLE" -m "$MSG"

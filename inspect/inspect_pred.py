"""Inspect what the current orbit_net.pt predicts on a single replay step.

Prints, for the winner's owned planets at one step:
  * the model's chosen target (or NOOP), its softmax prob, and ship count
  * the human's actual action that turn, for comparison

Usage:
  python inspect_pred.py                  # first data/*.json, first launching step
  python inspect_pred.py data/x.json      # that replay
  python inspect_pred.py data/x.json 40   # that replay, step 40
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import glob
import json
import sys

import numpy as np
import torch

import features as F
from model import OrbitNet

WEIGHTS = "orbit_net.pt"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("data/*.json"))[0]
    forced_step = int(sys.argv[2]) if len(sys.argv) > 2 else None

    d = json.load(open(path, encoding="utf-8"))
    steps, rewards = d["steps"], d["rewards"]
    winner = int(np.argmax(rewards))
    print(f"replay {path}  winner=player{winner}  steps={len(steps)}")

    net = OrbitNet()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    net.eval()

    # Pick the step: forced, else the first one where the winner actually
    # launched. The action at step t was decided on the *previous* board
    # (obs[t-1]) -- that pre-action board is the model input (matches play time).
    def launched(t):
        return bool(steps[t][winner].get("action") or [])
    if forced_step is not None:
        t = forced_step
    else:
        t = next((t for t in range(1, len(steps))
                  if steps[t - 1][winner].get("observation", {}).get("planets") and launched(t)),
                 1)
    print(f"inspecting decision at step {t} (input = pre-action board obs[{t-1}])\n")

    obs = steps[t - 1][winner]["observation"]   # pre-action board the agent saw
    action = steps[t][winner].get("action") or []

    feats, valid, mine, ids, xs, ys, ships = F.extract(obs)
    ft = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
    vm = torch.as_tensor(valid, dtype=torch.bool).unsqueeze(0)
    mm = torch.as_tensor(mine, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, h = net(ft, vm, mm)            # [1,N,N+1], [1,N,D]
    probs = torch.softmax(logits[0], dim=-1)   # [N, N+1]
    n_cls = probs.shape[1]
    noop = n_cls - 1

    print("MODEL prediction (owned planets):")
    print(f"  {'planet':>7} {'garrison':>8}  {'-> target':>12} {'prob':>6} {'ships':>7}")
    for i in np.nonzero(mine)[0]:
        p = probs[i]
        tgt = int(torch.argmax(p).item())
        prob = float(p[tgt].item())
        if tgt == noop:
            print(f"  {int(ids[i]):>7} {ships[i]:>8.0f}  {'NOOP':>12} {prob:>6.2f} {'-':>7}")
        else:
            pair = torch.cat([h[0, i], h[0, tgt]], dim=-1)
            ship_pred = float(torch.expm1(net.ship_head(pair)).item())
            tid = int(ids[tgt])
            print(f"  {int(ids[i]):>7} {ships[i]:>8.0f}  {('planet '+str(tid)):>12} "
                  f"{prob:>6.2f} {ship_pred:>7.1f}")

    print("\nHUMAN action this step  [from_planet, angle_rad, num_ships]:")
    if action:
        for mv in action:
            print(f"  {mv}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
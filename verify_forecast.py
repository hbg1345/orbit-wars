"""Validate forecast.simulate's rollout against reality, one turn ahead.

forecast assumes *no new launches*, but a fleet launched at turn t needs >1 turn
to land, so the k=1 projection (production + any fleet arriving next turn +
combat) should match the actual next observation for almost every planet. A low
match rate would mean the production/combat rollout is wrong.
"""
import glob
import json

import numpy as np

import forecast


def actual_state(obs):
    return {int(p[0]): (int(p[1]), float(p[5])) for p in (obs.get("planets") or [])}


owner_ok = owner_bad = ships_ok = ships_bad = 0
bad_examples = []
for path in sorted(glob.glob("data/*.json"))[:25]:
    steps = json.load(open(path, encoding="utf-8"))["steps"]
    for t in range(len(steps) - 1):
        obs = steps[t][0].get("observation", {})
        if not obs.get("planets"):
            continue
        traj = forecast.simulate(obs, 1)            # predict 1 turn ahead
        nxt = actual_state(steps[t + 1][0].get("observation", {}))
        for pid, (owner_seq, ships_seq, _posx, _posy) in traj.items():
            if pid not in nxt:
                continue                            # planet/comet gone from obs
            pred_owner, pred_ships = owner_seq[0], ships_seq[0]
            if pred_owner == forecast.GONE:
                continue
            real_owner, real_ships = nxt[pid]
            if pred_owner == real_owner:
                owner_ok += 1
            else:
                owner_bad += 1
            if abs(pred_ships - real_ships) <= 0.5:
                ships_ok += 1
            else:
                ships_bad += 1
                if len(bad_examples) < 8:
                    bad_examples.append((pid, pred_owner, pred_ships, real_owner, real_ships))

print(f"owner match: {owner_ok}/{owner_ok + owner_bad} "
      f"({100 * owner_ok / max(owner_ok + owner_bad, 1):.2f}%)")
print(f"ships match (±0.5): {ships_ok}/{ships_ok + ships_bad} "
      f"({100 * ships_ok / max(ships_ok + ships_bad, 1):.2f}%)")
print("sample ship mismatches (pid, pred_owner, pred_ships, real_owner, real_ships):")
for e in bad_examples:
    print(f"  {e}")
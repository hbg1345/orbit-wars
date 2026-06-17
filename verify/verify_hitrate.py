"""Run a real game and measure how often our launched fleets actually hit a
planet (vs dying on the sun / going out of bounds). Validates lead-shot aiming
against the actual engine, not a simulation."""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import logging
logging.disable(logging.WARNING)
import math

from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])
steps = env.steps
P = 0  # our player


def fleet_speed(ships):
    return 1.0 + 5.0 * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5

def fleets_at(t):
    return {f[0]: f for f in steps[t][0]["observation"]["fleets"]}

def planets_at(t):
    return steps[t][0]["observation"]["planets"]

idx = [fleets_at(t) for t in range(len(steps))]
hit = sun = oob = flying = 0
for t in range(len(steps)):
    prev = idx[t - 1].keys() if t > 0 else set()
    for fid, f in idx[t].items():
        if fid in prev or f[1] != P:
            continue  # only fleets our player launched this step
        # trace to disappearance
        last = f
        gone = None
        for k in range(t + 1, len(steps)):
            g = idx[k].get(fid)
            if g is None:
                gone = k
                break
            last = g
        if gone is None:
            flying += 1
            continue
        lx, ly = last[2], last[3]
        step = fleet_speed(last[6])  # one move of this fleet
        # A fleet vanishes right after its last record; it hit a planet if a
        # planet is within one step + its radius of that last position.
        near_planet = min((math.hypot(p[2] - lx, p[3] - ly) - p[4]
                           for p in planets_at(gone)), default=99)
        if near_planet < step + 0.5:
            hit += 1
        elif math.hypot(lx - 50, ly - 50) < 10 + step:
            sun += 1
        else:
            oob += 1

total = hit + sun + oob + flying
print(f"our launches: {total}")
print(f"  hit planet : {hit}")
print(f"  sun death  : {sun}")
print(f"  out/expired: {oob}")
print(f"  still flying at end: {flying}")
if hit + sun + oob:
    print(f"hit rate (of resolved): {hit/(hit+sun+oob)*100:.1f}%")

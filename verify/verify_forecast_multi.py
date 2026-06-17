"""Multi-turn validation of forecast.simulate against the real engine.

Strategy: both agents launch fleets up to turn T, then go *idle* (return [])
forever after. From turn T on, the engine evolves with NO new launches -- which
is exactly forecast's assumption -- so forecast.simulate(obs_T, H) can be checked
against the engine's actual states at T+1 .. T+H, isolating the rollout physics
(fleet ETA tracking, swept collision, production, rotation, comet motion, combat).

We pick T so no comet spawn (steps 50,150,250,350,450) falls inside (T, T+H],
since forecast cannot know about comets that have not appeared yet.

Usage: python verify_forecast_multi.py [--games N] [--T T] [--H H]
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import math
import logging

logging.disable(logging.WARNING)

from kaggle_environments import make

import forecast

COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def make_agent(stop_turn):
    """Aggressive nearest-target launcher that goes idle at/after stop_turn, so
    in-flight fleets exist at the cutoff but no new ones are created afterward."""
    def agent(obs):
        step = obs.get("step", 0)
        if step >= stop_turn:
            return []
        player = obs.get("player", 0)
        planets = obs.get("planets", []) or []
        mine = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]
        moves = []
        for mp in mine:
            if mp[5] < 4 or not targets:
                continue
            t = min(targets, key=lambda q: math.hypot(mp[2] - q[2], mp[3] - q[3]))
            angle = math.atan2(t[3] - mp[3], t[2] - mp[2])
            ships = int(mp[5] // 2)
            if ships >= 2:
                moves.append([mp[0], angle, ships])
        return moves
    return agent


def actual(obs):
    return {int(p[0]): (int(p[1]), float(p[5]), float(p[2]), float(p[3]))
            for p in (obs.get("planets") or [])}


def run_one(seed, T, H):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    ag = make_agent(T)
    env.run([ag, ag])
    steps = env.steps
    if T + H >= len(steps):
        return None

    obs_T = steps[T][0]["observation"]
    if not obs_T.get("planets"):
        return None

    traj = forecast.simulate(obs_T, H)

    res = {"owner_ok": 0, "owner_bad": 0, "ships_ok": 0, "ships_bad": 0,
           "pos_ok": 0, "pos_bad": 0, "bad": []}
    for k in range(1, H + 1):
        nxt = actual(steps[T + k][0]["observation"])
        for pid, (oseq, sseq, xseq, yseq) in traj.items():
            po = oseq[k - 1]
            if po == forecast.GONE:
                continue  # forecast says body is gone -> not comparable here
            if pid not in nxt:
                continue  # engine dropped it (e.g. expired comet); skip
            ro, rs, rx, ry = nxt[pid]
            if po == ro:
                res["owner_ok"] += 1
            else:
                res["owner_bad"] += 1
            if abs(sseq[k - 1] - rs) <= 0.5:
                res["ships_ok"] += 1
            else:
                res["ships_bad"] += 1
                if len(res["bad"]) < 10:
                    res["bad"].append((k, pid, po, round(sseq[k - 1], 1), ro, rs))
            if math.hypot(xseq[k - 1] - rx, yseq[k - 1] - ry) <= 0.5:
                res["pos_ok"] += 1
            else:
                res["pos_bad"] += 1
    return res


def main():
    games, T, H = 8, 40, 8
    argv = iter(sys.argv[1:])
    for a in argv:
        if a == "--games":
            games = int(next(argv))
        elif a == "--T":
            T = int(next(argv))
        elif a == "--H":
            H = int(next(argv))

    # warn if a comet spawn lands inside the horizon
    for s in COMET_SPAWN_STEPS:
        if T < s <= T + H:
            print(f"WARNING: comet spawn at step {s} is inside (T={T}, T+H={T+H}] "
                  f"-- forecast can't know future comets; expect some mismatches.")

    tot = {"owner_ok": 0, "owner_bad": 0, "ships_ok": 0, "ships_bad": 0,
           "pos_ok": 0, "pos_bad": 0}
    all_bad = []
    n = 0
    for seed in range(100, 100 + games):
        r = run_one(seed, T, H)
        if r is None:
            continue
        n += 1
        for kk in tot:
            tot[kk] += r[kk]
        all_bad.extend(r["bad"])

    def pct(ok, bad):
        d = ok + bad
        return f"{ok}/{d} ({100 * ok / max(d, 1):.2f}%)"

    print(f"\n{n} games, T={T}, H={H}  (comparing forecast vs engine, no launches after T)")
    print(f"  owner match:        {pct(tot['owner_ok'], tot['owner_bad'])}")
    print(f"  ships match (±0.5): {pct(tot['ships_ok'], tot['ships_bad'])}")
    print(f"  pos   match (±0.5): {pct(tot['pos_ok'], tot['pos_bad'])}")
    if all_bad:
        print("\nsample ship mismatches (k, pid, pred_owner, pred_ships, real_owner, real_ships):")
        for e in all_bad[:15]:
            print(f"  {e}")


if __name__ == "__main__":
    main()

"""Validate physics.aim against the real engine.

For many (source, target) pairs we compute aim's launch angle + intended arrival
turn, launch that ONE fleet in the real engine (everyone else idle), step tick by
tick, and check where/when the fleet actually lands:

  - hit the intended target?  (correctness of the angle)
  - on the turn aim predicted?  (correctness of the minimum-turn timing)

Tracking: with a single fleet in flight, the tick it disappears is its landing
tick; the planet whose garrison deviates from pure production that tick is the
body it struck (sun/out-of-bounds -> no deviation anywhere = a miss).
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import math
import logging

logging.disable(logging.WARNING)

from kaggle_environments import make

import physics as P


def aim_with_k(src, world_trajectories, radii, target_idx, v, kmax=P.K_MAX_GLOBAL):
    """physics.aim, but also returns the intended integer arrival turn k."""
    sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
    target_r = radii[target_idx]
    target_traj = world_trajectories[target_idx]

    def pos_at(t):
        k0 = int(math.floor(t))
        a = target_traj[k0]
        if a is None:
            return None
        b = target_traj[k0 + 1] if k0 + 1 < len(target_traj) else None
        if b is None:
            return a
        f = t - k0
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

    for k in range(1, kmax + 1):
        pos = target_traj[k]
        if pos is None:
            break
        px, py = pos
        reach = math.hypot(px - sx, py - sy) - src_r
        if abs(v * k - reach) <= target_r:
            lo, hi = float(k - 1), float(k)

            def f(t):
                p = pos_at(t)
                if p is None:
                    return None
                return v * t - (math.hypot(p[0] - sx, p[1] - sy) - src_r)
            flo, fhi = f(lo), f(hi)
            if flo is not None and fhi is not None and flo < 0.0 <= fhi:
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if fm is None:
                        break
                    if fm >= 0.0:
                        hi = mid
                    else:
                        lo = mid
                refined = pos_at(0.5 * (lo + hi))
                if refined is not None:
                    px, py = refined
            angle = math.atan2(py - sy, px - sx)
            ox = sx + (src_r + P.SPAWN_GAP) * math.cos(angle)
            oy = sy + (src_r + P.SPAWN_GAP) * math.sin(angle)
            hit_idx, hit_k = P.trace_fleet(ox, oy, angle, v, world_trajectories, radii, kmax)
            if hit_idx == target_idx:
                return angle, hit_k   # k_pred = engine-exact arrival turn
    return None, None


def snap(obs):
    return {int(p[0]): (int(p[1]), float(p[5]), float(p[6])) for p in (obs.get("planets") or [])}


def fleet_ids(obs):
    return {int(f[0]) for f in (obs.get("fleets") or [])}


def landed_planet(prev, cur):
    """Which planet's garrison deviated from pure production between two snaps
    (the body the fleet struck), or None if nothing deviated (sun / OOB miss)."""
    best_pid, best_dev = None, 0.5
    for pid, (own, ships, prod) in cur.items():
        if pid not in prev:
            continue
        powner, pships, pprod = prev[pid]
        expected = pships + (pprod if powner != -1 else 0.0)
        dev = abs(ships - expected) + (0 if own == powner else 1000)
        if dev > best_dev:
            best_dev, best_pid = dev, pid
    return best_pid


# The reset (step 0) observation reports planet positions one rotation tick out
# of phase with the engine's subsequent stepping; from step 1 on, predict(k)
# matches the engine exactly. Real agents aim from live mid-game observations,
# so warm up a few idle ticks before aiming to measure aim's true accuracy.
WARMUP = 5


def _make(seed, episode_steps):
    e = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_steps},
             debug=False)
    e.reset(num_agents=2)
    for _ in range(WARMUP):
        e.step([[], []])
    return e


def trial(seed, kmax=P.K_MAX_GLOBAL, episode_steps=90):
    env = _make(seed, episode_steps)
    obs0 = env.state[0]["observation"]
    planets = obs0.get("planets") or []
    if not planets:
        return []

    mine = [p for p in planets if int(p[1]) == 0]
    if not mine:
        return []
    src = mine[0]

    world = P.build_world(obs0)
    world_traj = [[w["predict"](k) for k in range(kmax + 1)] for w in world]
    radii = [w["radius"] for w in world]
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}

    # Classify targets as orbiting (moving) vs static to report separately.
    def is_orbiting(idx):
        a, b = world_traj[idx][0], world_traj[idx][1]
        return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) > 1e-6

    results = []
    n = int(src[5])  # launch the whole garrison as one fleet
    if n < 1:
        return []
    v = P.fleet_speed(n)

    for tp in planets:
        tid = int(tp[0])
        if tid == int(src[0]):
            continue
        tidx = id_to_idx[tid]
        angle, k_pred = aim_with_k(src, world_traj, radii, tidx, v, kmax=kmax)
        if angle is None:
            continue

        # Fresh env (same warmup) per launch so trials don't interfere.
        e = _make(seed, episode_steps)
        fid = e.state[0]["observation"].get("next_fleet_id", 0)
        prev = snap(e.state[0]["observation"])
        e.step([[[int(src[0]), angle, n]], []])

        landed_pid, landed_turn, prev_alive = None, None, True
        for t in range(1, episode_steps - WARMUP):
            cur_obs = e.state[0]["observation"]
            cur = snap(cur_obs)
            alive = fid in fleet_ids(cur_obs)
            if prev_alive and not alive:
                landed_pid = landed_planet(prev, cur)
                landed_turn = t
                break
            prev = cur
            prev_alive = alive
            if e.done:
                break
            e.step([[], []])

        results.append({
            "tid": tid, "orbiting": is_orbiting(tidx), "k_pred": k_pred,
            "landed_pid": landed_pid, "landed_turn": landed_turn,
        })
    return results


def main():
    tot = {"n": 0, "hit": 0, "turn_ok": 0, "miss": 0, "wrong": 0}
    orb = {"n": 0, "hit": 0, "turn_ok": 0}
    bad = []
    for seed in range(200, 230):
        for r in trial(seed):
            tot["n"] += 1
            o = r["orbiting"]
            if o:
                orb["n"] += 1
            if r["landed_pid"] == r["tid"]:
                tot["hit"] += 1
                if o:
                    orb["hit"] += 1
                if r["landed_turn"] == r["k_pred"]:
                    tot["turn_ok"] += 1
                    if o:
                        orb["turn_ok"] += 1
                elif len(bad) < 12:
                    bad.append(("TURN", r))
            elif r["landed_pid"] is None:
                tot["miss"] += 1
                if len(bad) < 12:
                    bad.append(("MISS", r))
            else:
                tot["wrong"] += 1
                if len(bad) < 12:
                    bad.append(("WRONG", r))

    def pct(a, b):
        return f"{a}/{b} ({100*a/max(b,1):.2f}%)"

    print(f"\naim vs engine: {tot['n']} launches")
    print(f"  hit intended target:     {pct(tot['hit'], tot['n'])}")
    print(f"  landed on predicted turn: {pct(tot['turn_ok'], tot['hit'])}  (of hits)")
    print(f"  orbiting targets hit:    {pct(orb['hit'], orb['n'])}")
    print(f"  orbiting turn exact:     {pct(orb['turn_ok'], orb['hit'])}  (of orbiting hits)")
    print(f"  missed entirely (sun/OOB): {tot['miss']}")
    print(f"  hit WRONG body:          {tot['wrong']}")
    if bad:
        print("\nsamples (kind, target, orbiting, k_pred, landed_pid, landed_turn):")
        for kind, r in bad:
            print(f"  {kind}: tid={r['tid']} orb={r['orbiting']} k_pred={r['k_pred']} "
                  f"landed_pid={r['landed_pid']} landed_turn={r['landed_turn']}")


if __name__ == "__main__":
    main()

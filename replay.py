"""Parse Orbit Wars replay JSONs into supervised-learning examples.

Each Kaggle replay has:
  steps[t][p] = {action, observation, reward, status, info}
  observation = {planets, fleets, player, step, angular_velocity, ...}
  action      = [[from_planet_id, angle, num_ships], ...]
  fleets      = [[id, owner, x, y, angle, from_planet_id, ships], ...]

For supervised learning we imitate the *winner* of each game. For every step we
turn the winner's owned planets into per-source labels matching OrbitNet's head:

  target_label[i] = slot index of the planet source i launched at,
                    or NOOP (== MAX_PLANETS) if it launched nothing.
                    -1 means "not an owned source" -> ignored in the loss.
  ships_label[i]  = absolute number of ships sent (only where a launch happened)
  has_ship[i]     = True where a real launch happened (ship loss applies)

Target resolution *forward-simulates* the launch with the same physics the
agent uses to aim (main.py): every planet/comet position is known at every turn,
so we march the fleet along its straight heading and find the first body whose
per-turn path segment comes within its radius of the fleet's move segment. This
catches small (radius-1) planets and comets that single-step trajectory tracing
missed, and resolves fleets still in flight when the game ends.
"""

import json
import math
import numpy as np

import features as F
import physics as P

NOOP = F.MAX_PLANETS  # extra class index for "send nothing"


def _index_fleets(steps):
    """Return per-step dict {fleet_id: fleet} and the set of ids present."""
    per_step = []
    for t in range(len(steps)):
        fleets = steps[t][0]["observation"].get("fleets", []) or []
        per_step.append({f[0]: f for f in fleets})
    return per_step


def _body_track(world):
    """Cache every body's position per turn so multiple launches in the same step
    don't re-predict the same planet/comet trajectories.

    Returns at(k) -> list of (id, radius, pos) aligned to `world`, where pos is
    (x, y) or None. Body motion depends only on the step, not on which fleet we
    are tracing, so the positions are computed once and shared across launches.
    """
    cache = {}

    def at(k):
        out = cache.get(k)
        if out is None:
            out = [(w["id"], w["radius"], w["predict"](k)) for w in world]
            cache[k] = out
        return out
    return at


def _resolve_target(fleet_idx, t, fid, ang, nsh, body_at, kmax=400):
    """Forward-simulate the fleet launched at step t; return the planet id it
    hits, or None if it leaves the board or crosses the sun first.

    The launched fleet already appears in the post-action observation at step t,
    so we find the fleet that is new at step t (not present at t-1) with the
    matching source/angle/ships and take that as the start of the trajectory.
    From there the fleet travels in a straight line at constant speed, while every
    planet/comet position is known at every future turn (supplied by `body_at`,
    a per-step position cache). Each turn we test the fleet's move segment against
    each body's per-turn path segment (both move simultaneously) within its radius
    -- the engine's continuous collision check. The first body hit is the target;
    reaching the board edge or the sun first means there is no planet target.
    """
    prev_ids = fleet_idx[t - 1].keys() if t > 0 else set()
    start = None
    for fleet_id, f in fleet_idx[t].items():
        if fleet_id in prev_ids:
            continue
        # f = [id, owner, x, y, angle, from_planet_id, ships]
        if f[5] == fid and f[6] == nsh and abs(f[4] - ang) < 1e-6:
            start = f
            break
    if start is None:
        return None

    sx, sy = float(start[2]), float(start[3])
    v = P.fleet_speed(start[6])
    cosA, sinA = math.cos(float(start[4])), math.sin(float(start[4]))

    px, py = sx, sy                                   # fleet position at turn k-1
    for k in range(1, kmax + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA  # fleet position at turn k

        # Planet/comet collision: fleet move segment vs each body's path segment.
        best_id, best_d = None, math.inf
        for (bid, radius, p), (_, _, q) in zip(body_at(k - 1), body_at(k)):
            # A fleet spawns on its source planet's surface, so it would always
            # "graze" that planet on the first turns. It can never land on the
            # planet it launched from (the net masks the self column), so the
            # source is excluded from collision regardless of launch angle.
            if bid == fid or p is None or q is None:
                continue
            # Continuous collision under simultaneous fleet+body motion
            # (P.swept_dist; seg_seg_dist would report phantom path crossings).
            d = P.swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1])
            if d <= radius and d < best_d:
                best_id, best_d = bid, d
        if best_id is not None:
            return best_id  # hit a planet (or comet) -> that's the target

        # Otherwise the only ways to vanish are the sun or the board edge.
        if P.seg_point_dist(px, py, fx, fy, P.SUN_CENTER[0], P.SUN_CENTER[1]) <= P.SUN_RADIUS:
            return None
        if not (0.0 <= fx <= P.BOARD and 0.0 <= fy <= P.BOARD):
            return None
        px, py = fx, fy
    return None  # never resolved within kmax turns (no clean target)


def parse_episode(path, perspective="winner"):
    """Return a list of SL example dicts (numpy arrays) from one replay file.

    perspective: "winner" (imitate the game winner) or "both".

    Observation/action alignment: in the replay, the action logged at step t was
    decided on the *previous* board, step t-1 (verified: the source planet only
    has enough ships to launch in obs[t-1]; obs[t] already shows it depleted with
    the new fleet in flight). So we pair the pre-action board obs[t-1] -- exactly
    what the agent sees before acting at play time -- with the action at step t,
    while still resolving the launched fleet's target from step t (where it first
    appears).
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return parse_episode_data(d, perspective)


def parse_episode_data(d, perspective="winner"):
    """Same as parse_episode but from an already-loaded replay dict, so callers
    can stream episodes straight out of a .zip without extracting them to disk."""
    rewards = d["rewards"]
    steps = d["steps"]
    fleet_idx = _index_fleets(steps)
    players = [int(np.argmax(rewards))] if perspective == "winner" else [0, 1]

    # A fleet's target/arrival is fixed geometry, so resolve each fleet's ETA
    # once per episode and reuse it across every step (forecast.simulate). Keep a
    # separate cache per player so each only reuses ETAs traced from its own
    # observation (fleet visibility can differ between the two perspectives).
    eta_caches = {p: {} for p in players}

    # Per-game canonicalisation (rotate my start quadrant -> Q1, CCW enemy keys),
    # computed ONCE from the initial board and reused for every step.
    canons = {}
    for p in players:
        obs0 = steps[0][p].get("observation", {})
        canons[p] = F.compute_canon(obs0) if obs0.get("planets") else None

    examples = []
    for t in range(1, len(steps)):
        # Body motion is global and depends only on the step, so build the world
        # and its per-turn position cache once and share them across this step's
        # launches (and both players when perspective="both").
        body_at = None
        for p in players:
            obs = steps[t - 1][p].get("observation", {})   # pre-action board
            if not obs.get("planets"):
                continue
            action = steps[t][p].get("action") or []        # decided on obs[t-1]

            feats, valid, mine, ids, xs, ys, ships, _traj = F.extract(obs, eta_cache=eta_caches[p], canon=canons[p])
            N = F.MAX_PLANETS
            id2slot = {int(pid): i for i, pid in enumerate(ids) if pid >= 0}

            # launch  : gate label over owned planets (1 = the human launched here)
            # target  : the slot the launch resolved to (-1 where unresolved/no-op)
            # ship    : ships sent as a *fraction* of the source's garrison [0,1]
            # has_ship: launch resolved to a known target (supervises target+ship)
            launch_label = np.zeros(N, dtype=bool)
            target_label = np.full(N, -1, dtype=np.int64)
            ship_label = np.zeros(N, dtype=np.float32)
            has_ship = np.zeros(N, dtype=bool)

            # Collapse multiple launches from one source to its largest launch.
            by_src = {}
            for mv in action:
                fid, ang, nsh = int(mv[0]), float(mv[1]), int(mv[2])
                si = id2slot.get(fid)
                if si is None or not mine[si]:
                    continue
                if si not in by_src or nsh > by_src[si][1]:
                    by_src[si] = (ang, nsh)

            for si, (ang, nsh) in by_src.items():
                launch_label[si] = True          # the human launched from here
                fid = int(ids[si])
                if body_at is None:
                    body_at = _body_track(P.build_world(steps[t][0]["observation"]))
                tgt_id = _resolve_target(fleet_idx, t, fid, ang, nsh, body_at)
                if tgt_id is None or tgt_id not in id2slot:
                    # Launched, but the fleet hits the sun / leaves the board /
                    # never resolves: the gate still learns "launch here", we just
                    # can't supervise which target or how many ships.
                    continue
                target_label[si] = id2slot[tgt_id]
                garrison = max(float(ships[si]), 1.0)         # source ships at launch
                ship_label[si] = min(float(nsh) / garrison, 1.0)
                has_ship[si] = True

            examples.append({
                "feats": feats,
                "valid": valid,
                "mine": mine,
                "launch": launch_label,
                "target": target_label,
                "ships": ship_label,
                "has_ship": has_ship,
            })
    return examples


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/78559827.json"
    ex = parse_episode(path)
    n_owned = sum(int(e["mine"].sum()) for e in ex)
    n_launch = sum(int(e["launch"].sum()) for e in ex)
    n_tgt = sum(int(e["has_ship"].sum()) for e in ex)
    print(f"{path}: {len(ex)} step-examples, {n_owned} owned sources, "
          f"{n_launch} launches, {n_tgt} resolved targets")

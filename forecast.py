"""Forward-simulate the board to build temporal (per-future-turn) features.

Every in-flight fleet has a deterministic target and arrival turn (same physics
as the aimer in main.py and the parser in replay.py). Assuming *no new launches*,
we roll the board forward HORIZON turns -- applying the orbit_star.md turn order
(production -> fleet movement/arrival -> combat) -- to get, for each planet and
comet, its garrison and owner at every future turn.

These trajectories become the network's "시간 별" temporal features
(features.py). It is a baseline projection: it shows where the fleets currently
in flight land, which planets flip, and how garrisons grow if neither side
issues any further orders.
"""

import math
from collections import defaultdict

import physics as P

GONE = -2  # sentinel: body has left the board (comet) and is no longer present

# When caching ETAs across an episode we must resolve a fleet's *absolute*
# arrival turn even if it lands beyond the current HORIZON, so later steps (when
# that arrival has drifted inside the horizon) get the right answer. The board is
# 100x100 and the slowest fleet moves ~1/turn, so any fleet that lands does so
# within ~141 turns of launch; 400 is a safe ceiling matching replay._resolve.
ETA_SEARCH = 400


def _track(world):
    """at(k) -> list of (id, radius, pos) for every body k turns from now,
    computed once per k and cached (shared across all fleet ETAs + the sim)."""
    cache = {}

    def at(k):
        out = cache.get(k)
        if out is None:
            out = [(w["id"], w["radius"], w["predict"](k)) for w in world]
            cache[k] = out
        return out
    return at


def _fleet_eta(fleet, at, kmax):
    """(target_id, arrival_turn) for one in-flight fleet, or (None, None) if it
    leaves the board / crosses the sun before hitting a planet within kmax.

    fleet = [id, owner, x, y, angle, from_planet_id, ships]. Collision is the
    same continuous segment test the engine uses. The source planet is NOT
    excluded: the engine never excludes it either (it just spawns the fleet
    radius+0.1 outside), so an orbiting source that later rotates back into the
    fleet's path recaptures it -- excluding it permanently made forecast miss
    those return arrivals (verify_forecast_multi.py)."""
    sx, sy = float(fleet[2]), float(fleet[3])
    v = P.fleet_speed(fleet[6])
    cosA, sinA = math.cos(float(fleet[4])), math.sin(float(fleet[4]))

    px, py = sx, sy
    for k in range(1, kmax + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA
        # Continuous collision under simultaneous fleet+body motion. The engine
        # iterates obs.planets in order and takes the FIRST body it intersects
        # (orbit_wars.py: break on first swept hit); at(k) preserves that order,
        # so we return the first hit too rather than the nearest one.
        for (bid, radius, p), (_, _, q) in zip(at(k - 1), at(k)):
            if p is None or q is None:
                continue
            d = P.swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1])
            if d <= radius:
                return bid, k
        if P.seg_point_dist(px, py, fx, fy, P.SUN_CENTER[0], P.SUN_CENTER[1]) <= P.SUN_RADIUS:
            return None, None
        if not (0.0 <= fx <= P.BOARD and 0.0 <= fy <= P.BOARD):
            return None, None
        px, py = fx, fy
    return None, None


def _resolve_combat(owner, ships, arrivals):
    """Apply one turn's arriving fleets to a planet (orbit_star.md combat rules).

    arrivals: list of (fleet_owner, fleet_ships). Returns the new (owner, ships).
    Arriving fleets are summed per owner; the largest force fights the second
    largest and the difference survives; that survivor then fights the garrison.
    """
    by = defaultdict(float)
    for fo, fs in arrivals:
        by[fo] += fs
    groups = sorted(by.items(), key=lambda kv: -kv[1])
    if not groups:
        return owner, ships
    if len(groups) >= 2 and groups[0][1] == groups[1][1]:
        return owner, ships  # top attackers tie -> all destroyed, garrison intact

    s_owner = groups[0][0]
    s_ships = groups[0][1] - (groups[1][1] if len(groups) >= 2 else 0.0)
    if s_ships <= 0:
        return owner, ships
    if s_owner == owner:
        return owner, ships + s_ships            # reinforcement from same owner
    if s_ships > ships:
        return s_owner, s_ships - ships          # captured; surplus becomes garrison
    return owner, ships - s_ships                # defender holds, garrison reduced


def simulate(obs, horizon, eta_cache=None):
    """Return {planet_id: (owner_seq, ships_seq, posx_seq, posy_seq)} for the next
    `horizon` turns.

    owner_seq[t] / ships_seq[t] are the owner (or GONE) and garrison at turn
    t+1 from now, recorded after that turn's production and combat. posx_seq[t] /
    posy_seq[t] are the body's (x, y) at that turn (0.0 once it is GONE) -- the
    positions are already computed for collision/arrival, so this just keeps them.
    This is what lets the temporal features carry motion (comets fly across the
    board, planets orbit) without a separate velocity channel.

    eta_cache: optional dict shared across the steps of one episode. A fleet's
    target and *absolute* arrival turn are fixed geometry (independent of the
    board's ownership/combat state), so we resolve each fleet once -- keyed by
    its id -- and on later steps just shift the cached absolute arrival into the
    current step's frame instead of re-tracing the trajectory. Pass None at play
    time (single observation) to disable caching; the result is identical.
    """
    world = P.build_world(obs)
    at = _track(world)

    raw = P._obs_get(obs, "planets", []) or []
    comet_ids = set(P._obs_get(obs, "comet_planet_ids", []) or [])

    owner = {int(p[0]): int(p[1]) for p in raw}
    ships = {int(p[0]): float(p[5]) for p in raw}
    prod = {int(p[0]): float(p[6]) for p in raw}
    ids = list(owner.keys())
    pos_index = {bid: i for i, (bid, _, _) in enumerate(at(0))}
    gone = {pid: False for pid in ids}

    step_now = int(P._obs_get(obs, "step", 0))

    # Bucket fleet arrivals by the turn they land. With a cache we resolve each
    # fleet's absolute arrival once (searching out to ETA_SEARCH so an arrival
    # beyond the current horizon is still recorded for future steps); without
    # one we only need to look as far as the horizon.
    arrivals = defaultdict(list)  # k -> [(target_id, fleet_owner, fleet_ships)]
    for f in (P._obs_get(obs, "fleets", []) or []):
        fid = int(f[0])
        if eta_cache is not None and fid in eta_cache:
            tid, abs_arr = eta_cache[fid]
        else:
            depth = ETA_SEARCH if eta_cache is not None else horizon
            tid, k = _fleet_eta(f, at, depth)
            abs_arr = (step_now + k) if tid is not None else None
            if eta_cache is not None:
                eta_cache[fid] = (tid, abs_arr)
        if tid is None or tid not in owner:
            continue
        k_rel = abs_arr - step_now
        if 1 <= k_rel <= horizon:
            arrivals[k_rel].append((tid, int(f[1]), float(f[6])))

    owner_seq = {pid: [GONE] * horizon for pid in ids}
    ships_seq = {pid: [0.0] * horizon for pid in ids}
    posx_seq = {pid: [0.0] * horizon for pid in ids}
    posy_seq = {pid: [0.0] * horizon for pid in ids}

    for k in range(1, horizon + 1):
        bodies_k = at(k)
        # Comet expiration: a comet whose predicted position is None has left.
        for pid in ids:
            if pid in comet_ids and not gone[pid] and bodies_k[pos_index[pid]][2] is None:
                gone[pid] = True
        # Production: every owned, still-present body generates its ships.
        for pid in ids:
            if not gone[pid] and owner[pid] != -1:
                ships[pid] += prod[pid]
        # Arrivals -> combat, grouped by target planet.
        by_target = defaultdict(list)
        for tid, fo, fs in arrivals.get(k, []):
            if not gone[tid]:
                by_target[tid].append((fo, fs))
        for tid, group in by_target.items():
            owner[tid], ships[tid] = _resolve_combat(owner[tid], ships[tid], group)
        # Record post-turn state (owner, garrison, and position).
        for pid in ids:
            if not gone[pid]:
                owner_seq[pid][k - 1] = owner[pid]
                ships_seq[pid][k - 1] = ships[pid]
                pos = bodies_k[pos_index[pid]][2]
                if pos is not None:
                    posx_seq[pid][k - 1] = pos[0]
                    posy_seq[pid][k - 1] = pos[1]

    return {pid: (owner_seq[pid], ships_seq[pid], posx_seq[pid], posy_seq[pid])
            for pid in ids}
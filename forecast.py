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


# ---------------------------------------------------------------------------
# Minimum-capture fleet heuristic
# ---------------------------------------------------------------------------
# How many ships to send is NOT a learned policy output -- it is derived from
# physics + this forecast. Given a (source, target), find the SMALLEST fleet that
# both (a) reaches the target and (b) outnumbers its garrison at arrival -- the
# cheapest capture, struck at whichever turn the forecast makes easiest (e.g.
# right after someone else has weakened it). Fleet speed is size-dependent
# (P.fleet_speed), so fleet size and arrival turn are coupled; we seed candidate
# sizes from the forecast and let aim/trace_fleet confirm the real, collision-
# checked path. Combat (orbit_star.md): capture needs attackers > garrison
# (a tie destroys both), hence garrison + 1.

_LN1000 = math.log(1000.0)
_MAX_SCAN = 10                      # extra speed samples tried when the min fleet is blocked


def ships_for_speed(v):
    """Fewest ships whose fleet_speed reaches v (inverse of P.fleet_speed).
    None if v exceeds the speed cap (no fleet can travel that fast)."""
    if v <= 1.0:
        return 1
    if v > P.MAX_SPEED:
        return None
    return math.ceil(math.exp(_LN1000 * ((v - 1.0) / (P.MAX_SPEED - 1.0)) ** (2.0 / 3.0)))


def _try_fleet(n, src, target_idx, ships_seq, world_traj, radii, kmax):
    """Launch n ships at target_idx and verify with engine-faithful geometry:
    (n, angle) if the fleet actually reaches the target AND outnumbers its
    garrison at the real arrival turn, else None. Different n -> different speed
    -> different (moving-)target lead angle and crossing TIMING, so each size is
    re-aimed and re-traced rather than assumed from arrival turn alone."""
    v = P.fleet_speed(n)
    angle = P.aim(src, world_traj, radii, target_idx, v, kmax)
    if angle is None:                                  # uncatchable / every shot blocked
        return None
    sx, sy, src_r = src[2], src[3], src[4]
    ox = sx + (src_r + P.SPAWN_GAP) * math.cos(angle)
    oy = sy + (src_r + P.SPAWN_GAP) * math.sin(angle)
    hit, arr = P.trace_fleet(ox, oy, angle, v, world_traj, radii, kmax)
    if hit != target_idx or arr is None or arr > len(ships_seq):
        return None                                    # arrival past forecast horizon -> unknown defense
    if n <= int(ships_seq[arr - 1]):                   # must outnumber garrison at arrival
        return None
    return n, angle


def min_capture_fleet(src, target_id, target_idx, traj, world_traj, radii,
                      avail, player, kmax=P.K_MAX_GLOBAL):
    """Smallest fleet (<= avail) from `src` that captures planet `target_id`,
    using `traj` (a simulate() output) for the target's per-turn owner, garrison
    and position. Returns (n_ships, launch_angle) or None if uncapturable.

    src        : source planet row [id, owner, x, y, radius, ships, ...].
    target_idx : index of the target in P.build_world order (for aim/world_traj).
    avail      : ships currently available at the source."""
    sx, sy, src_r = src[2], src[3], src[4]
    owner_seq, ships_seq, posx, posy = traj[target_id]

    # Seed: smallest fleet over all forecast turns, n(t) = max(ships to ARRIVE by
    # turn t, garrison@t + 1). The max resolves the speed<->size coupling.
    n0 = None
    for t in range(1, len(ships_seq) + 1):
        o = owner_seq[t - 1]
        if o == GONE or o == player:                   # gone (comet) / already ours -> skip
            continue
        reach = math.hypot(posx[t - 1] - sx, posy[t - 1] - sy) - (src_r + P.SPAWN_GAP)
        n_arr = ships_for_speed(reach / t)
        if n_arr is None:                              # too far to arrive by turn t
            continue
        n = max(n_arr, int(ships_seq[t - 1]) + 1)
        if n <= avail and (n0 is None or n < n0):
            n0 = n
    if n0 is None:
        return None                                    # can't capture even with everything

    # Cheapest fleet first -- one aim call in the common (unblocked) case.
    hit = _try_fleet(n0, src, target_idx, ships_seq, world_traj, radii, kmax)
    if hit:
        return hit

    # Blocked: a faster (bigger) fleet crosses the path EARLIER, so a moving
    # planet/comet sitting on the line may not be there yet -- this can unblock
    # even a STATIC target (same line, different timing). Scan up to _MAX_SCAN
    # larger sizes at equal speed steps; the smallest success wins.
    v0, vhi = P.fleet_speed(n0), P.fleet_speed(avail)
    if vhi <= v0:
        return None
    step = (vhi - v0) / _MAX_SCAN
    seen = set()
    for i in range(1, _MAX_SCAN + 1):
        n = ships_for_speed(v0 + i * step)
        if n is None or n <= n0 or n > avail or n in seen:
            continue
        seen.add(n)
        hit = _try_fleet(n, src, target_idx, ships_seq, world_traj, radii, kmax)
        if hit:
            return hit
    return None


def decisions_to_moves(obs, decisions, ids, ships, kmax=P.K_MAX_GLOBAL, horizon=20,
                       traj=None):
    """Convert model decisions [(src_idx, tgt_idx, mode), ...] into engine moves
    [[from_planet_id, angle_rad, num_ships], ...]. The amount is NOT in the
    decision -- `mode` selects how the heuristic sizes the fleet:
        mode 0 (min-capture): forecast.min_capture_fleet -> smallest reaching,
                              capturing fleet (None -> launch dropped).
        mode 1 (all-in):      send the whole garrison, aimed at the target.
    `ids[idx]` maps a model planet index to its planet id; `ships[idx]` is the
    source's available garrison. `horizon` must match features.HORIZON (the
    forecast depth the policy was trained on)."""
    if not decisions:
        return []
    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in raw}
    world = P.build_world(obs)
    world_traj = [[w["predict"](k) for k in range(kmax + 1)] for w in world]
    radii = [w["radius"] for w in world]
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    if traj is None:                                    # reuse extract's forecast if given
        traj = simulate(obs, horizon, eta_cache=None)
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)

    moves = []
    for src, tgt, mode in decisions:
        src_id, tgt_id = int(ids[src]), int(ids[tgt])
        sp = by_id.get(src_id)
        avail = int(ships[src])
        if sp is None or tgt_id not in id_to_idx or avail < 1:
            continue
        tidx = id_to_idx[tgt_id]
        if mode == 1:                                   # all-in: send everything
            n = avail
            angle = P.aim(sp, world_traj, radii, tidx, P.fleet_speed(n), kmax)
            if angle is None:
                continue
        else:                                           # min-capture
            res = min_capture_fleet(sp, tgt_id, tidx, traj, world_traj, radii,
                                    avail, player, kmax)
            if res is None:
                continue
            n, angle = res
        moves.append([src_id, angle, int(n)])
    return moves
"""Shared Orbit Wars physics — geometry, fleet speed, and body-motion prediction.

Both the agent (main.py, deciding where to aim) and the replay parser (replay.py,
resolving which planet a launch hit) need the same model of how the board moves:

  * fleets travel in straight lines at a size-dependent speed;
  * orbiting planets rotate about the sun, comets follow precomputed paths,
    static planets stay put;
  * collisions are *continuous* -- a fleet hits a body when their per-turn path
    segments come within the body's radius (orbit_star.md).

Keeping this in one module ensures the parser's labels and the agent's aiming use
identical geometry, so they never disagree about what a launch would hit.
"""

import math

# Board / physics constants (orbit_star.md). shipSpeed isn't in the observation,
# so we assume the default 6.0.
BOARD = 100.0
CENTER = (50.0, 50.0)
ROTATION_LIMIT = 50.0          # orbital_radius + planet_radius < this -> orbits
MAX_SPEED = 6.0
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0


def _obs_get(obs, key, default):
    """Read an observation field whether obs is a dict or an attribute object."""
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def fleet_speed(ships):
    """Fleet speed scales with size on a log curve (orbit_star.md), capped at
    MAX_SPEED -- the engine clamps with min(speed, max_speed), so fleets over
    ~1000 ships do not exceed the cap (orbit_wars.py interpreter)."""
    speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(speed, MAX_SPEED)


def seg_point_dist(ax, ay, bx, by, px, py):
    """Distance from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def seg_seg_dist(ax, ay, bx, by, cx, cy, dx, dy):
    """Minimum distance between segment AB and segment CD (2D)."""
    def ccw(ox, oy, px, py, qx, qy):
        return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)
    d1 = ccw(cx, cy, dx, dy, ax, ay)
    d2 = ccw(cx, cy, dx, dy, bx, by)
    d3 = ccw(ax, ay, bx, by, cx, cy)
    d4 = ccw(ax, ay, bx, by, dx, dy)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0  # segments intersect
    return min(
        seg_point_dist(ax, ay, bx, by, cx, cy),
        seg_point_dist(ax, ay, bx, by, dx, dy),
        seg_point_dist(cx, cy, dx, dy, ax, ay),
        seg_point_dist(cx, cy, dx, dy, bx, by),
    )


def swept_dist(ax, ay, bx, by, cx, cy, dx, dy):
    """Closest approach between two points moving *simultaneously* over one turn.

    Point 1 travels (ax,ay)->(bx,by) and point 2 travels (cx,cy)->(dx,dy), both
    linearly over t in [0, 1]. Their separation R(t) = P1(t) - P2(t) is itself
    linear (from R0 = (ax-cx, ay-cy) to R1 = (bx-dx, by-dy)), so the minimum gap
    during the turn is the origin's distance to that segment.

    This is the correct continuous-collision test for a fleet vs a moving body:
    seg_seg_dist is wrong here because it compares the two points at *different*
    times and so reports phantom crossings where the paths only cross in space.
    """
    return seg_point_dist(ax - cx, ay - cy, bx - dx, by - dy, 0.0, 0.0)


def comet_predictors(obs):
    """Map comet planet id -> (path, base_index) for future-position lookup."""
    out = {}
    for g in (_obs_get(obs, "comets", None) or []):
        paths = g.get("paths") or []
        base = g.get("path_index", 0)
        for i, pid in enumerate(g.get("planet_ids") or []):
            if i < len(paths):
                out[int(pid)] = (paths[i], base)
    return out


def predict_fn(planet, angular_velocity, comet_pred):
    """Return predict(k) -> (x, y): the body's position k turns from now (or None
    once a comet has left the board).

    Comet -> look up its precomputed path. Orbiting planet -> rotate about the
    sun. Static planet -> constant position.
    """
    pid, x, y, radius = int(planet[0]), float(planet[2]), float(planet[3]), float(planet[4])
    cx, cy = CENTER

    if pid in comet_pred:
        # path_index is one path point per turn (integer in all observed replays),
        # so the k-turn-ahead position is just path[path_index + k] -- no interp.
        path, base = comet_pred[pid]
        last = len(path) - 1
        base = int(base)

        def f(k, path=path, base=base, last=last):
            s = base + k
            if s > last:
                return None              # comet has left the board
            if s < 0:
                return path[0][0], path[0][1]
            return path[s][0], path[s][1]
        return f

    R = math.hypot(x - cx, y - cy)
    if R + radius < ROTATION_LIMIT and abs(angular_velocity) > 0.0:
        theta0 = math.atan2(y - cy, x - cx)

        def f(k, R=R, theta0=theta0, av=angular_velocity, cx=cx, cy=cy):
            th = theta0 + av * k
            return cx + R * math.cos(th), cy + R * math.sin(th)
        return f

    def f(k, x=x, y=y):
        return x, y
    return f


def build_world(obs):
    """A position-predictor for every planet/comet on the board at this step:
    a list of {id, radius, predict} dicts where predict(k) -> (x, y) or None."""
    av = _obs_get(obs, "angular_velocity", 0.0)
    comet_pred = comet_predictors(obs)
    raw = _obs_get(obs, "planets", []) or []
    return [
        {"id": int(p[0]), "radius": float(p[4]),
         "predict": predict_fn(p, av, comet_pred)}
        for p in raw
    ]


# --------------------------------------------------------------------------- #
# Aiming / launch-conversion -- shared by the SL agent (main.py), the RL agent
# (main_rl.py) and self-play rollouts (train_rl_v2.py) so all three aim with
# identical geometry. (Previously copy-pasted in each.)
# --------------------------------------------------------------------------- #
K_MAX_GLOBAL = 50              # near-range search horizon (turns) for aiming
SPAWN_GAP = 0.1                # engine spawns a fleet this far outside the source

# When the direct minimal-time lead angle is blocked (sun/body/edge), aim() retries
# a small angular sweep around that intercept direction -- enough to clear a GRAZING
# block where the straight path just nicks the sun's edge or an intervening planet.
# ±3 deg only, in 0.5 deg steps: this rescues "slightly angled" shots, NOT targets
# fully behind the sun (that needs a much wider detour). Widen SWEEP_MAX to also
# route around full occlusions.
SWEEP_MAX = math.radians(3.0)
SWEEP_STEP = math.radians(0.5)


def trace_fleet(sx, sy, angle, v, world_trajectories, radii, kmax):
    """Engine-faithful straight-line trace of a fleet launched from (sx, sy):
    return (hit_index, arrival_turn) for the first body it strikes, or
    (None, None) if it crosses the sun / leaves the board first.

    Mirrors the engine's fleet step exactly -- swept continuous collision under
    simultaneous fleet+body motion, bodies tested in world order with the first
    hit winning, and NO source exclusion (the engine relies on the SPAWN_GAP
    offset, not exclusion). Same physics as forecast._fleet_eta, so an aimed
    angle is accepted only when the fleet actually reaches the intended target
    first -- this both validates the path (no intervening body / sun / edge) and
    yields the exact arrival turn the engine will produce."""
    cosA, sinA = math.cos(angle), math.sin(angle)
    px, py = sx, sy
    for k in range(1, kmax + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA
        for i, traj in enumerate(world_trajectories):
            p, q = traj[k - 1], traj[k]
            if p is None or q is None:
                continue
            if swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1]) <= radii[i]:
                return i, k
        if not (0.0 <= fx <= BOARD and 0.0 <= fy <= BOARD):
            return None, None
        if seg_point_dist(px, py, fx, fy, CENTER[0], CENTER[1]) <= SUN_RADIUS:
            return None, None
        px, py = fx, fy
    return None, None


def aim(src, world_trajectories, radii, target_idx, v, kmax=K_MAX_GLOBAL):
    """Launch angle to hit `target` in the fewest turns, or None if uncatchable
    or every catching shot is blocked. The candidate angle (with fractional-turn
    lead refinement) is verified by trace_fleet, so a returned angle is one the
    engine will actually land on the target."""
    sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
    target_r = radii[target_idx]
    target_traj = world_trajectories[target_idx]

    def pos_at(t):
        """Target position at a fractional turn t (linear interp of the integer-
        sampled trajectory). None once the body has left the board."""
        k0 = int(math.floor(t))
        a = target_traj[k0]
        if a is None:
            return None
        b = target_traj[k0 + 1] if k0 + 1 < len(target_traj) else None
        if b is None:
            return a
        f = t - k0
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

    blocked = []                   # exact intercept angles that traced as obstructed
    for k in range(1, kmax + 1):
        pos = target_traj[k]
        if pos is None:
            break
        px, py = pos
        reach = math.hypot(px - sx, py - sy) - (src_r + SPAWN_GAP)
        if abs(v * k - reach) <= target_r:
            # The integer-turn solution aims where the target IS at turn k, but a
            # fleet covering v/turn meets a MOVING target at a fractional turn
            # t* < k -- aiming at the turn-k spot over-leads and grazes small
            # orbiting planets by a fraction of a step. Refine t* as the root of
            # f(t) = v*t - reach(t) in [k-1, k] and aim at the interpolated spot.
            # (Static target: reach is constant, so the spot/angle are unchanged.)
            lo, hi = float(k - 1), float(k)
            def f(t):
                p = pos_at(t)
                if p is None:
                    return None
                return v * t - (math.hypot(p[0] - sx, p[1] - sy) - (src_r + SPAWN_GAP))
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
            # Spawn the fleet exactly where the engine does (SPAWN_GAP outside the
            # source) and trace it with the engine's own collision physics. Accept
            # the angle only if the fleet reaches the intended target first; this
            # rejects shots blocked by an intervening body/sun/edge and guarantees
            # the same landing the engine produces.
            ox = sx + (src_r + SPAWN_GAP) * math.cos(angle)
            oy = sy + (src_r + SPAWN_GAP) * math.sin(angle)
            hit_idx, _ = trace_fleet(ox, oy, angle, v, world_trajectories, radii, kmax)
            if hit_idx == target_idx:
                return angle
            blocked.append(angle)            # exact shot obstructed -> sweep candidate

    # Fallback: every exact lead angle was blocked. Nudge each blocked candidate
    # (soonest-arrival first) by up to ±SWEEP_MAX, smallest nudge first, and accept
    # the first angle trace_fleet confirms still lands on the target -- threading a
    # grazing sun/planet edge. Early-exit on the first hit keeps this cheap.
    n_steps = int(round(SWEEP_MAX / SWEEP_STEP))
    for base in blocked:
        for j in range(1, n_steps + 1):
            for angle in (base + j * SWEEP_STEP, base - j * SWEEP_STEP):
                ox = sx + (src_r + SPAWN_GAP) * math.cos(angle)
                oy = sy + (src_r + SPAWN_GAP) * math.sin(angle)
                hit_idx, _ = trace_fleet(ox, oy, angle, v, world_trajectories, radii, kmax)
                if hit_idx == target_idx:
                    return angle
    return None


def decisions_to_moves(obs, decisions, ids, ships, kmax=K_MAX_GLOBAL):
    """Convert policy decisions [(src_idx, tgt_idx, frac), ...] into the env move
    format [[from_planet_id, angle_rad, num_ships], ...]. Precomputes each body's
    trajectory once, then aims every launch. Skips launches that round to <1 ship,
    have a missing/invalid source/target, or no clear (unblocked) angle."""
    if not decisions:
        return []
    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in raw}
    world = build_world(obs)
    world_trajectories = [[w["predict"](k) for k in range(kmax + 1)] for w in world]
    radii = [w["radius"] for w in world]
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    moves = []
    for src, tgt, frac in decisions:
        available = float(ships[src])
        n = min(int(round(frac * available)), int(available))
        if n < 1:
            continue
        spl = by_id.get(int(ids[src]))
        target_id = int(ids[tgt])
        if spl is None or target_id not in id_to_idx:
            continue
        target_idx = id_to_idx[target_id]
        angle = aim(spl, world_trajectories, radii, target_idx, fleet_speed(n), kmax=kmax)
        if angle is None:
            continue
        moves.append([int(ids[src]), angle, n])
    return moves

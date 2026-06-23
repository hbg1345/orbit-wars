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
    """Fleet speed scales with size on a log curve (orbit_star.md)."""
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5


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


def path_blocked(sx, sy, angle, v, world_trajectories, radii, target_idx, k_hit):
    """True if, before reaching the target at turn k_hit, the fleet's straight
    path crosses the sun, leaves the board, or hits another planet/comet."""
    cosA, sinA = math.cos(angle), math.sin(angle)
    px, py = sx, sy
    prev = [traj[0] for traj in world_trajectories]   # current positions, O(1)
    for k in range(1, k_hit + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA
        if seg_point_dist(px, py, fx, fy, CENTER[0], CENTER[1]) <= SUN_RADIUS:
            return True
        if not (0.0 <= fx <= BOARD and 0.0 <= fy <= BOARD):
            return True
        for i, traj in enumerate(world_trajectories):
            if i == target_idx:
                continue
            q = traj[k]
            if q is None:
                continue
            p = prev[i]
            prev[i] = q
            if p is None:
                continue
            if swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1]) <= radii[i]:
                return True
        px, py = fx, fy
    return False


def aim(src, world_trajectories, radii, target_idx, v, kmax=K_MAX_GLOBAL):
    """Launch angle to hit `target`, or None if uncatchable / path blocked."""
    sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
    target_r = radii[target_idx]
    target_traj = world_trajectories[target_idx]
    for k in range(1, kmax + 1):
        pos = target_traj[k]
        if pos is None:
            break
        px, py = pos
        reach = math.hypot(px - sx, py - sy) - src_r
        if abs(v * k - reach) <= target_r:
            angle = math.atan2(py - sy, px - sx)
            ox = sx + (src_r + 0.01) * math.cos(angle)
            oy = sy + (src_r + 0.01) * math.sin(angle)
            if not path_blocked(ox, oy, angle, v, world_trajectories, radii, target_idx, k):
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

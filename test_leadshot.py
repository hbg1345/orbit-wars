"""Lead-shot test harness.

Fire at a rotating (orbiting) target and sweep fleet sizes 1..500. For each
size we compute the launch angle two ways -- naive (aim at current position)
and lead (intercept) -- then simulate the fleet against the game's physics and
report whether it hits the moving planet.

Physics mirrors orbit_star.md and main.py (validated against real replays):
  * fleet speed = 1 + (maxSpeed-1)*(log(ships)/log(1000))^1.5
  * orbiting planet rotates theta += angular_velocity each turn
  * turn order: move fleet (continuous collision) -> rotate planet (sweep)
  * a fleet dies on the sun (r=10) or going out of bounds

Usage:
  python test_leadshot.py
"""

import math

import main  # reuse the exact _lead_angle / _fleet_speed the agent uses

CENTER = (50.0, 50.0)
SUN_R = 10.0
BOARD = 100.0
MAX_TURNS = 400

# --- Scenario (realistic magnitudes from the replays) --------------------- #
SOURCE = (12.0, 50.0)        # one of our (static) planets
TARGET_RADIUS_ORBIT = 38.0   # target's orbital radius around the sun
TARGET_PLANET_RADIUS = 1.0   # the planet's own size (like an orbiting comet/planet)
TARGET_START_THETA = 0.6     # where the target sits on its orbit at launch (rad)
ANGULAR_VELOCITY = 0.0466    # measured from replay


def _seg_point_dist(ax, ay, bx, by, px, py):
    """Distance from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def simulate(ships, lead=True):
    """Fire `ships` at the orbiting target; return (hit, turns, min_dist)."""
    cx, cy = CENTER
    R, trad, av = TARGET_RADIUS_ORBIT, TARGET_PLANET_RADIUS, ANGULAR_VELOCITY
    theta = TARGET_START_THETA
    tx0, ty0 = cx + R * math.cos(theta), cy + R * math.sin(theta)
    sx, sy = SOURCE

    if lead:
        angle = main._lead_angle(sx, sy, tx0, ty0, trad, ships, av)
    else:
        angle = math.atan2(ty0 - sy, tx0 - sx)

    v = main._fleet_speed(ships)
    fx, fy = sx, sy
    min_dist = math.inf
    for turn in range(1, MAX_TURNS + 1):
        nfx, nfy = fx + v * math.cos(angle), fy + v * math.sin(angle)

        # Target position this turn (before it rotates).
        tx, ty = cx + R * math.cos(theta), cy + R * math.sin(theta)
        min_dist = min(min_dist, _seg_point_dist(fx, fy, nfx, nfy, tx, ty))
        if _seg_point_dist(fx, fy, nfx, nfy, tx, ty) < trad:
            return True, turn, 0.0
        if _seg_point_dist(fx, fy, nfx, nfy, cx, cy) < SUN_R:
            return False, turn, min_dist  # destroyed by sun
        if not (0.0 <= nfx <= BOARD and 0.0 <= nfy <= BOARD):
            return False, turn, min_dist  # out of bounds
        fx, fy = nfx, nfy

        # Planet rotates; check if the moving planet sweeps onto the fleet.
        theta += av
        tx2, ty2 = cx + R * math.cos(theta), cy + R * math.sin(theta)
        d = math.hypot(fx - tx2, fy - ty2)
        min_dist = min(min_dist, d)
        if d < trad:
            return True, turn, 0.0

    return False, MAX_TURNS, min_dist


def main_test():
    sizes = [1] + list(range(10, 501, 10))
    print(f"target: orbiting r={TARGET_RADIUS_ORBIT} planet_radius={TARGET_PLANET_RADIUS} "
          f"av={ANGULAR_VELOCITY}  source={SOURCE}")
    print(f"{'ships':>5} | {'speed':>6} | {'lead':>10} {'turns':>5} | {'naive':>10} {'miss':>6}")
    print("-" * 56)
    lead_hits = naive_hits = 0
    for n in sizes:
        v = main._fleet_speed(n)
        h_l, t_l, _ = simulate(n, lead=True)
        h_n, t_n, md_n = simulate(n, lead=False)
        lead_hits += h_l
        naive_hits += h_n
        lead_s = f"HIT t={t_l}" if h_l else "MISS"
        naive_s = "HIT" if h_n else f"MISS"
        miss_s = "" if h_n else f"{md_n:.1f}"
        print(f"{n:>5} | {v:>6.2f} | {lead_s:>10} {t_l:>5} | {naive_s:>10} {miss_s:>6}")
    total = len(sizes)
    print("-" * 56)
    print(f"LEAD  hit rate: {lead_hits}/{total}")
    print(f"NAIVE hit rate: {naive_hits}/{total}")


if __name__ == "__main__":
    main_test()

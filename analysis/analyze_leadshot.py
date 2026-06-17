"""Relationship between the feasible (arrival-turn -> ships) menu and the ships
humans actually launched.

For each real winner launch we know the source, the resolved target, and the
ship count. A fleet of `ships` flies at fleet_speed(ships); to hit the target at
turn k it must satisfy v*k ~= reach_k, where reach_k is the distance from the
source to the target's predicted position at turn k (minus the source radius).
Inverting fleet_speed gives, per reachable turn k, the ship count that lands the
fleet exactly then -- the "menu". The human picks one point on this menu by
choosing how many ships to send.
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics as P
import replay

LN1000 = math.log(1000.0)


def invert_speed(v):
    """Ship count whose fleet_speed == v  (v in [1, inf)). Inverse of
    fleet_speed: v = 1 + 5*(ln(ships)/ln1000)**1.5."""
    if v <= 1.0:
        return 1.0
    return math.exp(LN1000 * ((v - 1.0) / (P.MAX_SPEED - 1.0)) ** (1.0 / 1.5))


def launches(path):
    """Yield (ships_actual, k_actual, ships_min, menu_pts) for each winner launch
    that resolves to a target visible on the launch board."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    steps = d["steps"]
    fleet_idx = replay._index_fleets(steps)
    winner = int(np.argmax(d["rewards"]))

    for t in range(1, len(steps)):
        obs = steps[t - 1][winner].get("observation", {})    # launch board
        planets = obs.get("planets")
        if not planets:
            continue
        action = steps[t][winner].get("action") or []
        by_id = {int(p[0]): p for p in planets}
        body_at = replay._body_track(P.build_world(steps[t][0]["observation"]))
        world = {w["id"]: w for w in P.build_world(obs)}

        for mv in action:
            fid, ang, nsh = int(mv[0]), float(mv[1]), int(mv[2])
            src = by_id.get(fid)
            if src is None or int(src[1]) != winner or nsh < 1:
                continue
            tgt_id = replay._resolve_target(fleet_idx, t, fid, ang, nsh, body_at)
            if tgt_id is None or tgt_id not in by_id or tgt_id not in world:
                continue
            sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
            target_r = float(by_id[tgt_id][4])
            tpred = world[tgt_id]["predict"]
            v_actual = P.fleet_speed(nsh)

            menu, k_actual = [], None
            for k in range(1, 401):
                pos = tpred(k)
                if pos is None:
                    break
                reach = math.hypot(pos[0] - sx, pos[1] - sy) - src_r
                if reach <= 0:
                    continue
                v_need = reach / k
                if v_need >= 1.0:
                    menu.append((k, invert_speed(v_need)))
                if k_actual is None and abs(v_actual * k - reach) <= target_r:
                    k_actual = k
            if not menu or k_actual is None:
                continue
            ships_min = min(s for _, s in menu)
            yield nsh, k_actual, ships_min, menu


def main():
    paths = sorted(glob.glob("data/*.json"))[:40]
    act_k, act_ships, ships_min = [], [], []
    menu_k, menu_ships = [], []
    for path in paths:
        for nsh, k_actual, smin, menu in launches(path):
            act_k.append(k_actual)
            act_ships.append(nsh)
            ships_min.append(smin)
            for k, s in menu:
                if k <= 120 and s <= 5000:
                    menu_k.append(k)
                    menu_ships.append(s)

    act_k = np.array(act_k)
    act_ships = np.array(act_ships, dtype=float)
    menu_k = np.array(menu_k)
    menu_ships = np.array(menu_ships, dtype=float)
    print(f"{len(paths)} replays, {len(act_k)} resolved winner launches")

    # Shared axes so the two panels are directly comparable.
    xlim = (0, 120)
    ylim = (1, 5000)
    xbins = np.linspace(*xlim, 60)
    ybins = np.logspace(0, np.log10(ylim[1]), 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

    # Panel 1: the feasible menu -- for each reachable turn k, the ship count that
    # lands a fleet exactly then (aggregated over all launches' targets).
    h1 = ax1.hist2d(menu_k, menu_ships, bins=[xbins, ybins], cmap="Blues",
                    norm=matplotlib.colors.LogNorm())
    ax1.set_yscale("log")
    ax1.set_xlabel("arrival turn k")
    ax1.set_ylabel("fleet size (ships)")
    ax1.set_title("feasible: ships needed to arrive at turn k")
    fig.colorbar(h1[3], ax=ax1, label="# (launch, turn) options")

    # Panel 2: what humans actually sent -- each launch's ship count placed at the
    # turn that ship count actually arrives.
    h2 = ax2.hist2d(act_k, act_ships, bins=[xbins, ybins], cmap="Reds",
                    norm=matplotlib.colors.LogNorm())
    ax2.set_yscale("log")
    ax2.set_xlabel("arrival turn k")
    ax2.set_title("actual: fleet size launched vs its arrival turn")
    fig.colorbar(h2[3], ax=ax2, label="# launches")

    fig.tight_layout()
    fig.savefig("reports/leadshot_relation.png", dpi=110)
    print("saved leadshot_relation.png")
    print(f"actual arrival turn k: median={np.median(act_k):.0f} "
          f"p90={np.percentile(act_k,90):.0f}")
    print(f"actual fleet size: median={np.median(act_ships):.0f} "
          f"p90={np.percentile(act_ships,90):.0f}")

    # Ship-count distribution at INTEGER resolution -- the 2D log-binned heatmap
    # can show false "gaps" where integer counts straddle log bins, so look at the
    # raw counts directly.
    ints = act_ships.astype(int)
    counts = np.bincount(ints)
    print("ship-count tally (ships: #launches):")
    for s in range(0, min(21, len(counts))):
        print(f"  {s:2d}: {counts[s]}")

    fig2, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(np.arange(1, 61), counts[1:61] if len(counts) > 61 else
           np.pad(counts[1:], (0, max(0, 61 - len(counts[1:])))), color="#C44E52")
    ax.set_xlabel("fleet size (ships, integer)")
    ax.set_ylabel("# launches")
    ax.set_title("actual fleet-size distribution (1..60, integer bins)")
    fig2.tight_layout()
    fig2.savefig("reports/fleet_size_dist.png", dpi=110)
    print("saved fleet_size_dist.png")


if __name__ == "__main__":
    main()
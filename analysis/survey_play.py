"""Characterize human (winner) launch decisions across the FULL replay set.

Produces a 4-panel overview:
  A) players per game (2p vs 4p)
  B) fleets launched from one source in one turn (the one-target assumption)
  C) fleet-size distribution (integer ship counts)
  D) NOOP vs launch share of owned-planet decisions

Everything here is read straight from the action / observation fields (no forward
sim), so it parallelizes across replays.
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def scan(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    steps, rewards = d.get("steps"), d.get("rewards")
    if not steps or not rewards:
        return None
    winner = int(np.argmax(rewards))

    ship = Counter()          # ship count -> #launches
    persrc = Counter()        # fleets-from-one-source-in-a-turn -> count
    owned = noop = launch = 0
    for t in range(1, len(steps)):
        obs = steps[t - 1][winner].get("observation", {})
        planets = obs.get("planets")
        if not planets:
            continue
        owned_ids = {int(p[0]) for p in planets if int(p[1]) == winner}
        owned += len(owned_ids)
        action = steps[t][winner].get("action") or []
        srcs = Counter(int(mv[0]) for mv in action if int(mv[0]) in owned_ids)
        launch += len(srcs)
        noop += max(0, len(owned_ids) - len(srcs))
        for c in srcs.values():
            persrc[c] += 1
        for mv in action:
            ship[int(mv[2])] += 1
    return len(rewards), ship, persrc, owned, noop, launch


def main():
    paths = sorted(glob.glob("data/*.json"))
    paths = [p for p in paths if "kaggle" not in p.lower()]
    workers = os.cpu_count() or 1
    print(f"{len(paths)} replays, {workers} workers")

    nplayers = Counter()
    ship = Counter()
    persrc = Counter()
    owned = noop = launch = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(scan, paths), total=len(paths), unit="ep"):
            if r is None:
                skipped += 1
                continue
            npl, s, ps, ow, no, la = r
            nplayers[npl] += 1
            ship.update(s)
            persrc.update(ps)
            owned += ow
            noop += no
            launch += la

    print(f"skipped {skipped}; player counts {dict(sorted(nplayers.items()))}")
    print(f"owned-planet decisions: {owned}  NOOP {noop} ({100*noop/max(owned,1):.1f}%)  "
          f"launch {launch} ({100*launch/max(owned,1):.1f}%)")
    total_src = sum(persrc.values())
    print("launches per source per turn: " +
          ", ".join(f"{c}->{persrc[c]} ({100*persrc[c]/total_src:.2f}%)" for c in sorted(persrc)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # A) players per game
    ax = axes[0, 0]
    ks = sorted(nplayers)
    ax.bar([str(k) for k in ks], [nplayers[k] for k in ks], color="#4C72B0")
    ax.set_xlabel("players in game")
    ax.set_ylabel("# replays")
    ax.set_title(f"A) game size ({sum(nplayers.values())} replays)")

    # B) fleets per source per turn
    ax = axes[0, 1]
    ks = sorted(persrc)
    ax.bar([str(k) for k in ks], [100 * persrc[k] / total_src for k in ks], color="#C44E52")
    ax.set_xlabel("fleets launched from one source in one turn")
    ax.set_ylabel("% of launching sources")
    ax.set_yscale("log")
    ax.set_title("B) one-target-per-source assumption")

    # C) fleet-size distribution
    ax = axes[1, 0]
    maxs = 60
    counts = np.array([ship.get(s, 0) for s in range(1, maxs + 1)])
    ax.bar(np.arange(1, maxs + 1), counts, color="#55A868")
    ax.set_xlabel("fleet size (ships, integer)")
    ax.set_ylabel("# launches")
    ax.set_title("C) fleet-size distribution (round-number comb)")

    # D) NOOP vs launch
    ax = axes[1, 1]
    ax.bar(["NOOP\n(no launch)", "launch"], [noop, launch], color=["#8172B3", "#CCB974"])
    for i, v in enumerate([noop, launch]):
        ax.text(i, v, f"{100*v/max(owned,1):.1f}%", ha="center", va="bottom")
    ax.set_ylabel("# owned-planet decisions")
    ax.set_title(f"D) NOOP vs launch  (ratio {noop/max(launch,1):.1f}:1)")

    fig.suptitle(f"Human (winner) play-data survey — full set, {sum(nplayers.values())} replays",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("reports/play_survey.png", dpi=110)
    print("saved play_survey.png")


if __name__ == "__main__":
    main()
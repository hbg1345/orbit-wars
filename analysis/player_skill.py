"""Rank players by win rate across the full replay set (no external leaderboard
needed -- replays carry info.TeamNames + rewards). Lets us filter training to
games won by strong players ("상위권") instead of imitating every winner.
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

MIN_GAMES = 30   # ignore players with too few games for a stable win rate


def scan(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    names = d.get("info", {}).get("TeamNames")
    rewards = d.get("rewards")
    if not names or not rewards:
        return None
    winner = int(np.argmax(rewards))
    return tuple(names), winner


def main():
    paths = [p for p in sorted(glob.glob("data/*.json")) if "kaggle" not in p.lower()]
    workers = os.cpu_count() or 1
    print(f"{len(paths)} replays, {workers} workers")

    games = Counter()
    wins = Counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(scan, paths), total=len(paths), unit="ep"):
            if r is None:
                continue
            names, winner = r
            results.append(r)
            for i, nm in enumerate(names):
                games[nm] += 1
                if i == winner:
                    wins[nm] += 1

    rated = [(nm, wins[nm], games[nm], wins[nm] / games[nm])
             for nm in games if games[nm] >= MIN_GAMES]
    rated.sort(key=lambda x: -x[3])
    print(f"\n{len(games)} distinct players; {len(rated)} with >= {MIN_GAMES} games")
    print(f"{'player':<24}{'win%':>6}{'wins':>7}{'games':>7}")
    for nm, w, g, wr in rated[:20]:
        print(f"  {nm[:22]:<22}{100*wr:>6.1f}{w:>7}{g:>7}")

    # How much TRAINING data each "top-K by win rate" cohort would give
    # (replays whose winner is in the cohort).
    top_sorted = [nm for nm, *_ in rated]
    def coverage(k):
        top = set(top_sorted[:k])
        return sum(1 for names, winner in results if names[winner] in top)
    ks = [5, 10, 20, 30, 50, 100]
    print("\nreplays where the winner is a top-K player:")
    for k in ks:
        if k <= len(top_sorted):
            print(f"  top {k:>3}: {coverage(k)} replays")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    top20 = rated[:20]
    ax1.barh([nm[:20] for nm, *_ in top20][::-1],
             [100 * wr for *_, wr in top20][::-1], color="#4C72B0")
    ax1.set_xlabel("win rate (%)")
    ax1.set_title(f"top 20 players by win rate (>= {MIN_GAMES} games)")

    ax2.plot(ks, [coverage(k) for k in ks if k <= len(top_sorted)], "o-", color="#C44E52")
    ax2.set_xlabel("cohort size (top-K players by win rate)")
    ax2.set_ylabel("# replays won by the cohort")
    ax2.set_title("training data available per top-K cohort")

    fig.tight_layout()
    fig.savefig("reports/player_skill.png", dpi=110)
    print("\nsaved player_skill.png")


if __name__ == "__main__":
    main()
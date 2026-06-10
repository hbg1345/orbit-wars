"""Analyze the leaderboard #1 player's launching behaviour over one day of replays.

Two distributions, computed only from steps the rank-1 team actually played:

  1. launches-per-step  -- how many fleets that player launches in a single turn
                           (counted as the # of moves in its action list).
  2. launch-fraction    -- when it launches from a planet, what share of that
                           planet's garrison it sends: num_ships / source_n_ships,
                           read off the pre-action board obs[t-1] (the board the
                           action was decided on -- see replay.py). Same quantity
                           as replay.py's `ship_label`.

  python analyze_rank1_launches.py                       # default zip + rank-1 team
  python analyze_rank1_launches.py data/zips/orbit-wars-episodes-2026-06-06.zip
  python analyze_rank1_launches.py --team "Jake Will"    # any team by name
  python analyze_rank1_launches.py --rank 3              # nth team on the board
  python analyze_rank1_launches.py a.zip b.json ...      # explicit files

Reads the leaderboard CSV (data/orbit-wars-publicleaderboard-*.csv) to resolve
which TeamName is rank N. Episode JSONs carry info.TeamNames[p] per player slot.
"""

import argparse
import csv
import glob
import io
import json
import os
import zipfile
from collections import Counter

import numpy as np


def leaderboard_team(rank):
    """TeamName at the given leaderboard rank (1-based)."""
    csvs = sorted(glob.glob("data/orbit-wars-publicleaderboard-*.csv"))
    if not csvs:
        raise SystemExit("no leaderboard CSV under data/")
    with open(csvs[-1], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["Rank"]) == rank:
            return r["TeamName"]
    raise SystemExit(f"rank {rank} not found in {csvs[-1]}")


def iter_episodes(paths):
    """Yield loaded replay dicts from .json files and .zip archives (streamed)."""
    for p in paths:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".json"):
                        with zf.open(n) as fh:
                            yield json.load(io.TextIOWrapper(fh, encoding="utf-8"))
        elif p.lower().endswith(".json"):
            with open(p, encoding="utf-8") as fh:
                yield json.load(fh)


def analyze(paths, team, limit=None):
    per_step = []        # # launches in each turn the team acted (incl. 0)
    fractions = []       # garrison share sent per individual launch
    n_episodes = 0
    n_games_seen = 0

    for d in iter_episodes(paths):
        n_games_seen += 1
        tns = d.get("info", {}).get("TeamNames") or []
        if team not in tns:
            continue
        p = tns.index(team)
        n_episodes += 1
        steps = d["steps"]
        for t in range(1, len(steps)):
            action = steps[t][p].get("action") or []     # decided on obs[t-1]
            per_step.append(len(action))
            if not action:
                continue
            obs_prev = steps[t - 1][p].get("observation") or {}
            # planet id -> garrison (n_ships) on the pre-action board
            garrison = {int(pl[0]): float(pl[5]) for pl in (obs_prev.get("planets") or [])}
            for mv in action:
                fid, nsh = int(mv[0]), float(mv[2])
                g = garrison.get(fid)
                if g is None or g <= 0:
                    continue
                fractions.append(min(nsh / g, 1.0))

        if limit and n_episodes >= limit:
            break

    return per_step, fractions, n_episodes, n_games_seen


def hist_text(values, bins, label):
    """Compact text histogram so this works without a display."""
    counts, edges = np.histogram(values, bins=bins)
    total = counts.sum() or 1
    lines = [f"\n{label}  (n={len(values)}, mean={np.mean(values):.3f}, "
             f"median={np.median(values):.3f})"]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * c / counts.max()) if counts.max() else ""
        lines.append(f"  [{lo:6.2f}, {hi:6.2f})  {c:7d}  {100*c/total:5.1f}%  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help=".zip/.json replays (default: latest day zip)")
    ap.add_argument("--team", help="team name (default: leaderboard rank-1)")
    ap.add_argument("--rank", type=int, default=1, help="leaderboard rank to pick (default 1)")
    ap.add_argument("--limit", type=int, default=20,
                    help="stop after this many of the team's episodes (default 20; 0 = all)")
    ap.add_argument("--plot", action="store_true", help="also save PNG histograms")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("data/zips/orbit-wars-episodes-*.zip"))[-1:]
    if not paths:
        raise SystemExit("no replays found; pass a .zip/.json or fill data/zips/")
    team = args.team or leaderboard_team(args.rank)

    print(f"team    : {team!r}")
    print(f"replays : {', '.join(os.path.basename(p) for p in paths)}")

    per_step, fractions, n_ep, n_seen = analyze(paths, team, limit=args.limit or None)
    print(f"games   : {n_ep} with this team / {n_seen} scanned")
    if not per_step:
        raise SystemExit("team did not appear in these replays")

    ps = np.array(per_step)
    print(f"\nlaunches-per-step: turns acted={len(ps)}  "
          f"silent(0 launches)={100*np.mean(ps == 0):.1f}%  max={ps.max()}")
    for k in range(0, int(ps.max()) + 1):
        c = int((ps == k).sum())
        if c:
            print(f"  {k:2d} launches : {c:7d}  {100*c/len(ps):5.1f}%  "
                  f"{'#' * int(40 * c / len(ps))}")

    print(hist_text(fractions, np.linspace(0, 1, 21),
                    "launch-fraction (garrison share sent per launch)"))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(ps, bins=range(0, int(ps.max()) + 2), align="left", rwidth=0.85)
        ax[0].set(title=f"{team}: launches per step", xlabel="# fleets launched", ylabel="turns")
        ax[1].hist(fractions, bins=np.linspace(0, 1, 21), rwidth=0.9)
        ax[1].set(title=f"{team}: launch fraction", xlabel="ships sent / garrison", ylabel="launches")
        fig.tight_layout()
        out = "rank1_launches.png"
        fig.savefig(out, dpi=110)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

"""Pull ONE replay from the leaderboard rank-1 team and test two hypotheses
about how that player launches fleets:

  H1  "It launches *all* the ships it has, from *every* owned planet, every turn."
      Split into two measurable claims:
        H1a (full-garrison)  : when a planet launches, num_ships == its garrison.
        H1b (every-planet)   : every owned planet with ships>0 launches each turn.

  H2  "No planet launches on two or more *consecutive* turns."
      For each owned planet id, look at the turns it launched and count how often
      turn t and turn t+1 both fired from that same planet.

Usage:
  python analysis/analyze_rank1_single.py                 # first rank-1 episode in latest zip
  python analysis/analyze_rank1_single.py --rank 1 --index 0
  python analysis/analyze_rank1_single.py --rank 1 --best # rank-1 game it scored highest in
  python analysis/analyze_rank1_single.py data/zips/orbit-wars-episodes-2026-06-16.zip
  python analysis/analyze_rank1_single.py some_episode.json
  python analysis/analyze_rank1_single.py ... --dump rank1_episode.json   # save the picked replay

Notes:
  * Action at steps[t] is decided on the board obs[t-1] (see replay.py), so a
    launch's garrison/ownership is read off step t-1.
  * planets tuple: [id, owner, x, y, radius, n_ships, production]  (observation_format.md)
  * action: [[from_planet_id, angle, num_ships], ...]
"""

import os as _os, sys as _sys  # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import csv
import glob
import io
import json
import os
import zipfile
from collections import defaultdict

import numpy as np

FULL_TOL = 1e-6  # treat num_ships >= garrison*(1-tol) as a full-garrison launch


def leaderboard_team(rank):
    csvs = sorted(glob.glob("data/orbit-wars-publicleaderboard-*.csv"))
    if not csvs:
        raise SystemExit("no leaderboard CSV under data/")
    with open(csvs[-1], encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["Rank"]) == rank:
                return r["TeamName"]
    raise SystemExit(f"rank {rank} not found in {csvs[-1]}")


def iter_episodes(paths):
    """Yield (source_name, replay_dict)."""
    for p in paths:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".json"):
                        with zf.open(n) as fh:
                            yield n, json.load(io.TextIOWrapper(fh, encoding="utf-8"))
        elif p.lower().endswith(".json"):
            with open(p, encoding="utf-8") as fh:
                yield os.path.basename(p), json.load(fh)


def final_score(d, p):
    """The team's reward/score at the last step (best-effort)."""
    steps = d["steps"]
    last = steps[-1][p]
    for k in ("reward",):
        v = last.get(k)
        if v is not None:
            return float(v)
    return 0.0


def pick_episode(paths, team, index, best):
    """Return (name, replay, player_slot) for the chosen rank-1 episode."""
    matches = []  # (score, name, d, p)  -- only kept when --best, else short-circuit
    seen = 0
    for name, d in iter_episodes(paths):
        seen += 1
        tns = d.get("info", {}).get("TeamNames") or []
        if team not in tns:
            continue
        p = tns.index(team)
        if best:
            matches.append((final_score(d, p), name, d, p))
            continue
        if len(matches) == index:
            print(f"scanned {seen} episodes; picked match #{index}")
            return name, d, p
        matches.append((None, name, d, p))  # only counting toward index
    if best and matches:
        matches.sort(key=lambda m: m[0], reverse=True)
        s, name, d, p = matches[0]
        print(f"scanned {seen} episodes; picked best (score={s:.1f}) of "
              f"{len(matches)} rank-team games")
        return name, d, p
    raise SystemExit(f"could not find episode #{index} for team {team!r} "
                     f"(found {len(matches)} of its games in {seen} scanned)")


def analyze_episode(d, p):
    steps = d["steps"]
    n = len(steps)

    # H1a: full-garrison launches
    full = partial = 0
    frac_samples = []
    # H1b: owned-with-ships planets that launched vs not, per turn
    owned_with_ships = 0
    owned_with_ships_launched = 0
    turns_all_owned_fired = 0   # turns where every owned-with-ships planet fired
    turns_acted = 0
    # H2: per-planet launch turns
    launch_turns = defaultdict(list)   # planet_id -> [t, ...]
    per_step_counts = []

    for t in range(1, n):
        action = steps[t][p].get("action") or []
        per_step_counts.append(len(action))
        obs_prev = steps[t - 1][p].get("observation") or {}
        planets = obs_prev.get("planets") or []
        garrison = {int(pl[0]): float(pl[5]) for pl in planets}
        owned = {int(pl[0]) for pl in planets if int(pl[1]) == p}
        owned_ships = {pid for pid in owned if garrison.get(pid, 0) > 0}

        launched_ids = set()
        for mv in action:
            fid, nsh = int(mv[0]), float(mv[2])
            launch_turns[fid].append(t)
            launched_ids.add(fid)
            g = garrison.get(fid)
            if g and g > 0:
                frac = nsh / g
                frac_samples.append(min(frac, 1.0))
                if nsh >= g * (1 - FULL_TOL):
                    full += 1
                else:
                    partial += 1

        if owned_ships:
            turns_acted += 1
            owned_with_ships += len(owned_ships)
            fired = owned_ships & launched_ids
            owned_with_ships_launched += len(fired)
            if fired == owned_ships:
                turns_all_owned_fired += 1

    # H2: consecutive-turn launches from same planet
    consec_pairs = 0
    total_launch_turns = 0
    max_streak = 0
    planets_with_consec = 0
    for pid, ts in launch_turns.items():
        ts = sorted(set(ts))
        total_launch_turns += len(ts)
        streak = 1
        had_consec = False
        for a, b in zip(ts, ts[1:]):
            if b == a + 1:
                consec_pairs += 1
                streak += 1
                had_consec = True
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if had_consec:
            planets_with_consec += 1

    return {
        "n_steps": n,
        "per_step": np.array(per_step_counts),
        "full": full, "partial": partial,
        "frac": np.array(frac_samples),
        "owned_with_ships": owned_with_ships,
        "owned_with_ships_launched": owned_with_ships_launched,
        "turns_acted": turns_acted,
        "turns_all_owned_fired": turns_all_owned_fired,
        "consec_pairs": consec_pairs,
        "total_launch_turns": total_launch_turns,
        "max_streak": max(max_streak, 1) if total_launch_turns else 0,
        "n_planets_launched": len(launch_turns),
        "planets_with_consec": planets_with_consec,
        "launch_turns": {k: sorted(set(v)) for k, v in launch_turns.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help=".zip/.json (default: latest day zip)")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--team", help="override team name")
    ap.add_argument("--index", type=int, default=0, help="which rank-team game to take")
    ap.add_argument("--best", action="store_true", help="pick the rank-team's highest-scoring game")
    ap.add_argument("--dump", help="write the picked replay JSON here")
    ap.add_argument("--show-streaks", action="store_true",
                    help="list planets that fired on consecutive turns")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("data/zips/orbit-wars-episodes-*.zip"))[-1:]
    if not paths:
        raise SystemExit("no replays; pass a .zip/.json or fill data/zips/")
    team = args.team or leaderboard_team(args.rank)
    print(f"team    : {team!r}  (rank {args.rank})")
    print(f"source  : {', '.join(os.path.basename(x) for x in paths)}")

    name, d, p = pick_episode(paths, team, args.index, args.best)
    print(f"episode : {name}   (player slot {p})")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(d, f)
        print(f"dumped  : {args.dump}")

    r = analyze_episode(d, p)
    ps = r["per_step"]
    print(f"\nsteps={r['n_steps']}  turns acted (had owned ships)={r['turns_acted']}")
    print(f"launches/turn: mean={ps.mean():.2f} max={ps.max()} "
          f"silent={100*np.mean(ps==0):.1f}%")

    print("\n=== H1a  full-garrison launches (sends ALL ships) ===")
    tot = r["full"] + r["partial"]
    if tot:
        print(f"  full   : {r['full']:5d}  {100*r['full']/tot:5.1f}%")
        print(f"  partial: {r['partial']:5d}  {100*r['partial']/tot:5.1f}%")
        f = r["frac"]
        print(f"  fraction sent: mean={f.mean():.3f} median={np.median(f):.3f} "
              f"min={f.min():.3f}")
    else:
        print("  (no launches)")

    print("\n=== H1b  launches from EVERY owned planet that has ships ===")
    o = r["owned_with_ships"]
    if o:
        print(f"  owned-with-ships planet-turns        : {o}")
        print(f"  of those that launched               : {r['owned_with_ships_launched']} "
              f"({100*r['owned_with_ships_launched']/o:.1f}%)")
        print(f"  turns ALL owned-with-ships fired      : {r['turns_all_owned_fired']}"
              f"/{r['turns_acted']} ({100*r['turns_all_owned_fired']/r['turns_acted']:.1f}%)")

    print("\n=== H2  same planet on consecutive turns ===")
    print(f"  distinct planets ever launched        : {r['n_planets_launched']}")
    print(f"  total (planet,turn) launches          : {r['total_launch_turns']}")
    print(f"  consecutive-turn launch pairs         : {r['consec_pairs']}")
    print(f"  planets that ever fired 2+ in a row   : {r['planets_with_consec']}")
    print(f"  longest same-planet streak (turns)    : {r['max_streak']}")
    if r["consec_pairs"] == 0:
        print("  -> H2 HOLDS for this game: no planet fired on back-to-back turns.")
    else:
        print("  -> H2 violated: some planet fired on consecutive turns.")

    if args.show_streaks:
        print("\n  per-planet consecutive runs:")
        for pid, ts in sorted(r["launch_turns"].items()):
            runs = []
            run = [ts[0]]
            for a, b in zip(ts, ts[1:]):
                if b == a + 1:
                    run.append(b)
                else:
                    runs.append(run); run = [b]
            runs.append(run)
            longest = max(len(x) for x in runs)
            if longest >= 2:
                segs = ",".join(f"{x[0]}-{x[-1]}" for x in runs if len(x) >= 2)
                print(f"    planet {pid:3d}: {len(ts)} launches, "
                      f"consec runs [{segs}]")


if __name__ == "__main__":
    main()

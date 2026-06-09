"""Run one or more Orbit Wars games and report Player 0's win rate.

Usage:
    python run_game.py                       # main.py vs main.py (1 game)
    python run_game.py main.py random        # vs the built-in random bot
    python run_game.py main.py main.py main.py main.py   # 4-player game
    python run_game.py --games 20 main.py random         # 20 games, P0 win rate
    python run_game.py --games 40 --workers 8 main.py random   # 40 games in parallel

With --games 1 the per-step observations are dumped (obs_log.jsonl / .txt) for
inspection; multi-game runs skip that. In sequential mode (--workers 1) the LAST
game is saved to replay.html. Parallel mode (--workers > 1) only tallies the win
rate -- replay.html / obs dump are skipped (use --workers 1 to save them).
"""

import sys
import json
import logging

logging.disable(logging.WARNING)  # silence env-loading INFO/WARNING noise

from kaggle_environments import make


def play_one(agents, keep_env=False):
    """Run a single game; return (result, rewards) for Player 0, where result is
    one of WIN/DRAW/LOSS. When keep_env=True also return the env (sequential only;
    the env is NOT picklable, so workers must call with keep_env=False)."""
    env = make("orbit_wars", debug=True)
    env.run(agents)
    rewards = [s["reward"] if s["reward"] is not None else 0 for s in env.steps[-1]]
    r0, best_other = rewards[0], max(rewards[1:])
    if r0 > best_other:
        res = "WIN"
    elif r0 == best_other:
        res = "DRAW"
    else:
        res = "LOSS"
    return (res, rewards, env) if keep_env else (res, rewards)


def dump_observations(env):
    """Dump the single game's per-step observations (inspection tool)."""
    JSONL_PATH, TXT_PATH = "obs_log.jsonl", "obs_log.txt"
    with open(JSONL_PATH, "w", encoding="utf-8") as jf, \
         open(TXT_PATH, "w", encoding="utf-8") as tf:
        for t, step in enumerate(env.steps):
            obs = step[0].get("observation", {})
            jf.write(json.dumps(obs, default=str) + "\n")

            planets = obs.get("planets") or []
            fleets = obs.get("fleets") or []
            comets = obs.get("comets") or []
            tf.write(f"\n===== step {t} =====\n")
            tf.write(f"keys={sorted(obs.keys())}\n")
            tf.write(f"player={obs.get('player')} step={obs.get('step')} "
                     f"angular_velocity={obs.get('angular_velocity')}\n")
            tf.write(f"#planets={len(planets)} #fleets={len(fleets)} "
                     f"#comet_groups={len(comets)} "
                     f"comet_planet_ids={obs.get('comet_planet_ids')}\n")
            if planets:
                tf.write(f"planet[0]={planets[0]} (len={len(planets[0])})\n")
            if fleets:
                tf.write(f"fleet[0]={fleets[0]} (len={len(fleets[0])})\n")
            for gi, grp in enumerate(comets):
                pi = grp.get("path_index")
                paths = grp.get("paths") or []
                tf.write(f"  comet[{gi}] path_index={pi!r}({type(pi).__name__}) "
                         f"planet_ids={grp.get('planet_ids')} "
                         f"path_len={[len(p) for p in paths]}\n")
                if paths and paths[0]:
                    tf.write(f"    paths[0][:2]={paths[0][:2]}\n")
    print(f"observations logged -> {JSONL_PATH}, {TXT_PATH} ({len(env.steps)} steps)")


def main():
    # ---- parse args: agents (positional) + --games N + --workers N ----
    games = 1
    workers = 1
    agents = []
    argv = iter(sys.argv[1:])
    for a in argv:
        if a == "--games":
            games = int(next(argv))
        elif a.startswith("--games="):
            games = int(a.split("=", 1)[1])
        elif a == "--workers":
            workers = int(next(argv))
        elif a.startswith("--workers="):
            workers = int(a.split("=", 1)[1])
        else:
            agents.append(a)
    agents = agents or ["main.py", "main.py"]   # default: my model on both sides
    workers = max(1, min(workers, games))
    print(f"agents: {agents}  games: {games}  workers: {workers}")

    wins = draws = 0
    last_env = None

    def tally(g, res, rewards):
        nonlocal wins, draws
        if res == "WIN":
            wins += 1
        elif res == "DRAW":
            draws += 1
        print(f"game {g+1}/{games}: rewards={rewards} -> P0 {res}")

    if workers == 1:
        # Sequential: keep the last game's env for replay / obs dump.
        for g in range(games):
            keep = (g == games - 1)
            out = play_one(agents, keep_env=keep)
            if keep:
                res, rewards, last_env = out
            else:
                res, rewards = out
            tally(g, res, rewards)
    else:
        # Parallel: workers return only (res, rewards); env stays in the child.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for g, (res, rewards) in enumerate(ex.map(play_one, [agents] * games)):
                tally(g, res, rewards)

    rate = wins / games
    print(f"\nPlayer 0: {wins}/{games} wins = {rate:.1%}"
          + (f"  ({draws} draws)" if draws else ""))

    if last_env is not None:
        # Single-game run also dumps per-step observations for inspection.
        if games == 1:
            dump_observations(last_env)
        with open("replay.html", "w", encoding="utf-8") as f:
            f.write(last_env.render(mode="html", width=800, height=600))
        print("last game saved to replay.html")
    else:
        print("(parallel mode: replay.html / obs dump skipped; use --workers 1 to save them)")


if __name__ == "__main__":
    main()

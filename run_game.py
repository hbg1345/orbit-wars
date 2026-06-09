"""Run one or more Orbit Wars games and report Player 0's win rate.

Usage:
    python run_game.py                       # main.py vs main.py (1 game)
    python run_game.py main.py random        # vs the built-in random bot
    python run_game.py main.py main.py main.py main.py   # 4-player game
    python run_game.py --games 20 main.py random         # 20 games, P0 win rate

With --games 1 the per-step observations are dumped (obs_log.jsonl / .txt) for
inspection; multi-game runs skip that. The LAST game is always saved to replay.html.
"""

import sys
import json
import logging
logging.disable(logging.WARNING)  # silence env-loading INFO/WARNING noise

from kaggle_environments import make

# ---- parse args: agents (positional) + --games N ----
games = 1
agents = []
argv = iter(sys.argv[1:])
for a in argv:
    if a == "--games":
        games = int(next(argv))
    elif a.startswith("--games="):
        games = int(a.split("=", 1)[1])
    else:
        agents.append(a)
agents = agents or ["main.py", "main.py"]   # default: my model on both sides
print(f"agents: {agents}  games: {games}")



wins = draws = 0
for g in range(games):
    env = make("orbit_wars", debug=True)    
    env.run(agents)
    rewards = [s["reward"] if s["reward"] is not None else 0 for s in env.steps[-1]]
    r0, best_other = rewards[0], max(rewards[1:])
    if r0 > best_other:
        wins += 1; res = "WIN"
    elif r0 == best_other:
        draws += 1; res = "DRAW"
    else:
        res = "LOSS"
    print(f"game {g+1}/{games}: rewards={rewards} -> P0 {res}")

rate = wins / games
print(f"\nPlayer 0: {wins}/{games} wins = {rate:.1%}"
      + (f"  ({draws} draws)" if draws else ""))

# Dump the last game's per-step observations only for a single-game run (so it
# stays an inspection tool without bloating multi-game runs).
if games == 1:
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

# Save the last game as an HTML replay you can open in a browser.
with open("replay.html", "w", encoding="utf-8") as f:
    f.write(env.render(mode="html", width=800, height=600))
print("last game saved to replay.html")

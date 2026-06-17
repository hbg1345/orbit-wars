"""Throwaway profiler: split one rollout's wall time into extract / net / env.step
so we know what actually dominates the (CPU-bound) self-play loop.

Mirrors train_rl_v2.play_game but with per-stage timers. Run in the `orbit` env:
    conda run -n orbit python profile_rollout.py
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time
import numpy as np
import torch

import features as F
from model import OrbitNet
from kaggle_environments import make
from train_rl_v2 import to_action


def profile(n_games=3, n_players=2):
    torch.set_num_threads(1)
    nets = [OrbitNet().eval() for _ in range(n_players)]
    t_extract = t_net = t_action = t_step = 0.0
    turns = 0
    wall0 = time.perf_counter()
    for _ in range(n_games):
        env = make("orbit_wars")
        env.reset(num_agents=n_players)
        eta = [{} for _ in range(n_players)]
        while not env.done:
            actions = []
            for p in range(n_players):
                obs = env.state[p]["observation"]
                a = time.perf_counter()
                feats, valid, mine, ids, xs, ys, ships = F.extract(obs, eta_cache=eta[p])
                b = time.perf_counter()
                decisions, rec = nets[p].act_rl(feats, valid, mine)
                c = time.perf_counter()
                actions.append(to_action(decisions, obs, ids, ships))
                d = time.perf_counter()
                t_extract += b - a
                t_net += c - b
                t_action += d - c
            e = time.perf_counter()
            env.step(actions)
            t_step += time.perf_counter() - e
            turns += 1
    wall = time.perf_counter() - wall0
    tot = t_extract + t_net + t_action + t_step
    print(f"\ngames={n_games}  players={n_players}  turns={turns}  wall={wall:.2f}s")
    print(f"  extract : {t_extract:6.2f}s  ({100*t_extract/tot:4.1f}%)")
    print(f"  net     : {t_net:6.2f}s  ({100*t_net/tot:4.1f}%)")
    print(f"  action  : {t_action:6.2f}s  ({100*t_action/tot:4.1f}%)")
    print(f"  env.step: {t_step:6.2f}s  ({100*t_step/tot:4.1f}%)")
    print(f"  per-turn: {1000*tot/turns:.1f} ms  ({1000*tot/turns/n_players:.1f} ms/seat)")


if __name__ == "__main__":
    profile()

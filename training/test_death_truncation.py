"""Test that play_game truncates a recorded seat's trajectory at its death.

Run in the orbit env:
    conda run -n orbit python test_death_truncation.py

Two things are checked:

1. `_is_eliminated` unit cases (no planet AND no fleet -> dead).

2. End-to-end: for many seeded 4p games, the recorded trajectory length must
   equal the turn the learner was eliminated (or the full turn count if it
   survived). The reference death turn is computed by a FULL-record replay of
   the SAME game. The two runs stay bit-identical because a dead seat owns no
   planet -> act_rl returns before drawing any RNG (model.py:153), so skipping
   its forward pass in play_game never desyncs the random stream.

   With the OLD code (no truncation) len(data[seat]) always == turns, so the
   `len < turns` cases below would fail -> this test is a real regression guard.
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import random
import time

import numpy as np
import torch
from kaggle_environments import make

import features as F
from model import OrbitNet
from train_rl_v2 import play_game, to_action, _is_eliminated, GATE_SAMPLE_BIAS

N_PLAYERS = 4
SEAT = 0
N_GAMES = 1


def reseed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def reference_death_turn(nets, seat):
    """Full-record replay (no truncation): return (turns, death_turn).

    Mirrors play_game's per-seat act_rl calls exactly so the RNG stream matches.
    death_turn = first turn after which `seat` owns no planet and no fleet, else
    None.
    """
    n = len(nets)
    env = make("orbit_wars")
    env.reset(num_agents=n)
    eta = [{} for _ in range(n)]
    canons = [F.compute_canon(env.state[p]["observation"]) for p in range(n)]
    turns = 0
    death = None
    while not env.done:
        actions = []
        for p in range(n):
            obs = env.state[p]["observation"]
            feats, valid, mine, ids, xs, ys, ships = F.extract(
                obs, eta_cache=eta[p], canon=canons[p])
            decisions, _ = nets[p].act_rl(feats, valid, mine, gate_bias=GATE_SAMPLE_BIAS)
            actions.append(to_action(decisions, obs, ids, ships))
        env.step(actions)
        turns += 1
        if death is None and _is_eliminated(env.state[seat]["observation"], seat):
            death = turns
    return turns, death


def test_is_eliminated():
    # obs rows: planet = [id, owner, x, y, angle, ships, prod];
    #           fleet  = [id, owner, x, y, angle, from_planet_id, ships].
    own_planet = {"planets": [[0, SEAT, 1, 1, 0, 5, 1]], "fleets": []}
    own_fleet = {"planets": [[0, 1, 1, 1, 0, 5, 1]],
                 "fleets": [[9, SEAT, 2, 2, 0, 0, 3]]}
    dead = {"planets": [[0, 1, 1, 1, 0, 5, 1]],
            "fleets": [[9, 2, 2, 2, 0, 0, 3]]}
    empty = {"planets": [], "fleets": []}
    assert not _is_eliminated(own_planet, SEAT), "owns a planet -> alive"
    assert not _is_eliminated(own_fleet, SEAT), "fleet in transit -> alive"
    assert _is_eliminated(dead, SEAT), "no planet/fleet -> dead"
    assert _is_eliminated(empty, SEAT), "empty board -> dead"
    print("[ok] _is_eliminated unit cases")


def test_truncation():
    torch.manual_seed(0)
    nets = [OrbitNet().eval() for _ in range(N_PLAYERS)]

    n_died = 0
    n_truncated = 0
    t_trunc = 0.0          # wall time inside play_game (truncated recording)
    t_full = 0.0           # wall time inside the full-record reference
    for s in range(N_GAMES):
        reseed(s)
        t0 = time.perf_counter()
        data, rewards, shaped, turns, _rr, _tf = play_game(
            nets, record_seats={SEAT}, greedy_seats=set(), track_seat=None)
        t_trunc += time.perf_counter() - t0
        rec_len = len(data[SEAT])

        reseed(s)
        t0 = time.perf_counter()
        ref_turns, death = reference_death_turn(nets, SEAT)
        t_full += time.perf_counter() - t0

        # Same seed + same calls -> identical game.
        assert turns == ref_turns, f"seed {s}: turns {turns} != ref {ref_turns}"
        # data / shaped stay aligned.
        assert len(data[SEAT]) == len(shaped[SEAT]), \
            f"seed {s}: data {len(data[SEAT])} != shaped {len(shaped[SEAT])}"
        # The headline invariant.
        expected = death if death is not None else turns
        assert rec_len == expected, \
            f"seed {s}: recorded {rec_len} != expected {expected} (turns={turns}, death={death})"

        if death is not None:
            n_died += 1
            if death < turns:
                n_truncated += 1
                # A truncated trajectory must be strictly shorter than the game.
                assert rec_len < turns

    print(f"[ok] truncation: {N_GAMES} games, learner eliminated in {n_died}, "
          f"of which {n_truncated} died before game end (tail truncated)")
    # The reference replays every seat every turn (old, no-truncation behavior);
    # play_game skips dead seats. The gap is the forward-pass time saved.
    saved = 100.0 * (t_full - t_trunc) / t_full if t_full else 0.0
    print(f"[time] truncated play_game: {t_trunc:.2f}s | full-record ref: "
          f"{t_full:.2f}s | saved {saved:+.1f}%  ({N_GAMES} games)")
    assert n_truncated > 0, (
        "no game truncated the tail -> test is vacuous; raise N_GAMES or check "
        "that 4p eliminations actually occur")


if __name__ == "__main__":
    test_is_eliminated()
    test_truncation()
    print("\nALL PASSED")

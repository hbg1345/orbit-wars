"""Validate features.extract's temporal-block encoding against the raw forecast.

forecast.simulate is already verified bit-exact vs the engine; this checks the
*packing* step in features.extract -- that the temporal channels (owner one-hot
with CCW quadrant re-keying, log-scaled garrison, rotated positions) decode back
to exactly the trajectory forecast produced. Catches channel-offset, scale, and
rotation bugs in the encoding.

For each planet/turn we decode the block and compare:
  - owner one-hot  -> the channel features._owner_channel says it should be
  - ships channel  -> expm1(ch * log1000) == forecast ships
  - x,y channels   -> un-rotate / un-normalise == forecast position
  - GONE turns     -> all temporal channels exactly 0
"""
import math
import logging

import numpy as np

logging.disable(logging.WARNING)

from kaggle_environments import make

import forecast
import features as F


def inv_rot(xn, yn, rot_k):
    """Inverse of features._rot_ccw(., rot_k)."""
    return F._rot_ccw(xn, yn, (4 - rot_k) % 4)


def check_obs(obs, canon, tol=1e-5):
    """Return (n_checks, errors[]) comparing extract's temporal block to forecast."""
    player = obs.get("player", 0)
    quad = canon["quad"]
    rot_k = canon["rot_k"]

    traj = forecast.simulate(obs, F.HORIZON)
    feats, valid, _mine, ids, _xs, _ys, _sh = F.extract(obs, canon=canon)

    errs = []
    n = 0
    half = F.BOARD / 2.0
    for i in range(F.MAX_PLANETS):
        if not valid[i]:
            continue
        pid = int(ids[i])
        seq = traj.get(pid)
        if seq is None:
            continue
        owner_seq, ships_seq, posx_seq, posy_seq = seq
        block = feats[i, F.STATIC_DIM:].reshape(F.HORIZON, F.TEMP_CH)
        for t in range(F.HORIZON):
            n += 1
            o = owner_seq[t]
            owner_oh = block[t, :F.OWNER_CH]
            if o == forecast.GONE:
                # entire temporal row must be zero
                if np.any(block[t] != 0.0):
                    errs.append((pid, t, "GONE-not-zero", block[t].tolist()))
                continue
            # --- owner one-hot ---
            if abs(owner_oh.sum() - 1.0) > tol:
                errs.append((pid, t, "owner-not-onehot", owner_oh.tolist()))
            exp_ch = F._owner_channel(o, player, quad)
            if owner_oh[exp_ch] != 1.0:
                errs.append((pid, t, f"owner-ch exp{exp_ch} owner{o}", owner_oh.tolist()))
            # --- ships (log-scaled) ---
            rec_ships = math.expm1(block[t, F.OWNER_CH] * F._LOG1000)
            if abs(rec_ships - max(ships_seq[t], 0.0)) > 1e-2:
                errs.append((pid, t, "ships", (rec_ships, ships_seq[t])))
            # --- position (rotated, normalised) ---
            xn, yn = block[t, F.OWNER_CH + 1], block[t, F.OWNER_CH + 2]
            ux, uy = inv_rot(float(xn), float(yn), rot_k)
            rec_x = ux * half + F.CENTER[0]
            rec_y = uy * half + F.CENTER[1]
            if abs(rec_x - posx_seq[t]) > 1e-3 or abs(rec_y - posy_seq[t]) > 1e-3:
                errs.append((pid, t, "pos", (rec_x, rec_y, posx_seq[t], posy_seq[t])))
    return n, errs


def make_agent(stop):
    def agent(obs):
        s = obs.get("step", 0)
        if s >= stop:
            return []
        player = obs.get("player", 0)
        pl = obs.get("planets", []) or []
        mine = [p for p in pl if p[1] == player]
        tgt = [p for p in pl if p[1] != player]
        mv = []
        for mp in mine:
            if mp[5] < 4 or not tgt:
                continue
            t = min(tgt, key=lambda q: math.hypot(mp[2] - q[2], mp[3] - q[3]))
            mv.append([mp[0], math.atan2(t[3] - mp[3], t[2] - mp[2]), int(mp[5] // 2)])
        return mv
    return agent


def main():
    total_checks = 0
    total_errs = []
    for seed in range(300, 312):
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 120}, debug=False)
        ag = make_agent(40)  # launch for a while, then idle -> varied owners in flight
        env.run([ag, ag])
        steps = env.steps
        canon = F.compute_canon(steps[0][0]["observation"])
        # sample several mid-game steps (where forecast is exact and board is rich)
        for t in (10, 20, 35, 45, 60, 80):
            if t >= len(steps):
                continue
            obs = steps[t][0]["observation"]
            if not obs.get("planets"):
                continue
            n, errs = check_obs(obs, canon)
            total_checks += n
            total_errs.extend(errs)

    print(f"\nfeatures.extract temporal encoding: {total_checks} (planet,turn) cells checked")
    if not total_errs:
        print("  ALL MATCH -- owner channels, ship scale, and position rotation "
              "decode exactly back to forecast.")
    else:
        print(f"  {len(total_errs)} mismatches:")
        for e in total_errs[:20]:
            print(f"    {e}")


if __name__ == "__main__":
    main()

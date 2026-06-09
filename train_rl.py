"""A2C RL for OrbitNet, initialised from the best RL checkpoint (or the SL policy
on a fresh start) and trained against a snapshot-pool league of its recent bests.

Each iteration plays `--games` games at a 3:1 ratio of 2-player to 4-player. The
opponent set is 2 recent-best snapshots + the fixed SL teacher (orbit_net.pt):
  * 2p games: the learner vs one of those 3 references (cycled).
  * 4p games: the learner + all 3 references, one per seat.
Until 2 bests exist the missing slots fall back to the SL teacher. Each turn the
gate (launch?) and target are sampled; the ship fraction stays the deterministic
SL head. The win/loss reward at game end is discounted back to every step, the
value head is the baseline (advantage = return - value), and we do an actor-critic
update on the learner's trajectory only. Whenever the learner's win rate vs the
league clears --win-gate (default 0.5) it is saved (orbit_net_rl.pt) and pushed
into the pool -- which strengthens the league, so the bar auto-rises (a ratchet)
without the freeze of a max-win-rate criterion. The SL weights (orbit_net.pt) are
read for init but never overwritten; latest -> orbit_net_rl_last.pt.

Not yet: teacher-KL regulariser. Usage:
    python train_rl.py --iters 200 --games 8 --workers 8
"""
import argparse
import csv
import logging
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

import features as F
from model import OrbitNet

logging.disable(logging.WARNING)            # quiet kaggle_environments banners
from kaggle_environments import make
from main import _aim, _build_world, _fleet_speed, GATE_THRESHOLD   # reuse launch geometry + greedy gate

GAMMA = 0.997
LR = 1e-5
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MINIBATCH = 256
POOL_SIZE = 2                      # managed recent-best snapshots (rest of the 4p
                                  # opponents are the fixed SL teacher)
N_OPP = 3                          # opponents in a 4p game (learner + 3 = 4 seats);
                                  # = POOL_SIZE bests + (N_OPP - POOL_SIZE) fixed SL
SL_WEIGHTS = "orbit_net.pt"
RL_BEST = "orbit_net_rl.pt"        # best-by-league-winrate (this is what main_rl.py loads)
RL_LAST = "orbit_net_rl_last.pt"   # latest weights (for resume / debugging)
RL_POOL_DIR = "rl_pool"            # last-POOL_SIZE best snapshots persisted here (snap0=newest)
RL_LOG_CSV = "train_rl_log.csv"    # per-iter metrics, plotted by plot_rl_log.py


def _pool_path(i):
    return os.path.join(RL_POOL_DIR, f"snap{i}.pt")


def load_pool():
    """Restore the snapshot pool (most-recent first) from disk, skipping any file
    whose architecture no longer matches the current OrbitNet."""
    dq = deque(maxlen=POOL_SIZE)
    for i in range(POOL_SIZE):
        p = _pool_path(i)
        if not os.path.exists(p):
            break
        sd = torch.load(p, map_location="cpu")
        try:
            OrbitNet().load_state_dict(sd)          # strict: reject a stale architecture
        except Exception as e:
            print(f"  pool snapshot {p} incompatible, skipping ({e})")
            continue
        dq.append(sd)
    return dq


def save_pool(best_pool):
    """Mirror the in-memory pool to disk (snap0=newest), keeping at most POOL_SIZE."""
    os.makedirs(RL_POOL_DIR, exist_ok=True)
    for i, sd in enumerate(best_pool):
        torch.save(sd, _pool_path(i))
    for i in range(len(best_pool), POOL_SIZE):       # drop stale files
        if os.path.exists(_pool_path(i)):
            os.remove(_pool_path(i))


def to_action(decisions, obs, ids, ships):
    """Turn sampled (src, tgt, frac) decisions into the engine's move list
    [[from_id, angle, num_ships], ...] -- same geometry as main._neural_moves."""
    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in raw}
    world = _build_world(obs)
    moves = []
    for src, tgt, frac in decisions:
        available = float(ships[src])
        n = min(int(round(frac * available)), int(available))
        if n < 1:
            continue
        tpl, spl = by_id.get(int(ids[tgt])), by_id.get(int(ids[src]))
        if tpl is None or spl is None:
            continue
        angle = _aim(spl, tpl, world, _fleet_speed(n))
        if angle is None:
            continue
        moves.append([int(ids[src]), angle, n])
    return moves


def _ship_share(obs, me):
    """Fraction of all ships (garrisons + in-flight fleets) owned by `me`, in [0,1].
    Ship count is the win metric, so its per-turn change is a dense reward signal."""
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    mine = (sum(p[5] for p in planets if p[1] == me)
            + sum(f[6] for f in fleets if f[1] == me))
    total = sum(p[5] for p in planets) + sum(f[6] for f in fleets)
    return (mine / total) if total > 0 else 0.0


def play_game(nets, env, record_seats=None, greedy_seats=None):
    """Run one game with one policy per seat on a reusable `env` (reset here, not
    re-made -- make() per game is wasteful). `record_seats` (set) chooses which
    seats' step data to keep (the learner, sampled). `greedy_seats` play the
    deterministic main.py policy (act + gate threshold) -- used for the SL teacher
    so the in-training opponent matches how it is actually deployed. Other seats
    sample (act_rl), matching main_rl's deployment. Returns (data, rewards, shaped);
    shaped[p] is the per-step ship-share delta (dense reward) for kept seats."""
    n = len(nets)
    keep = set(range(n)) if record_seats is None else set(record_seats)
    greedy_seats = greedy_seats or set()
    env.reset(num_agents=n)
    data = [[] for _ in range(n)]
    shaped = [[] for _ in range(n)]
    # A fleet's target/arrival is fixed geometry, so cache ETAs across the game
    # (per seat) -- same trick as replay.parse_episode; avoids re-tracing every
    # in-flight fleet from scratch each step (the dominant per-step cost).
    eta_caches = [{} for _ in range(n)]
    while not env.done:
        actions = []
        before = {}
        for p in range(n):
            obs = env.state[p]["observation"]
            feats, valid, mine, ids, xs, ys, ships = F.extract(obs, eta_cache=eta_caches[p])
            if p in keep:                                  # learner: sample + record
                decisions, rec = nets[p].act_rl(feats, valid, mine)
                data[p].append((feats, valid, mine, rec))
                before[p] = _ship_share(obs, p)            # share before acting
            elif p in greedy_seats:                        # SL teacher: greedy (= main.py)
                decisions = nets[p].act(feats, valid, mine, greedy=True,
                                        gate_threshold=GATE_THRESHOLD)
            else:                                          # RL snapshot: sample (= main_rl)
                decisions, _ = nets[p].act_rl(feats, valid, mine)
            actions.append(to_action(decisions, obs, ids, ships))
        env.step(actions)
        for p in keep:                                     # dense reward = share delta
            shaped[p].append(_ship_share(env.state[p]["observation"], p) - before[p])
    rewards = [float(env.state[p]["reward"] or 0.0) for p in range(n)]
    return data, rewards, shaped


# --------------------------------------------------------------------------- #
# Parallel rollout workers (actor processes). Each builds a learner net + a few
# reusable opponent nets once, then per task loads the learner + opponent weights
# and plays one game. Opponents come from the snapshot pool (see main) and are
# passed in the task payload -- the pickling cost is negligible next to a rollout.
# Seats = len(opp_sds)+1, so 2p and 4p share this path.
# --------------------------------------------------------------------------- #
_W_LEARNER = None    # learner (weights reloaded per task)
_W_OPP = []          # reusable opponent nets (loaded per task)
_W_ENV = None        # reused across games (avoid make() per game)


def _worker_init():
    global _W_LEARNER, _W_ENV
    logging.disable(logging.WARNING)
    _W_LEARNER = OrbitNet()
    _W_LEARNER.eval()
    _W_ENV = make("orbit_wars")


def _rollout_one(payload):
    """Play one game: learner at `seat`, pool opponents in the other seats. SL-teacher
    opponents (opp_is_sl) play greedily; RL snapshots sample. Returns the learner
    seat's (data, reward, shaped, won, players) -- on-policy for the learner."""
    learner_sd, opp_sds, opp_is_sl, seat = payload
    _W_LEARNER.load_state_dict(learner_sd)
    while len(_W_OPP) < len(opp_sds):
        n = OrbitNet(); n.eval(); _W_OPP.append(n)
    players = len(opp_sds) + 1
    nets = [None] * players
    nets[seat] = _W_LEARNER
    greedy_seats = set()
    oi = 0
    for p in range(players):
        if p == seat:
            continue
        _W_OPP[oi].load_state_dict(opp_sds[oi])
        nets[p] = _W_OPP[oi]
        if opp_is_sl[oi]:
            greedy_seats.add(p)
        oi += 1
    data, rewards, shaped = play_game(nets, _W_ENV, record_seats={seat},
                                      greedy_seats=greedy_seats)
    won = rewards[seat] > max(rewards[i] for i in range(players) if i != seat)
    return data[seat], rewards[seat], shaped[seat], won, players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=8,
                    help="total games per iteration (split 3:1 between 2p and 4p)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="parallel rollout processes (1 = serial)")
    ap.add_argument("--win-gate", type=float, default=0.44,
                    help="save RL_BEST + push to pool when league win rate exceeds this")
    ap.add_argument("--shape-coef", type=float, default=1.0,
                    help="weight on the per-turn ship-share dense reward (0 = terminal only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = OrbitNet().to(device)
    teacher = OrbitNet().to(device)            # SL anchor (pads the pool until 3 bests)
    if os.path.exists(SL_WEIGHTS):
        teacher.load_state_dict(torch.load(SL_WEIGHTS, map_location=device), strict=False)
    else:
        print(f"warning: no {SL_WEIGHTS}; teacher (pool padding) is random")
    # Learner resumes from the best RL checkpoint if present, else the SL policy.
    init_src = RL_BEST if os.path.exists(RL_BEST) else SL_WEIGHTS
    if os.path.exists(init_src):
        net.load_state_dict(torch.load(init_src, map_location=device), strict=False)
        print(f"learner initialised from {init_src}  |  teacher from {SL_WEIGHTS}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    teacher_sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    # Snapshot pool: the last POOL_SIZE best learners (most-recent first). Together
    # with the fixed SL teacher they form the N_OPP opponents (4p uses all, 2p
    # cycles). Padded with the SL teacher until enough bests exist. Restored from
    # disk so a run resumes its
    # league (compatible snapshots only).
    best_pool = load_pool()
    if best_pool:
        print(f"restored {len(best_pool)} pool snapshot(s) from {RL_POOL_DIR}/")

    workers = max(1, min(args.workers, args.games))
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
            if workers > 1 else None)
    if pool is None:
        _worker_init()                         # set up the in-process rollout globals
    print(f"rollout workers: {workers}  |  games/iter: {args.games} (3:1 2p:4p)  "
          f"| pool size: {POOL_SIZE}")

    def opp_refs():
        """The N_OPP reference opponents: the recent bests first, the remaining
        slots fixed to the SL teacher (so 4p = 2 moving bests + 1 anchored SL)."""
        return (list(best_pool) + [teacher_sd] * N_OPP)[:N_OPP]

    def make_tasks(learner_sd):
        """This iter's games at 3:1 2p:4p. 2p vs one reference (cycled); 4p vs all.
        Returns (tasks, tags); tag marks 2p-vs-SL games (a fixed-opponent yardstick)
        -- an opponent is the SL teacher iff it's the teacher_sd object itself."""
        refs = opp_refs()
        refs_is_sl = [r is teacher_sd for r in refs]         # SL teacher slots -> greedy
        n4 = args.games // 4
        n2 = args.games - n4
        tasks, tags = [], []
        for j in range(n2):                                  # 2p
            k = j % N_OPP
            tasks.append((learner_sd, [refs[k]], [refs_is_sl[k]], j % 2))
            tags.append("sl2p" if refs_is_sl[k] else "best2p")
        for j in range(n4):                                  # 4p
            tasks.append((learner_sd, list(refs), list(refs_is_sl), j % 4))
            tags.append("4p")
        return tasks, tags

    def rollouts():
        """Yield (tag, (data, reward, won, players)) for this iter's games."""
        learner_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        tasks, tags = make_tasks(learner_sd)
        results = pool.map(_rollout_one, tasks) if pool is not None \
            else (_rollout_one(tk) for tk in tasks)
        yield from zip(tags, results)

    log_f = open(RL_LOG_CSV, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "steps", "win", "win2", "win4", "vssl", "wsl", "gsl",
                  "policy", "value", "entropy"])

    pbar = tqdm(range(args.iters), desc="RL", unit="iter")
    for it in pbar:
        # ---- collect league rollouts ----
        net.eval()
        t_roll = time.perf_counter()
        feats_buf, valid_buf, mine_buf, rec_buf, ret_buf = [], [], [], [], []
        w2 = g2 = w4 = g4 = wsl = gsl = 0
        for tag, (seat_data, reward, shaped, won, players) in tqdm(rollouts(), total=args.games,
                                                                   desc="rollout", unit="game", leave=False):
            if players == 2:
                g2 += 1; w2 += won
            else:
                g4 += 1; w4 += won
            if tag == "sl2p":                        # fixed-opponent yardstick (2p vs SL)
                gsl += 1; wsl += won
            # Discounted returns: per-turn dense reward (ship-share delta) plus the
            # terminal win/loss at the last step.
            T = len(seat_data)
            rets = [0.0] * T
            G = 0.0
            for t in reversed(range(T)):
                r = args.shape_coef * shaped[t] + (reward if t == T - 1 else 0.0)
                G = r + GAMMA * G
                rets[t] = G
            for t, (feats, valid, mine, rec) in enumerate(seat_data):
                feats_buf.append(feats)
                valid_buf.append(valid)
                mine_buf.append(mine)
                rec_buf.append(rec)
                ret_buf.append(rets[t])
        win2 = w2 / g2 if g2 else 0.0
        win4 = w4 / g4 if g4 else 0.0
        win_rate = (w2 + w4) / (g2 + g4)

        roll_s = time.perf_counter() - t_roll
        # ---- actor-critic update (one pass, minibatched) ----
        t_upd = time.perf_counter()
        net.train()
        order = list(range(len(ret_buf)))
        random.shuffle(order)
        tot_p = tot_v = tot_e = 0.0
        nb = 0
        for s in range(0, len(order), MINIBATCH):
            chunk = order[s:s + MINIBATCH]
            fb = torch.from_numpy(np.stack([feats_buf[j] for j in chunk])).to(device)
            vb = torch.from_numpy(np.stack([valid_buf[j] for j in chunk])).to(device)
            mb = torch.from_numpy(np.stack([mine_buf[j] for j in chunk])).to(device)
            recs = [rec_buf[j] for j in chunk]
            ret = torch.tensor([ret_buf[j] for j in chunk], dtype=torch.float32, device=device)

            logp, value, ent = net.evaluate_actions(fb, vb, mb, recs)
            adv = ret - value.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalize -> stable gradients
            policy_loss = -(logp * adv).mean()
            value_loss = ((ret - value) ** 2).mean()
            entropy = ent.mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_p += policy_loss.item(); tot_v += value_loss.item(); tot_e += entropy.item(); nb += 1

        upd_s = time.perf_counter() - t_upd
        p_loss, v_loss, e_loss = tot_p / nb, tot_v / nb, tot_e / nb
        # win_rate vs the league this iter is the progress signal (2p rand baseline
        # ~0.5, 4p ~0.25). On a new best, save it and push it into the pool.
        sl_str = f"{wsl}/{gsl}" if gsl else "-"
        line = (f"iter {it+1:4d}  steps={len(ret_buf)}  rollout={roll_s:.0f}s update={upd_s:.0f}s  "
                f"win={win_rate:.2f} (2p={win2:.2f} 4p={win4:.2f} vsSL={sl_str})  "
                f"policy={p_loss:.4f} value={v_loss:.4f} entropy={e_loss:.3f}")
        if win_rate > args.win_gate:
            torch.save(net.state_dict(), RL_BEST)
            best_pool.appendleft({k: v.detach().cpu() for k, v in net.state_dict().items()})
            save_pool(best_pool)                     # persist the (sliding) league pool
            line += f"  <- win>{args.win_gate:.2f}, saved {RL_BEST} (+pool {len(best_pool)})"
        pbar.set_postfix(win=f"{win_rate:.2f}", vsSL=sl_str, pool=len(best_pool),
                         policy=f"{p_loss:.3f}", value=f"{v_loss:.3f}", entropy=f"{e_loss:.2f}")
        tqdm.write(line)
        log.writerow([it + 1, len(ret_buf), f"{win_rate:.4f}", f"{win2:.4f}", f"{win4:.4f}",
                      (f"{wsl/gsl:.4f}" if gsl else ""), wsl, gsl,
                      f"{p_loss:.6f}", f"{v_loss:.6f}", f"{e_loss:.6f}"])
        log_f.flush()                           # so plot_rl_log.py can read live
        torch.save(net.state_dict(), RL_LAST)   # latest, every iter

    log_f.close()
    if pool is not None:
        pool.shutdown()
    print(f"done. latest -> {RL_LAST}, last gated best -> {RL_BEST}  (log -> {RL_LOG_CSV})")


if __name__ == "__main__":
    main()
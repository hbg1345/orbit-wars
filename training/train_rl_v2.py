"""A2C/PPO RL for OrbitNet, initialised from the best RL checkpoint (or the SL policy
on a fresh start) and trained against a snapshot-pool league of its recent bests.

Features GAE, Teacher KL Regularization, Mixed Opponent Execution, and single-pass Evaluation.
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import atexit
import csv
import json
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

import features as F
from model import OrbitNet

logging.disable(logging.WARNING)
from kaggle_environments import make
from main import GATE_THRESHOLD, K_MAX_GLOBAL
from physics import decisions_to_moves

# PPO & RL Hyperparameters
GAMMA = 1
GAE_LAMBDA = 0.97
LR = 1e-5
VALUE_COEF = 0.5

# Faster wins are worth more: on a win, add a speed bonus on top of the base +1,
# linear in how many turns were saved vs WIN_SPEED_TREF (=0 at/after that ref).
# terminal = base_reward + WIN_SPEED_COEF * max(0, (TREF - turns) / TREF).
# Only applies to wins; losses keep their env reward.
WIN_SPEED_COEF = 0.5
WIN_SPEED_TREF = 500

# Dense lead-margin (PBRS) shaping weight. SHAPE_COEF=0 -> off (pure terminal
# reward: win/loss + rank/speed). Set > 0 to add the dense lead-margin signal.
# Held CONSTANT over the run by default; set SHAPE_DECAY_FRAC > 0 to instead decay
# it linearly to 0 over that fraction of the run (e.g., 1.0 = whole run).
SHAPE_COEF = 0.5
SHAPE_DECAY_FRAC = 0
# The shaping potential blends two normalized lead terms: ships on the board and
# owned-planet production (territory). PROD_WEIGHT scales the production term
# relative to the ship term. Ships catch attrition (wasting a fleet drops your
# ship lead); production catches captures (a tie that fails to take a planet
# forfeits the production jump). Set PROD_WEIGHT=0 for the old ship-only potential.
PROD_WEIGHT = 1.0

# Penalty subtracted from the shaped reward when one of OUR fleets arrives and
# leaves its TARGET NEUTRAL planet at exactly 0 ships without capturing it -- the
# "one short" waste (attacker == garrison -> planet holds at 0; one more ship
# takes it). Only neutral targets count (enemy planets are excluded). The penalty
# is scaled by the target's PRODUCTION (ships/turn, 1..5): a high-output planet is
# worth more, so failing to take it costs more. Unlike the lead-margin PBRS term
# this is intentionally NON-potential: it changes the optimal policy on purpose,
# to discourage under-committed attacks on neutrals. Set 0 to off.
ZERO_PENALTY_COEF = 0.05

# Small entropy bonus to keep the (stochastic) rollout policy from collapsing.
# Only meaningful because rollouts SAMPLE; eval/deploy stay greedy. With the
# discrete 101-bin Categorical ship head there's no continuous sigma to run away,
# so a small coef is safe -- start low, raise only if entropy doesn't inflate / win hold.
ENTROPY_COEF = 1e-3
# Teacher-KL (behavior-cloning anchor) decays from START to FLOOR over the run,
# like shape_coef: anchor hard to the SL policy early, then loosen so the late
# policy can deviate toward better-than-SL play. Floor (not 0) keeps a light
# tether so it can't drift off into nonsense. Set frac to 0 for a constant START.
KL_COEF_START = 1e-2
KL_COEF_FLOOR = 0
KL_DECAY_FRAC = 1.0
MINIBATCH = 512

PPO_CLIP = 0.1
PPO_EPOCHS = 4

# SL trains the gate with pos_weight (class imbalance), which INFLATES its raw
# launch probability. Greedy (threshold) tolerates that, but Bernoulli sampling
# over-fires. De-bias the gate logit by -log(pos_weight) at sample time so RL
# fires at the true ~16% rate. MUST match train_sl.GATE_POS_WEIGHT.
GATE_POS_WEIGHT = 5.4
GATE_SAMPLE_BIAS = -math.log(GATE_POS_WEIGHT)

# Per-iter debug print of reward/value/return/advantage scales (sanity-check the
# unbounded value head + env reward scale). Set False to silence.
DEBUG = False

# Evaluation is itself a full set of rollouts (EVAL_GAMES_2P + EVAL_GAMES_4P per
# iter). Running it every iter roughly doubles rollout cost, so gate it.
EVAL_EVERY = 15

# Fixed eval yardstick sizes (independent of the training --games split):
#   2p -> vs the SL teacher (the headline performance metric, so larger sample);
#   4p -> vs the 3 strongest pool members (diagnostic, smaller sample).
EVAL_GAMES_2P = 100
EVAL_GAMES_4P = 20

# [3] 카타스트로픽 망각을 막기 위한 League Pool 확장
POOL_SIZE = 8
N_OPP = 3

# Opponents are sampled UNIFORMLY from the league (all pool snapshots + SL teacher,
# equal chance). PFSP was dropped: in 4-player games outcomes can't be attributed to
# one opponent, so the per-snapshot win stats it needs were only ever fed by 2p
# games -- the 4p prioritisation rode on stats it never collected. Uniform is honest
# and gives broader, less self-similar exposure (no fixed "latest snapshot" anchor).

SL_WEIGHTS = "orbit_net.pt"
RL_BEST = "orbit_net_rl.pt"        
RL_LAST = "orbit_net_rl_last.pt"   
RL_POOL_DIR = "rl_pool"            
RL_LOG_CSV = "reports/train_rl_log.csv"
GREEDY_RATIO = 0.0

# Live launch-distribution monitor: each iter we append a histogram of the SAMPLED
# ship fractions to this JSONL file, which watch_launch_dist.py tails/animates.
LAUNCH_DIST_FILE = "launch_dist.jsonl"          # training rollouts (SAMPLED fractions)
LAUNCH_DIST_EVAL_FILE = "launch_dist_eval.jsonl"  # eval games (GREEDY/mode fractions)
LAUNCH_DIST_BINS = 50


def _write_launch_dist(it, fracs, path=LAUNCH_DIST_FILE, bins=LAUNCH_DIST_BINS):
    """Append one iter's ship-fraction histogram as a JSON line for the live
    monitor. `fracs` is this iter's sampled launch fractions in [0,1] (one per
    gate-fired planet). Compact binned counts -> the viewer needs no raw samples."""
    fracs = np.asarray(fracs, dtype=np.float32)
    counts, _ = np.histogram(fracs, bins=bins, range=(0.0, 1.0))
    rec = {"iter": int(it), "n": int(fracs.size), "bins": int(bins),
           "mean": float(fracs.mean()) if fracs.size else 0.0,
           "counts": counts.tolist()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

def _load_state(path, map_location="cpu"):
    """Return a raw parameter state_dict from a checkpoint. train_sl.py saves the
    SL net as {'model_state_dict', 'optimizer_state_dict'}, so its weights must be
    unwrapped -- passing that whole dict to load_state_dict(..., strict=False)
    silently matches ZERO keys and leaves the net at random init (the bug that made
    the SL teacher/init random). RL checkpoints are raw state_dicts and pass through."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def save_resume(path, net, opt, it, next_snap_id):
    """Persist a FULL resume bundle (not just weights): model + optimizer state so
    Adam momentum survives a restart, plus the iteration counter (the LR/KL/shape
    schedules are functions of `it`, so resuming at the right step matters) and
    next_snap_id. RNG states are saved too so the rollout stream is continuous.
    Uses the 'model_state_dict' key so _load_state() still unwraps weights from it."""
    rng = {"torch": torch.get_rng_state(),
           "numpy": np.random.get_state(),
           "python": random.getstate(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "iter": int(it),
                "next_snap_id": int(next_snap_id),
                "rng": rng}, path)


def _restore_rng(rng):
    """Best-effort RNG restore; a device/version mismatch must not abort a resume."""
    try:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:
        print(f"  RNG restore skipped ({e})")


def _pool_path(i):
    return os.path.join(RL_POOL_DIR, f"snap{i}.pt")


# A pool entry is {"id": int, "sd": state_dict, "vssl": float}. `vssl` is the
# vs-SL 2p eval win rate the snapshot passed gating with -- its "strength", used to
# pick the strongest opponents for 4p eval. The sd persists as snap<i>.pt; the
# vssl scores persist alongside in scores.json (id is reassigned by slot on load).
def _scores_path():
    return os.path.join(RL_POOL_DIR, "scores.json")


def load_pool():
    dq = deque(maxlen=POOL_SIZE)
    scores = {}
    if os.path.exists(_scores_path()):
        try:
            with open(_scores_path(), encoding="utf-8") as f:
                scores = json.load(f)
        except Exception:
            scores = {}
    for i in range(POOL_SIZE):
        p = _pool_path(i)
        if not os.path.exists(p):
            break
        sd = torch.load(p, map_location="cpu")
        try:
            OrbitNet().load_state_dict(sd)
        except Exception as e:
            print(f"  pool snapshot {p} incompatible, skipping ({e})")
            continue
        dq.append({"id": i, "sd": sd, "vssl": float(scores.get(str(i), 0.5))})
    return dq


def save_pool(best_pool):
    os.makedirs(RL_POOL_DIR, exist_ok=True)
    for i, e in enumerate(best_pool):
        torch.save(e["sd"], _pool_path(i))
    with open(_scores_path(), "w", encoding="utf-8") as f:
        json.dump({str(i): e.get("vssl", 0.5) for i, e in enumerate(best_pool)}, f)
    for i in range(len(best_pool), POOL_SIZE):
        if os.path.exists(_pool_path(i)):
            os.remove(_pool_path(i))


def parity_winrate(g2, g4):
    """Expected sole-winner rate for an equally-skilled player given the game
    mix: 1/2 per 2-player game, 1/4 per 4-player game. Used as the snapshot-save
    gate so 'better than parity' adapts to the 2p:4p ratio instead of a hardcode."""
    n = g2 + g4
    return (g2 * 0.5 + g4 * 0.25) / n if n else 0.0


def to_action(decisions, obs, ids, ships):
    # Thin wrapper over the shared aiming/launch geometry (physics), kept for the
    # (decisions, obs, ...) arg order used by play_game and profile_rollout.
    return decisions_to_moves(obs, decisions, ids, ships, kmax=K_MAX_GLOBAL)


def _owner_margin(by_owner, me):
    # Lead over the strongest rival as a fraction of the board total:
    # (mine - best opponent) / total. Per-step deltas of this reward EXTENDING a
    # lead / closing a gap, which matches the real objective (be #1) far better
    # than raw share -- especially in 4p, where maximising global share can keep
    # rewarding an already-winning player. In 2p it reduces to 2*share - 1, so 2p
    # behaviour is preserved (just scaled). owner -1 = neutral (in total, not mine).
    total = sum(by_owner.values())
    if total <= 0:
        return 0.0
    mine = by_owner.get(me, 0.0)
    others = [v for o, v in by_owner.items() if o >= 0 and o != me]
    best_other = max(others) if others else 0.0
    return (mine - best_other) / total


def _lead_margin(obs, me):
    # Shaping potential = ship-lead + PROD_WEIGHT * production-lead, each a
    # separately-normalized owner margin (see _owner_margin). Ships (planets +
    # in-flight fleets) catch attrition; production (planets only -- fleets carry
    # no production) catches territory: capturing a planet jumps the production
    # lead, so a tie that fails to capture forfeits that jump.
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    ships, prod = {}, {}
    for p in planets:
        o = p[1]
        ships[o] = ships.get(o, 0.0) + p[5]
        prod[o] = prod.get(o, 0.0) + p[6]
    for f in fleets:
        ships[f[1]] = ships.get(f[1], 0.0) + f[6]
    return _owner_margin(ships, me) + PROD_WEIGHT * _owner_margin(prod, me)


def _is_eliminated(obs, me):
    # Env elimination criterion (see orbit_wars: alive_players = owners of any
    # planet or fleet). No planets means no launch source and no fleets means
    # nothing in transit to recapture one, so the state is PERMANENT -- the seat
    # can never act again. We treat that turn as terminal for the learner.
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    return not (any(p[1] == me for p in planets) or any(f[1] == me for f in fleets))


def _zeroing_penalty(tracked, launches, obs_before, obs_after, me):
    """Track each of OUR in-flight fleets -> its target planet, and charge the
    target's production whenever a tracked fleet vanishes this turn while its
    TARGET is a NEUTRAL planet left at 0 ships (an under-committed attack that
    zeroed a neutral but failed to take it). `tracked` is the per-seat live
    {fleet_id: (target_id, target_prod)} map, mutated in place. `launches` is this
    turn's {source_planet_id: (target_id, target_prod)} for the fleets we just
    ordered. Returns the summed wasted-target production (0 if none). Only neutral
    targets (owner -1) count -- captures, reinforcements, and enemy planets do not."""
    fb = {int(f[0]) for f in (obs_before.get("fleets") or [])}
    fa = {int(f[0]): f for f in (obs_after.get("fleets") or [])}
    pa = {int(p[0]): p for p in (obs_after.get("planets") or [])}
    # 1) tracked fleets that disappeared this turn -> charge if target zeroed, untaken
    penalty = 0.0
    for fid in list(tracked):
        if fid in fa:
            continue                              # still flying
        tid, prod = tracked.pop(fid)
        t = pa.get(tid)
        if t is not None and int(t[1]) == -1 and float(t[5]) == 0.0:
            penalty += prod                       # neutral planet only (owner -1)
    # 2) register the fleets we launched this turn (new fleets matched by source)
    for fid, f in fa.items():
        if fid in fb or fid in tracked or int(f[1]) != me:
            continue
        tgt = launches.get(int(f[5]))             # f[5] = from_planet_id
        if tgt is not None:
            tracked[fid] = tgt
    return penalty


def _rank_rewards(rewards, n):
    """Winner-take-all reward straight from the env's terminal rewards: the
    player(s) with the highest env reward win and get +1; everyone else gets -1,
    in BOTH 2p and 4p (so a 4p win is worth the same +1 as a 2p win, not +3 --
    the larger 4p advantage magnitude would otherwise dominate the update). The
    env already decides the winner, so we don't reconstruct rank from survival.
    An all-equal end (everyone survives tied) is a draw -> 0 for all."""
    if n <= 1:
        return [0.0] * n
    top = max(rewards)
    winners = [p for p in range(n) if rewards[p] == top]
    if len(winners) == n:                              # everyone tied -> draw
        return [0.0] * n
    return [1.0 if rewards[p] == top else -1.0 for p in range(n)]


def play_game(nets, record_seats=None, greedy_seats=None, track_seat=None):
    n = len(nets)
    keep = set(range(n)) if record_seats is None else set(record_seats)
    greedy_seats = greedy_seats or set()
    env = make('orbit_wars')
    env.reset(num_agents=n)
    data = [[] for _ in range(n)]
    shaped = [[] for _ in range(n)]
    track_fracs = []                  # launch fractions of track_seat (any policy)
    eta_caches = [{} for _ in range(n)]
    # Per-game canonicalisation per seat (rotate start quadrant -> Q1, CCW enemy
    # keys), computed ONCE from the initial board and reused every turn.
    canons = [F.compute_canon(env.state[p]["observation"]) for p in range(n)]
    fleet_target = [{} for _ in range(n)]     # per seat: live fleet_id -> (target_id, target_prod)
    # Recorded seats permanently eliminated (no planet, no fleet). Their death
    # turn is the LAST step we record -- afterwards they take no action and can
    # never revive, so recording the empty no-op tail would only pollute value
    # learning. Death is treated as terminal for that seat's trajectory.
    dead = set()
    turns = 0
    while not env.done:
        actions = []
        before = {}
        obs_seen = {}                 # raw obs each keep-seat acted on this turn
        launch_tgt = {}               # keep-seat -> {source_planet_id: (target_id, target_prod)}
        acted = []                    # recorded seats that logged a step this turn
        for p in range(n):
            obs = env.state[p]["observation"]
            feats, valid, mine, ids, xs, ys, ships = F.extract(obs, eta_cache=eta_caches[p], canon=canons[p])
            if p in keep and p not in dead:
                decisions, rec = nets[p].act_rl(feats, valid, mine, gate_bias=GATE_SAMPLE_BIAS)
                data[p].append((feats, valid, mine, rec))
                before[p] = _lead_margin(obs, p)
                obs_seen[p] = obs
                prod_of = {int(pl[0]): float(pl[6]) for pl in (obs.get("planets") or [])}
                launch_tgt[p] = {int(ids[s]): (int(ids[tg]), prod_of.get(int(ids[tg]), 1.0))
                                 for (s, tg, _fr) in decisions}
                acted.append(p)
            elif p in dead:
                decisions = []        # eliminated -> no-op, skip the wasted forward pass
            elif p in greedy_seats:
                decisions = nets[p].act(feats, valid, mine, greedy=True,
                                        gate_threshold=GATE_THRESHOLD)
            else:
                decisions, _ = nets[p].act_rl(feats, valid, mine, gate_bias=GATE_SAMPLE_BIAS)
            if p == track_seat:
                track_fracs.extend(float(d[2]) for d in decisions)   # (src, tgt, frac)
            actions.append(to_action(decisions, obs, ids, ships))
        env.step(actions)
        turns += 1
        terminal = env.done
        for p in acted:
            # Proper potential-based reward shaping (PBRS, Ng et al. 1999):
            #   F = gamma * Phi(s') - Phi(s),   with Phi(terminal) = 0.
            # The gamma factor + zero terminal potential make the shaping
            # policy-invariant: it can only speed learning, never change the
            # optimal policy, and it never double-counts the env's win/loss
            # reward at the final step (Phi(s') is forced to 0 there).
            obs_after = env.state[p]["observation"]
            # The seat's own elimination is terminal for its trajectory, so zero
            # its next potential too (same PBRS treatment as the env's terminal).
            seat_dead = _is_eliminated(obs_after, p)
            phi_next = 0.0 if (terminal or seat_dead) else _lead_margin(obs_after, p)
            sval = GAMMA * phi_next - before[p]
            # Non-potential behavior penalty: a tracked fleet that vanished this
            # turn leaving its target at 0 ships and untaken (see _zeroing_penalty),
            # scaled by the target's production.
            penalty = _zeroing_penalty(fleet_target[p], launch_tgt.get(p, {}),
                                       obs_seen[p], obs_after, p)
            shaped[p].append(sval - ZERO_PENALTY_COEF * penalty)
            if seat_dead:
                dead.add(p)           # stop recording this seat from next turn on
    rewards = [float(env.state[p]["reward"] or 0.0) for p in range(n)]
    # Winner-take-all reward straight from the env's terminal rewards (highest =
    # winner; the env already decides this). Sole winner +1, everyone else -1, in
    # both 2p and 4p (a 4p win is worth the same +1 as a 2p win).
    rank_rewards = _rank_rewards(rewards, n)
    return data, rewards, shaped, turns, rank_rewards, track_fracs


_W_LEARNER = None    
_W_OPP = []          
_W_ENV = None        


def _worker_init():
    global _W_LEARNER, _W_ENV
    torch.set_num_threads(1)               # avoid CPU oversubscription across workers
    logging.disable(logging.WARNING)
    _W_LEARNER = OrbitNet()
    _W_LEARNER.eval()
    _W_ENV = make("orbit_wars")


def _rollout_one(payload):
    learner_sd, opp_sds, opp_is_sl, seat, is_eval = payload
    _W_LEARNER.load_state_dict(learner_sd)
    while len(_W_OPP) < len(opp_sds):
        n = OrbitNet(); n.eval(); _W_OPP.append(n)
    players = len(opp_sds) + 1
    nets = [None] * players
    nets[seat] = _W_LEARNER
    
    # [2] Mixed Execution 도입: 훈련 중에도 일부 상대방은 Greedy를 사용하여 평가(Eval) 상황에 대비
    greedy_seats = set()
    if is_eval:
        greedy_seats.update(range(players))
    else:
        for p in range(players):
            if p != seat and random.random() < GREEDY_RATIO:
                greedy_seats.add(p)
            
    oi = 0
    for p in range(players):
        if p == seat:
            continue
        _W_OPP[oi].load_state_dict(opp_sds[oi])
        nets[p] = _W_OPP[oi]
        oi += 1
        
    record_seats = set() if is_eval else {seat}
    # In eval the learner plays greedily and nothing is recorded, so track its
    # launch fractions explicitly to monitor the deployed (mode) distribution.
    data, rewards, shaped, turns, rank_rewards, track_fracs = play_game(
        nets, record_seats=record_seats, greedy_seats=greedy_seats,
        track_seat=(seat if is_eval else None))
    # `won` (sole winner) stays the env-based eval/gating metric -- the headline
    # objective is still the 1st-place win rate.
    won = rewards[seat] > max(rewards[i] for i in range(players) if i != seat)
    # Training terminal reward = winner-take-all (+1 win / -1 loss, same in 2p & 4p),
    # plus a speed bonus for an outright win. Eval ignores the reward (uses `won`).
    r_seat = rank_rewards[seat]
    if won and not is_eval:
        r_seat += WIN_SPEED_COEF * max(0.0, (WIN_SPEED_TREF - turns) / WIN_SPEED_TREF)
    return (data[seat] if not is_eval else [], r_seat,
            shaped[seat] if not is_eval else [], won, players, track_fracs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--games", type=int, default=99)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--win-gate", type=float, default=None,
                    help="absolute win-rate gate to save a snapshot; default None "
                         "= use the parity (expected) win rate for the game mix")
    ap.add_argument("--rebuild", action="store_true",
                    help="start fresh from the SL policy: delete the RL pool dir "
                         f"({RL_POOL_DIR}/) and RL checkpoints ({RL_BEST}, {RL_LAST}) first")
    args = ap.parse_args()

    if args.rebuild:
        if os.path.isdir(RL_POOL_DIR):
            shutil.rmtree(RL_POOL_DIR)
            print(f"rebuild: removed pool dir {RL_POOL_DIR}/")
        for f in (RL_BEST, RL_LAST):
            if os.path.exists(f):
                os.remove(f)
                print(f"rebuild: removed checkpoint {f}")

    if torch.cuda.is_available():
        device = "cuda"                        # Windows/Linux + NVIDIA
    elif torch.backends.mps.is_available():
        device = "mps"                         # Apple Silicon GPU
    else:
        device = "cpu"
    net = OrbitNet().to(device)
    teacher = OrbitNet().to(device)            
    
    if os.path.exists(SL_WEIGHTS):
        teacher.load_state_dict(_load_state(SL_WEIGHTS, device), strict=False)
    else:
        print(f"warning: no {SL_WEIGHTS}; teacher is random")
    teacher.eval()

    # Resume: if a FULL RL_LAST bundle exists (and not --rebuild), pick up where we
    # left off -- weights, Adam momentum, iteration counter and RNG. A legacy RL_LAST
    # (raw state_dict, no optimizer) only restores weights and starts a fresh schedule.
    start_iter = 0
    resume = None
    if not args.rebuild and os.path.exists(RL_LAST):
        # weights_only=False: the resume bundle holds numpy/python RNG state, which
        # PyTorch>=2.6's default safe-unpickler rejects. This is our own checkpoint.
        ck = torch.load(RL_LAST, map_location=device, weights_only=False)
        if isinstance(ck, dict) and "optimizer_state_dict" in ck:
            resume = ck

    if resume is not None:
        net.load_state_dict(resume["model_state_dict"], strict=False)
        start_iter = int(resume.get("iter", -1)) + 1
        print(f"resuming from {RL_LAST} at iter {start_iter}  "
              f"(optimizer + RNG restored)  |  teacher from {SL_WEIGHTS}")
    else:
        init_src = RL_BEST if os.path.exists(RL_BEST) else \
            (RL_LAST if os.path.exists(RL_LAST) else SL_WEIGHTS)
        if os.path.exists(init_src):
            net.load_state_dict(_load_state(init_src, device), strict=False)
            print(f"learner initialised from {init_src}  |  teacher from {SL_WEIGHTS}")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    if resume is not None:
        opt.load_state_dict(resume["optimizer_state_dict"])
        if "rng" in resume:
            _restore_rng(resume["rng"])

    teacher_sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    best_pool = load_pool()
    next_snap_id = max((e["id"] for e in best_pool), default=-1) + 1   # unique pool ids
    if resume is not None:                              # keep ids monotonic across restarts
        next_snap_id = max(next_snap_id, int(resume.get("next_snap_id", 0)))

    # SL teacher as a permanent (never-evicted) league member with id -1, sampled
    # uniformly exactly like any pool snapshot. vssl=0.5 (parity vs itself) so it
    # only fills 4p-eval slots when the pool has fewer than 3 stronger members.
    sl_entry = {"id": -1, "sd": teacher_sd, "vssl": 0.5}
    if best_pool:
        print(f"restored {len(best_pool)} pool snapshot(s) from {RL_POOL_DIR}/")

    workers = max(1, min(args.workers, max(args.games, EVAL_GAMES_2P + EVAL_GAMES_4P)))
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
            if workers > 1 else None)
    if pool is not None:
        # Ensure workers are torn down on ANY graceful exit (normal end, Ctrl-C,
        # unhandled exception) -- not just the explicit shutdown at the end, which
        # is skipped on interrupt and leaks orphaned worker processes. cancel_futures
        # drops queued rollouts so cleanup only waits on the few in-flight games.
        atexit.register(pool.shutdown, cancel_futures=True)
    if pool is None:
        _worker_init()
    print(f"rollout workers: {workers}  |  games/iter: {args.games} tr, "
          f"{EVAL_GAMES_2P}+{EVAL_GAMES_4P} eval (2p+4p)  | pool size: {POOL_SIZE}")

    def as_ref(e):
        return (e["sd"], e["id"], e["id"] == -1)

    def league():
        # Full opponent league: all pool snapshots + the SL teacher, all sampled
        # with equal (uniform) probability.
        return list(best_pool) + [sl_entry]

    def uniform_pick(candidates):
        # Uniformly sample one opponent from the league.
        return as_ref(random.choice(candidates))

    def strongest_refs(k=N_OPP):
        # The k strongest pool members by their recorded vsSL eval score (the win
        # rate each passed gating with). Padded with the SL teacher when the pool
        # has fewer than k members. Used to seed the 4-player eval opponents.
        ranked = sorted(best_pool, key=lambda e: e.get("vssl", 0.0), reverse=True)
        picks = list(ranked[:k])
        while len(picks) < k:
            picks.append(sl_entry)
        return [as_ref(e) for e in picks]

    def opp_trio():
        # Three opponents for a 4-player game, sampled uniformly from the league.
        # No fixed anchor: every seat is a fresh uniform draw. Sample WITHOUT
        # replacement when the league is large enough, else top up WITH replacement
        # (small early league). SL teacher is just another member.
        cands = league()
        if len(cands) >= N_OPP:
            picks = random.sample(cands, N_OPP)
        else:
            picks = [random.choice(cands) for _ in range(N_OPP)]
        return [as_ref(e) for e in picks]

    def make_tasks(learner_sd, is_eval=False):
        tasks, tags, opp_ids = [], [], []

        if is_eval:
            # Eval is a FIXED yardstick, not uniform self-play, with fixed sample
            # sizes (EVAL_GAMES_2P / EVAL_GAMES_4P) independent of the training split:
            #   2p -> always vs the SL teacher (this win rate IS the performance
            #         metric we report and gate on);
            #   4p -> vs the 3 strongest pool members (highest recorded vsSL score).
            n2, n4 = EVAL_GAMES_2P, EVAL_GAMES_4P
            sl_sd, _sl_id, sl_is_sl = as_ref(sl_entry)
            for j in range(n2):
                tasks.append((learner_sd, [sl_sd], [sl_is_sl], j % 2, True))
                tags.append("sl2p")
                opp_ids.append(-1)
            refs = strongest_refs()                  # same trio for every 4p eval game
            sds = [r[0] for r in refs]
            is_sls = [r[2] for r in refs]
            for j in range(n4):
                tasks.append((learner_sd, sds, is_sls, j % 4, True))
                tags.append("4p")
                opp_ids.append(None)
            return tasks, tags, opp_ids

        # Training: uniform self-play opponents, 2:1 (2p:4p) split.
        n4 = args.games // 3
        n2 = args.games - n4
        cands = league()
        for j in range(n2):
            sd_k, id_k, is_sl_k = uniform_pick(cands)   # fresh uniform opponent per game
            tasks.append((learner_sd, [sd_k], [is_sl_k], j % 2, False))
            tags.append("sl2p" if is_sl_k else "best2p")
            opp_ids.append(id_k)                     # clean attribution in 2p
        for j in range(n4):
            refs = opp_trio()                        # fresh trio per game
            sds = [r[0] for r in refs]
            is_sls = [r[2] for r in refs]
            tasks.append((learner_sd, sds, is_sls, j % 4, False))
            tags.append("4p")
            opp_ids.append(None)                     # ambiguous in 4p -> no stat update
        return tasks, tags, opp_ids

    def rollouts(is_eval=False):
        learner_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        tasks, tags, opp_ids = make_tasks(learner_sd, is_eval=is_eval)
        results = pool.map(_rollout_one, tasks) if pool is not None \
            else (_rollout_one(tk) for tk in tasks)
        yield from zip(tags, opp_ids, results)

    # On resume, APPEND to the log/monitor streams instead of truncating them so the
    # run's history (and the live monitor's tail) stays continuous across restarts.
    log_f = open(RL_LOG_CSV, "a" if resume is not None else "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if resume is None:
        open(LAUNCH_DIST_FILE, "w", encoding="utf-8").close()       # fresh per run (live monitor tails it)
        open(LAUNCH_DIST_EVAL_FILE, "w", encoding="utf-8").close()  # eval (greedy) launch distribution
        log.writerow(["iter", "steps", "win", "win2", "win4",
                  # win = win2 = vssl = vs-SL 2p win rate (the performance metric).
                  # win4 = win rate in 4p vs the 3 strongest pool members (by vsSL score).
                  "vssl", "wsl", "gsl",          # win rate vs the fixed SL teacher (== headline)
                  "vsbest", "wbest", "gbest",    # deprecated (always 0): no best2p phase in eval now
                  "policy", "value", "entropy", "kl_div", "tr_win", "tr_win2", "tr_win4",
                  # objective + PPO update-health diagnostics (the policy loss is a
                  # near-0 surrogate, NOT a progress signal -- mean_ret is the actual
                  # RL objective; explained_var/approx_kl/clip_frac diagnose the update).
                  "mean_ret", "explained_var", "approx_kl", "clip_frac"])

    pbar = tqdm(range(start_iter, args.iters), desc="RL", unit="iter", initial=start_iter, total=args.iters)
    for it in pbar:
        # =======================================================
        # 1. 훈련 모드
        # =======================================================
        net.eval()
        t_roll = time.perf_counter()
        
        # shape_coef is the module constant SHAPE_COEF, held constant unless
        # SHAPE_DECAY_FRAC > 0, in which case it decays linearly to 0 over that
        # fraction of the run (PBRS keeps the optimum invariant; decay only shifts
        # the learning aid).
        shape_coef = SHAPE_COEF
        if SHAPE_DECAY_FRAC > 0:
            shape_coef *= max(0.0, 1.0 - it / max(1, SHAPE_DECAY_FRAC * args.iters))

        # Teacher-KL anchor: decay START -> FLOOR linearly over KL_DECAY_FRAC of the
        # run (frac=0 -> constant START). Used in the PPO loss + logging below.
        kl_coef = KL_COEF_START
        if KL_DECAY_FRAC > 0:
            frac = max(0.0, 1.0 - it / max(1, KL_DECAY_FRAC * args.iters))
            kl_coef = KL_COEF_FLOOR + (KL_COEF_START - KL_COEF_FLOOR) * frac

        feats_buf, valid_buf, mine_buf, rec_buf, step_rewards_buf, dones_buf = [], [], [], [], [], []
        tr_w = tr_g = 0
        tr_w2 = tr_g2 = tr_w4 = tr_g4 = 0     # split train win rate by game type

        for tag, opp_id, (seat_data, reward, shaped, won, players, _ef) in tqdm(rollouts(is_eval=False), total=args.games,
                                                                           desc="train rollout", unit="game", leave=False):
            tr_g += 1; tr_w += won
            if players == 2:
                tr_g2 += 1; tr_w2 += won
            else:
                tr_g4 += 1; tr_w4 += won
            T = len(seat_data)
            for t, (feats, valid, mine, rec) in enumerate(seat_data):
                feats_buf.append(feats)
                valid_buf.append(valid)
                mine_buf.append(mine)
                rec_buf.append(rec)
                r = shape_coef * shaped[t] + (reward if t == T - 1 else 0.0)
                step_rewards_buf.append(r)
                dones_buf.append(t == T - 1)  
                
        tr_win_rate = tr_w / tr_g if tr_g else 0.0
        tr_win2 = tr_w2 / tr_g2 if tr_g2 else 0.0
        tr_win4 = tr_w4 / tr_g4 if tr_g4 else 0.0
        roll_s = time.perf_counter() - t_roll

       # Live monitor: histogram this iter's SAMPLED launch fractions (gate-fired
        # planets only) so watch_launch_dist.py can show the ship-fraction shape.
        # (it + 1) % EVAL_EVERY == 0 루프 아래 및 위쪽의 _write_launch_dist 부분
        _write_launch_dist(it + 1,
                           [(1.0 if f >= OrbitNet.SHIP_BINS - 1 else (f + 0.5) / (OrbitNet.SHIP_BINS - 1))
                            for rec in rec_buf for (_i, la, _t, f) in rec if la])

        # =======================================================
        # 2. PPO 업데이트
        # =======================================================
        t_upd = time.perf_counter()
        
        all_feats = torch.tensor(np.stack(feats_buf), dtype=torch.float32, device=device)
        all_valid = torch.tensor(np.stack(valid_buf), dtype=torch.bool, device=device)
        all_mine = torch.tensor(np.stack(mine_buf), dtype=torch.float32, device=device)
        all_step_rewards = torch.tensor(step_rewards_buf, dtype=torch.float32, device=device)
        
        all_logp_old = []
        all_values_old = []

        with torch.no_grad():
            for s in range(0, len(step_rewards_buf), MINIBATCH):
                chunk_fb = all_feats[s:s+MINIBATCH]
                chunk_vb = all_valid[s:s+MINIBATCH]
                chunk_mb = all_mine[s:s+MINIBATCH]
                chunk_recs = rec_buf[s:s+MINIBATCH]

                lp, v, _ = net.evaluate_actions(chunk_fb, chunk_vb, chunk_mb, chunk_recs,
                                                gate_bias=GATE_SAMPLE_BIAS)
                all_logp_old.append(lp)
                all_values_old.append(v)

        all_logp_old = torch.cat(all_logp_old)
        all_values_old = torch.cat(all_values_old)

        # Teacher KL is now a differentiable loss term in the PPO update below
        # (KL(teacher||student) per component), NOT a reward penalty. This keeps
        # the value function estimating the true return and gives a direct BC
        # gradient toward the human policy (cf. AlphaStar human_policy_kl_loss).
        # avg_kl is accumulated in the update loop for logging.

        r_arr = all_step_rewards.cpu().numpy()
        v_arr = all_values_old.cpu().numpy()
        d_arr = np.array(dones_buf, dtype=bool)
        adv_arr = np.zeros_like(r_arr)
        
        lastgaelam = 0.0
        for t in reversed(range(len(r_arr))):
            if d_arr[t]:
                nextnonterminal = 0.0
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0
                nextvalues = v_arr[t+1]
                
            delta = r_arr[t] + GAMMA * nextvalues * nextnonterminal - v_arr[t]
            lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            adv_arr[t] = lastgaelam
            
        adv_buf = torch.tensor(adv_arr, dtype=torch.float32, device=device)
        all_ret = adv_buf + all_values_old

        if DEBUG:
            def _c(x):  # compact "mean±std[min,max]" for a tensor/array
                x = x.detach().float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
                return f"{x.mean():+.2f}±{x.std():.2f}[{x.min():+.2f},{x.max():+.2f}]"
            done_mask = torch.tensor(dones_buf, device=device)
            term_r = all_step_rewards[done_mask]               # env win/loss at episode end
            launches = sum(1 for rec in rec_buf for (_i, la, _t, _f) in rec if la)  # gate fires
            owned = sum(len(rec) for rec in rec_buf)
            tqdm.write(
                f"  [dbg it{it+1}] n={len(step_rewards_buf)} games={tr_g} "
                f"launch={launches}/{owned} ({100*launches/max(1,owned):.0f}%) | "
                f"rew {_c(all_step_rewards)} term {_c(term_r)} | "
                f"val {_c(all_values_old)} ret {_c(all_ret)} adv {_c(adv_arr)}"
            )

        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        net.train()
        tot_loss = tot_p = tot_v = tot_e = tot_k = 0.0
        sum_approx_kl = sum_clip_frac = 0.0
        nb = 0

        for epoch in range(PPO_EPOCHS):
            order = list(range(len(step_rewards_buf)))
            random.shuffle(order)
            
            for s in range(0, len(order), MINIBATCH):
                chunk = order[s:s + MINIBATCH]
                
                fb = all_feats[chunk]
                vb = all_valid[chunk]
                mb = all_mine[chunk]
                recs = [rec_buf[j] for j in chunk]
                ret = all_ret[chunk]
                old_logp = all_logp_old[chunk]
                mb_adv = adv_buf[chunk]

                logp, value, ent, kl = net.evaluate_actions(fb, vb, mb, recs, teacher=teacher,
                                                            gate_bias=GATE_SAMPLE_BIAS)

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    # Update-health diagnostics (not part of the loss):
                    #   approx_kl -- how far this update moved the policy from the
                    #     data-collection policy (PPO blows up if it runs away).
                    #   clip_frac -- fraction of samples hitting the PPO_CLIP rail.
                    approx_kl = (old_logp - logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > PPO_CLIP).float().mean()

                value_loss = ((ret - value) ** 2).mean()
                entropy = ent.mean()
                kl_loss = kl.mean()                       # KL(teacher||student) BC anchor
                loss = (policy_loss + VALUE_COEF * value_loss
                        - ENTROPY_COEF * entropy + kl_coef * kl_loss)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

                tot_loss += loss.item()
                tot_p += policy_loss.item(); tot_v += value_loss.item(); tot_e += entropy.item()
                tot_k += kl_loss.item(); nb += 1
                sum_approx_kl += approx_kl.item(); sum_clip_frac += clip_frac.item()

        upd_s = time.perf_counter() - t_upd
        loss_val = tot_loss / nb
        p_loss, v_loss, e_loss = tot_p / nb, tot_v / nb, tot_e / nb
        avg_kl = tot_k / nb
        avg_approx_kl = sum_approx_kl / nb
        avg_clip_frac = sum_clip_frac / nb
        # The actual RL objective: average return over this iter's rollout data.
        # This is what should trend UP if learning works (policy loss won't show it).
        mean_return = float(all_ret.mean())
        # Critic quality: 1 = value head perfectly predicts returns, 0 = no better
        # than guessing the mean, <0 = worse than the mean.
        explained_var = float(1.0 - (all_ret - all_values_old).var() / (all_ret.var() + 1e-8))
        
        # =======================================================
        # 3. 평가 모드 (단일 패스, 2p 100판 + 4p 20판) -- EVAL_EVERY iter마다만
        # =======================================================
        do_eval = ((it + 1) % EVAL_EVERY == 0) or (it == args.iters - 1)
        net.eval()
        t_eval = time.perf_counter()

        if do_eval:
            eval_wsl = eval_gsl = eval_w4 = eval_g4 = 0
            eval_fracs = []                            # learner's GREEDY launch fractions

            # Fixed-yardstick eval: 2p vs SL (the performance metric) + 4p vs the
            # 3 strongest pool members. Performance/gating = the vsSL 2p win rate.
            for tag, _opp_id, (_, _, _, won, players, efracs) in tqdm(rollouts(is_eval=True), total=EVAL_GAMES_2P + EVAL_GAMES_4P,
                                                                      desc="eval", unit="game", leave=False):
                if tag == "sl2p":
                    eval_gsl += 1; eval_wsl += won
                else:                                  # "4p" vs the strongest trio
                    eval_g4 += 1; eval_w4 += won
                eval_fracs.extend(efracs)

            # Greedy (mode) launch distribution at eval, for the live monitor to
            # overlay against the SAMPLED training distribution.
            _write_launch_dist(it + 1, eval_fracs, path=LAUNCH_DIST_EVAL_FILE)
            wsl, gsl = eval_wsl, eval_gsl
            win_rate = wsl / max(1, gsl)               # vsSL 2p == headline performance
            win2 = win_rate
            win4 = eval_w4 / max(1, eval_g4)           # vs the 3 strongest (diagnostic)
            wbest = gbest = 0                          # best2p no longer part of eval
        else:
            win_rate = win2 = win4 = None
            wsl = gsl = 0
            wbest = gbest = 0

        eval_s = time.perf_counter() - t_eval

        sl_str = f"{wsl}/{gsl}" if gsl else "-"
        win4_str = f"{eval_w4}/{eval_g4}" if do_eval and eval_g4 else "-"
        if do_eval:
            eval_str = (f"EVAL vsSL={win_rate:.2f} ({sl_str})  "
                        f"4p-vs-top3={win4:.2f} ({win4_str})")
        else:
            eval_str = "EVAL skipped"
        line = (f"iter {it+1:4d}  steps={len(step_rewards_buf)}  tr_s={roll_s:.0f}s upd={upd_s:.0f}s ev_s={eval_s:.0f}s  "
                f"{eval_str} [Tr_Win={tr_win_rate:.2f} (2p={tr_win2:.2f} 4p={tr_win4:.2f})]\n"
                f"      objective(mean_return)={mean_return:+.3f}  loss={loss_val:+.4f}  entropy={e_loss:.3f}  "
                f"explained_var={explained_var:.2f}\n"
                f"      loss terms: policy={p_loss:+.4f} value={VALUE_COEF*v_loss:+.4f} "
                f"entropy={-ENTROPY_COEF*e_loss:+.4f} kl={kl_coef*avg_kl:+.4f}  |  "
                f"raw: value_loss={v_loss:.4f} teacher_kl={avg_kl:.4f} "
                f"approx_kl={avg_approx_kl:.4f} clip_frac={avg_clip_frac:.2f}")

        # Gate on the vsSL 2p win rate beating 2-player parity (0.5), or --win-gate.
        gate = (args.win_gate if args.win_gate is not None
                else parity_winrate(eval_gsl, 0)) if do_eval else None
        if do_eval and win_rate > gate:
            torch.save(net.state_dict(), RL_BEST)
            snap_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            # Record the vsSL score so this snapshot can be ranked as a 4p opponent.
            best_pool.appendleft({"id": next_snap_id, "sd": snap_sd, "vssl": win_rate})
            next_snap_id += 1
            save_pool(best_pool)
            line += f"  <- eval>{gate:.2f}, saved {RL_BEST} (+pool {len(best_pool)})"

        pbar.set_postfix(eval_win=(f"{win_rate:.2f}" if do_eval else "-"),
                         vsSL=sl_str, vs4p=(f"{win4:.2f}" if do_eval else "-"), pool=len(best_pool),
                         objective=f"{mean_return:+.2f}", loss=f"{loss_val:+.2f}",
                         entropy=f"{e_loss:.2f}", teacher_kl=f"{avg_kl:.3f}")
        tqdm.write(line)
        log.writerow([it + 1, len(step_rewards_buf),
                      ("" if win_rate is None else f"{win_rate:.4f}"),
                      ("" if win2 is None else f"{win2:.4f}"),
                      ("" if win4 is None else f"{win4:.4f}"),
                      (f"{wsl/gsl:.4f}" if gsl else ""), wsl, gsl,
                      (f"{wbest/gbest:.4f}" if gbest else ""), wbest, gbest,
                      f"{p_loss:.6f}", f"{v_loss:.6f}", f"{e_loss:.6f}", f"{avg_kl:.6f}",
                      f"{tr_win_rate:.4f}", f"{tr_win2:.4f}", f"{tr_win4:.4f}",
                      f"{mean_return:.6f}", f"{explained_var:.4f}",
                      f"{avg_approx_kl:.6f}", f"{avg_clip_frac:.4f}"])
        log_f.flush()
        save_resume(RL_LAST, net, opt, it, next_snap_id)   # full bundle -> true resume

    log_f.close()
    if pool is not None:
        pool.shutdown()
    print(f"done. latest -> {RL_LAST}, last gated best -> {RL_BEST}  (log -> {RL_LOG_CSV})")


if __name__ == "__main__":
    main()
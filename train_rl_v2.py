"""A2C/PPO RL for OrbitNet, initialised from the best RL checkpoint (or the SL policy
on a fresh start) and trained against a snapshot-pool league of its recent bests.

Features GAE, Teacher KL Regularization, Mixed Opponent Execution, and 2-Stage Evaluation.
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

logging.disable(logging.WARNING)
from kaggle_environments import make
from main import _aim, _build_world, _fleet_speed, GATE_THRESHOLD, K_MAX_GLOBAL

# PPO & RL Hyperparameters
GAMMA = 0.997
GAE_LAMBDA = 0.95                  
LR = 5e-6                          
VALUE_COEF = 0.5

# [1] 엔트로피 붕괴(Collapse)를 막기 위해 0.01 보너스 추가
ENTROPY_COEF = 0.01                 
KL_COEF = 0.05                     
MINIBATCH = 1024

PPO_CLIP = 0.1
PPO_EPOCHS = 4

# Evaluation is itself a full set of rollouts (eval_games per iter, doubled when
# stage 2 triggers). Running it every iter doubles rollout cost, so gate it.
EVAL_EVERY = 2

# [3] 카타스트로픽 망각을 막기 위한 League Pool 확장
POOL_SIZE = 20                     
N_OPP = 3                          

SL_WEIGHTS = "orbit_net.pt"
RL_BEST = "orbit_net_rl.pt"        
RL_LAST = "orbit_net_rl_last.pt"   
RL_POOL_DIR = "rl_pool"            
RL_LOG_CSV = "train_rl_log.csv"    


def _pool_path(i):
    return os.path.join(RL_POOL_DIR, f"snap{i}.pt")


def load_pool():
    dq = deque(maxlen=POOL_SIZE)
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
        dq.append(sd)
    return dq


def save_pool(best_pool):
    os.makedirs(RL_POOL_DIR, exist_ok=True)
    for i, sd in enumerate(best_pool):
        torch.save(sd, _pool_path(i))
    for i in range(len(best_pool), POOL_SIZE):       
        if os.path.exists(_pool_path(i)):
            os.remove(_pool_path(i))


def to_action(decisions, obs, ids, ships):
    # Most turns the gate launches nothing; skip the (expensive) world build and
    # full trajectory precompute below when there is nothing to aim.
    if not decisions:
        return []

    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in raw}
    world = _build_world(obs)

    # 매 스텝 궤적 전처리
    world_trajectories = []
    radii = []
    for w in world:
        traj = [w["predict"](k) for k in range(K_MAX_GLOBAL + 1)]
        world_trajectories.append(traj)
        radii.append(w["radius"])
        
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    moves = []
    for src, tgt, frac in decisions:
        available = float(ships[src])
        n = min(int(round(frac * available)), int(available))
        if n < 1:
            continue
            
        spl = by_id.get(int(ids[src]))
        target_id = int(ids[tgt])
        
        if spl is None or target_id not in id_to_idx:
            continue
            
        target_idx = id_to_idx[target_id]
        angle = _aim(spl, world_trajectories, radii, target_idx, _fleet_speed(n), kmax=K_MAX_GLOBAL)
        if angle is None:
            continue
        moves.append([int(ids[src]), angle, n])
        
    return moves


def _ship_share(obs, me):
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    mine = (sum(p[5] for p in planets if p[1] == me)
            + sum(f[6] for f in fleets if f[1] == me))
    total = sum(p[5] for p in planets) + sum(f[6] for f in fleets)
    return (mine / total) if total > 0 else 0.0


def play_game(nets, record_seats=None, greedy_seats=None):
    n = len(nets)
    keep = set(range(n)) if record_seats is None else set(record_seats)
    greedy_seats = greedy_seats or set()
    env = make('orbit_wars')
    env.reset(num_agents=n)
    data = [[] for _ in range(n)]
    shaped = [[] for _ in range(n)]
    eta_caches = [{} for _ in range(n)]
    while not env.done:
        actions = []
        before = {}
        for p in range(n):
            obs = env.state[p]["observation"]
            feats, valid, mine, ids, xs, ys, ships = F.extract(obs, eta_cache=eta_caches[p])
            if p in keep:                                  
                decisions, rec = nets[p].act_rl(feats, valid, mine)
                data[p].append((feats, valid, mine, rec))
                before[p] = _ship_share(obs, p)            
            elif p in greedy_seats:                        
                decisions = nets[p].act(feats, valid, mine, greedy=True,
                                        gate_threshold=GATE_THRESHOLD)
            else:                                          
                decisions, _ = nets[p].act_rl(feats, valid, mine)
            actions.append(to_action(decisions, obs, ids, ships))
        env.step(actions)
        for p in keep:                                     
            shaped[p].append(_ship_share(env.state[p]["observation"], p) - before[p])
    rewards = [float(env.state[p]["reward"] or 0.0) for p in range(n)]
    return data, rewards, shaped


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
            if p != seat and random.random() < 0.2:
                greedy_seats.add(p)
            
    oi = 0
    for p in range(players):
        if p == seat:
            continue
        _W_OPP[oi].load_state_dict(opp_sds[oi])
        nets[p] = _W_OPP[oi]
        oi += 1
        
    record_seats = set() if is_eval else {seat}
    data, rewards, shaped = play_game(nets, record_seats=record_seats,
                                      greedy_seats=greedy_seats)
    won = rewards[seat] > max(rewards[i] for i in range(players) if i != seat)
    return data[seat] if not is_eval else [], rewards[seat], shaped[seat] if not is_eval else [], won, players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=16, help="games per iter for pure greedy evaluation")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--win-gate", type=float, default=0.44)
    ap.add_argument("--shape-coef", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = OrbitNet().to(device)
    teacher = OrbitNet().to(device)            
    
    if os.path.exists(SL_WEIGHTS):
        teacher.load_state_dict(torch.load(SL_WEIGHTS, map_location=device), strict=False)
    else:
        print(f"warning: no {SL_WEIGHTS}; teacher is random")
    teacher.eval() 
    
    init_src = RL_BEST if os.path.exists(RL_BEST) else SL_WEIGHTS
    if os.path.exists(init_src):
        net.load_state_dict(torch.load(init_src, map_location=device), strict=False)
        print(f"learner initialised from {init_src}  |  teacher from {SL_WEIGHTS}")
        
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    teacher_sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    best_pool = load_pool()
    if best_pool:
        print(f"restored {len(best_pool)} pool snapshot(s) from {RL_POOL_DIR}/")

    workers = max(1, min(args.workers, args.games + args.eval_games))
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
            if workers > 1 else None)
    if pool is None:
        _worker_init()                         
    print(f"rollout workers: {workers}  |  games/iter: {args.games} tr, {args.eval_games} eval  "
          f"| pool size: {POOL_SIZE}")

    def opp_refs():
        # [3] 카타스트로픽 망각 방지를 위한 League 풀 다각화 구성
        pool_list = list(best_pool)
        refs = []
        if pool_list:
            refs.append(pool_list[0])  # 최근 SOTA는 반드시 1자리 차지
        while len(refs) < N_OPP - 1 and pool_list:
            refs.append(random.choice(pool_list))  # 나머지는 풀에서 과거 메타 랜덤 샘플링
        while len(refs) < N_OPP:
            refs.append(teacher_sd)    # 남는 자리는 항상 기본 Teacher 할당
        return refs[:N_OPP]

    def make_tasks(learner_sd, is_eval=False):
        refs = opp_refs()
        refs_is_sl = [r is teacher_sd for r in refs]         
        target_games = args.eval_games if is_eval else args.games
        n4 = target_games // 4
        n2 = target_games - n4
        tasks, tags = [], []
        for j in range(n2):                                  
            k = j % N_OPP
            tasks.append((learner_sd, [refs[k]], [refs_is_sl[k]], j % 2, is_eval))
            tags.append("sl2p" if refs_is_sl[k] else "best2p")
        for j in range(n4):                                  
            tasks.append((learner_sd, list(refs), list(refs_is_sl), j % 4, is_eval))
            tags.append("4p")
        return tasks, tags

    def rollouts(is_eval=False):
        learner_sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        tasks, tags = make_tasks(learner_sd, is_eval=is_eval)
        results = pool.map(_rollout_one, tasks) if pool is not None \
            else (_rollout_one(tk) for tk in tasks)
        yield from zip(tags, results)

    log_f = open(RL_LOG_CSV, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "steps", "win", "win2", "win4", "vssl", "wsl", "gsl",
                  "policy", "value", "entropy", "kl_div", "tr_win"])

    pbar = tqdm(range(args.iters), desc="RL", unit="iter")
    for it in pbar:
        # =======================================================
        # 1. 훈련 모드
        # =======================================================
        net.eval()
        t_roll = time.perf_counter()
        
        feats_buf, valid_buf, mine_buf, rec_buf, step_rewards_buf, dones_buf = [], [], [], [], [], []
        tr_w = tr_g = 0
        
        for tag, (seat_data, reward, shaped, won, players) in tqdm(rollouts(is_eval=False), total=args.games,
                                                                   desc="train rollout", unit="game", leave=False):
            tr_g += 1; tr_w += won
            T = len(seat_data)
            for t, (feats, valid, mine, rec) in enumerate(seat_data):
                feats_buf.append(feats)
                valid_buf.append(valid)
                mine_buf.append(mine)
                rec_buf.append(rec)
                r = args.shape_coef * shaped[t] + (reward if t == T - 1 else 0.0)
                step_rewards_buf.append(r)
                dones_buf.append(t == T - 1)  
                
        tr_win_rate = tr_w / tr_g if tr_g else 0.0
        roll_s = time.perf_counter() - t_roll

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
        all_logp_teacher = []
        
        with torch.no_grad():
            for s in range(0, len(step_rewards_buf), MINIBATCH):
                chunk_fb = all_feats[s:s+MINIBATCH]
                chunk_vb = all_valid[s:s+MINIBATCH]
                chunk_mb = all_mine[s:s+MINIBATCH]
                chunk_recs = rec_buf[s:s+MINIBATCH]
                
                lp, v, _ = net.evaluate_actions(chunk_fb, chunk_vb, chunk_mb, chunk_recs)
                all_logp_old.append(lp)
                all_values_old.append(v)
                
                lp_t, _, _ = teacher.evaluate_actions(chunk_fb, chunk_vb, chunk_mb, chunk_recs)
                all_logp_teacher.append(lp_t)
                
        all_logp_old = torch.cat(all_logp_old)
        all_values_old = torch.cat(all_values_old)
        all_logp_teacher = torch.cat(all_logp_teacher)

        # [4] KL Penalty 안정화 (음수 페널티 차단 및 분산 제어)
        kl_penalty = torch.clamp(all_logp_old - all_logp_teacher, min=0.0)
        all_step_rewards -= KL_COEF * kl_penalty
        avg_kl = kl_penalty.mean().item()

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
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        net.train()
        tot_p = tot_v = tot_e = 0.0
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

                logp, value, ent = net.evaluate_actions(fb, vb, mb, recs)
                
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
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
        
        # =======================================================
        # 3. 평가 모드 (2단계 심층 검증 적용) -- EVAL_EVERY iter마다만
        # =======================================================
        do_eval = ((it + 1) % EVAL_EVERY == 0) or (it == args.iters - 1)
        net.eval()
        t_eval = time.perf_counter()

        if do_eval:
            eval_w2 = eval_g2 = eval_w4 = eval_g4 = eval_wsl = eval_gsl = 0

            for tag, (_, _, _, won, players) in tqdm(rollouts(is_eval=True), total=args.eval_games,
                                                     desc="eval stage 1", unit="game", leave=False):
                if players == 2:
                    eval_g2 += 1; eval_w2 += won
                else:
                    eval_g4 += 1; eval_w4 += won
                if tag == "sl2p":
                    eval_gsl += 1; eval_wsl += won

            stage1_win_rate = (eval_w2 + eval_w4) / max(1, (eval_g2 + eval_g4))

            stage2_triggered = False
            if stage1_win_rate >= args.win_gate - 0.1:
                stage2_triggered = True
                for tag, (_, _, _, won, players) in tqdm(rollouts(is_eval=True), total=args.eval_games,
                                                         desc="eval stage 2 (deep)", unit="game", leave=False):
                    if players == 2:
                        eval_g2 += 1; eval_w2 += won
                    else:
                        eval_g4 += 1; eval_w4 += won
                    if tag == "sl2p":
                        eval_gsl += 1; eval_wsl += won

            win_rate = (eval_w2 + eval_w4) / max(1, (eval_g2 + eval_g4))
            win2 = eval_w2 / max(1, eval_g2)
            win4 = eval_w4 / max(1, eval_g4)
            wsl, gsl = eval_wsl, eval_gsl
        else:
            stage2_triggered = False
            win_rate = win2 = win4 = None
            wsl = gsl = 0

        eval_s = time.perf_counter() - t_eval

        sl_str = f"{wsl}/{gsl}" if gsl else "-"
        if do_eval:
            stage_str = "(Stage 1+2)" if stage2_triggered else "(Stage 1)"
            eval_str = (f"EVAL WIN={win_rate:.2f} {stage_str} "
                        f"(2p={win2:.2f} 4p={win4:.2f} vsSL={sl_str})")
        else:
            eval_str = "EVAL skipped"
        line = (f"iter {it+1:4d}  steps={len(step_rewards_buf)}  tr_s={roll_s:.0f}s upd={upd_s:.0f}s ev_s={eval_s:.0f}s  "
                f"{eval_str} [Tr_Win={tr_win_rate:.2f}]  "
                f"policy={p_loss:.4f} val={v_loss:.4f} ent={e_loss:.3f} kl={avg_kl:.4f}")

        if do_eval and win_rate > args.win_gate:
            torch.save(net.state_dict(), RL_BEST)
            best_pool.appendleft({k: v.detach().cpu() for k, v in net.state_dict().items()})
            save_pool(best_pool)
            line += f"  <- eval>{args.win_gate:.2f}, saved {RL_BEST} (+pool {len(best_pool)})"

        pbar.set_postfix(eval_win=(f"{win_rate:.2f}" if do_eval else "-"), vsSL=sl_str,
                         pool=len(best_pool), kl=f"{avg_kl:.3f}", ent=f"{e_loss:.2f}")
        tqdm.write(line)
        log.writerow([it + 1, len(step_rewards_buf),
                      ("" if win_rate is None else f"{win_rate:.4f}"),
                      ("" if win2 is None else f"{win2:.4f}"),
                      ("" if win4 is None else f"{win4:.4f}"),
                      (f"{wsl/gsl:.4f}" if gsl else ""), wsl, gsl,
                      f"{p_loss:.6f}", f"{v_loss:.6f}", f"{e_loss:.6f}", f"{avg_kl:.6f}", f"{tr_win_rate:.4f}"])
        log_f.flush()
        torch.save(net.state_dict(), RL_LAST)

    log_f.close()
    if pool is not None:
        pool.shutdown()
    print(f"done. latest -> {RL_LAST}, last gated best -> {RL_BEST}  (log -> {RL_LOG_CSV})")


if __name__ == "__main__":
    main()
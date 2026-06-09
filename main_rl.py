"""Agent that plays with the RL-trained weights (orbit_net_rl.pt).

(Standalone Deployment Version)
Uses the greedy policy exactly matching the rigorous Evaluation phase 
of our RL league training. All necessary logic from main.py has been 
merged here to remove the dependency.
"""
import math
import os
import sys

import torch

import features as F
from model import OrbitNet

# ==============================================================================
# 1. Kaggle 제출 환경 경로 설정 및 상수
# ==============================================================================
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
_this_file = globals().get("__file__")   
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _HERE = _KAGGLE_AGENT_DIR
elif _this_file:
    _HERE = os.path.dirname(os.path.abspath(_this_file))
else:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

GATE_THRESHOLD = 0.5
K_MAX_GLOBAL = 50

from physics import (
    CENTER, SUN_RADIUS as SUN_R, BOARD,
    fleet_speed as _fleet_speed,
    seg_point_dist as _seg_point_dist,
    swept_dist as _swept_dist,
    build_world as _build_world,
)

# ==============================================================================
# 2. 발사 각도 및 충돌 예측 로직 (Time-Space Tradeoff 적용)
# ==============================================================================
def _path_blocked(sx, sy, angle, v, world_trajectories, radii, target_idx, k_hit):
    cosA, sinA = math.cos(angle), math.sin(angle)
    px, py = sx, sy                                   
    
    # 딕셔너리({w["id"]: ...}) 대신 미리 계산된 캐시 배열의 0번 인덱스만 $O(1)$로 접근
    prev = [traj[0] for traj in world_trajectories]  
    
    for k in range(1, k_hit + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA  
        
        if _seg_point_dist(px, py, fx, fy, CENTER[0], CENTER[1]) <= SUN_R:
            return True
        if not (0.0 <= fx <= BOARD and 0.0 <= fy <= BOARD):
            return True
            
        for i, traj in enumerate(world_trajectories):
            if i == target_idx:
                continue
                
            q = traj[k]  # 매 틱마다 predict()를 호출하는 대신 배열 캐시 히트
            if q is None:
                continue
                
            p = prev[i]
            prev[i] = q
            
            if p is None:
                continue
                
            if _swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1]) <= radii[i]:
                return True
                
        px, py = fx, fy
    return False

def _aim(src, world_trajectories, radii, target_idx, v, kmax=K_MAX_GLOBAL):
    sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
    target_r = radii[target_idx]
    
    target_traj = world_trajectories[target_idx]

    for k in range(1, kmax + 1):
        pos = target_traj[k]
        if pos is None:
            break  
        px, py = pos
        reach = math.hypot(px - sx, py - sy) - src_r   
        
        if abs(v * k - reach) <= target_r:
            angle = math.atan2(py - sy, px - sx)
            ox = sx + (src_r + 0.01) * math.cos(angle)
            oy = sy + (src_r + 0.01) * math.sin(angle)
            
            # world 객체 대신 전처리된 배열들을 넘김
            if not _path_blocked(ox, oy, angle, v, world_trajectories, radii, target_idx, k):
                return angle
    return None

# ==============================================================================
# 3. RL 에이전트 구동 로직
# ==============================================================================
_net = OrbitNet()
_net.load_state_dict(torch.load(os.path.join(_HERE, "orbit_net_rl.pt"),
                                map_location="cpu"), strict=False)
_net.eval()

def _moves(obs):
    feats, valid, mine, ids, xs, ys, ships = F.extract(obs)
    decisions = _net.act(feats, valid, mine, greedy=True, gate_threshold=GATE_THRESHOLD)
    
    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    by_id = {int(p[0]): p for p in raw}
    world = _build_world(obs)
    
    # --- [핵심] 스텝당 단 한 번, 모든 행성의 400틱 미래를 전처리 (DP 테이블 초기화와 유사) ---
    world_trajectories = []
    radii = []
    for w in world:
        traj = [w["predict"](k) for k in range(K_MAX_GLOBAL + 1)]
        world_trajectories.append(traj)
        radii.append(w["radius"])
        
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    # ---------------------------------------------------------------------------------
    
    moves = []
    for src, tgt, frac in decisions:
        available = float(ships[src])
        n = min(int(round(frac * available)), int(available))   
        if n < 1:
            continue
            
        tpl, spl = by_id.get(int(ids[tgt])), by_id.get(int(ids[src]))
        if tpl is None or spl is None:
            continue
            
        target_id = int(ids[tgt])
        if target_id not in id_to_idx:
            continue
        target_idx = id_to_idx[target_id]
        
        # 바뀐 _aim 시그니처에 맞춰 호출
        angle = _aim(spl, world_trajectories, radii, target_idx, _fleet_speed(n))
        
        if angle is None:
            continue                                          
        moves.append([int(ids[src]), angle, n])
        
    return moves

def agent(obs):           
    return _moves(obs)
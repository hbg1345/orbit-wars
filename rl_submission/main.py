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
# 1. Kaggle 제출 환경 경로 설정 및 상수 (기존 main.py 복사본)
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

# main.py를 거치지 않고 physics.py에서 직접 물리 연산 함수들을 가져옵니다.
from physics import (
    CENTER, SUN_RADIUS as SUN_R, BOARD,
    fleet_speed as _fleet_speed,
    seg_point_dist as _seg_point_dist,
    swept_dist as _swept_dist,
    build_world as _build_world,
)

# ==============================================================================
# 2. 발사 각도 및 충돌 예측 로직 (기존 main.py 복사본)
# ==============================================================================
def _path_blocked(sx, sy, angle, v, world, target_id, k_hit):
    cosA, sinA = math.cos(angle), math.sin(angle)
    px, py = sx, sy                                   
    prev = {w["id"]: w["predict"](0) for w in world}  
    for k in range(1, k_hit + 1):
        fx, fy = sx + v * k * cosA, sy + v * k * sinA  
        if _seg_point_dist(px, py, fx, fy, CENTER[0], CENTER[1]) <= SUN_R:
            return True
        if not (0.0 <= fx <= BOARD and 0.0 <= fy <= BOARD):
            return True
        for w in world:
            q = w["predict"](k)
            p = prev[w["id"]]
            prev[w["id"]] = q
            if w["id"] == target_id or p is None or q is None:
                continue
            if _swept_dist(px, py, fx, fy, p[0], p[1], q[0], q[1]) <= w["radius"]:
                return True
        px, py = fx, fy
    return False

def _aim(src, target, world, v, kmax=400):
    sx, sy, src_r = float(src[2]), float(src[3]), float(src[4])
    target_id = int(target[0])
    target_r = float(target[4])
    tpred = next(w["predict"] for w in world if w["id"] == target_id)

    for k in range(1, kmax + 1):
        pos = tpred(k)
        if pos is None:
            break  
        px, py = pos
        reach = math.hypot(px - sx, py - sy) - src_r   
        if abs(v * k - reach) <= target_r:
            angle = math.atan2(py - sy, px - sx)
            ox = sx + (src_r + 0.01) * math.cos(angle)
            oy = sy + (src_r + 0.01) * math.sin(angle)
            if not _path_blocked(ox, oy, angle, v, world, target_id, k):
                return angle
    return None

# ==============================================================================
# 3. RL 에이전트 구동 로직 (100% 실력 진검승부)
# ==============================================================================
_net = OrbitNet()
# RL 가중치(orbit_net_rl.pt) 로드
_net.load_state_dict(torch.load(os.path.join(_HERE, "orbit_net_rl.pt"),
                                map_location="cpu"), strict=False)
_net.eval()

def _moves(obs):
    feats, valid, mine, ids, xs, ys, ships = F.extract(obs)
    
    # 확률적 샘플링(act_rl) 대신 실전용 그리디(act) 사용
    decisions = _net.act(feats, valid, mine, greedy=True, gate_threshold=GATE_THRESHOLD)
    
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

def agent(obs):           
    return _moves(obs)
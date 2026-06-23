"""Agent that plays with the RL-trained weights (orbit_net_rl.pt).

(Standalone Deployment Version)
Uses the greedy policy exactly matching the rigorous Evaluation phase
of our RL league training. Aiming/collision geometry is shared via forecast.py
(decisions_to_moves), which is bundled alongside this agent.
"""
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

GATE_THRESHOLD = 0.65
K_MAX_GLOBAL = 50

# [수정 1] physics가 아닌 forecast에서 임포트하도록 변경
from forecast import decisions_to_moves as _decisions_to_moves

# ==============================================================================
# 2. RL 에이전트 구동 로직
# ==============================================================================
_device = "cuda" if torch.cuda.is_available() else "cpu"
_net = OrbitNet()

# [수정 2] 모델 가중치 껍데기(Resume Bundle) 벗기기 로직 추가
ckpt_path = os.path.join(_HERE, "rl_pool", "snap0.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 딕셔너리 형태의 Resume 체크포인트라면 순수 가중치만 추출
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]

_net.load_state_dict(ckpt)
_net.to(_device)
_net.eval()

# [수정 3] 캐글 환경에서 두 에이전트가 같은 변수를 덮어쓰지 않도록 dict 형태로 변경
_canon = {}   

def _moves(obs):
    global _canon
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    
    if player not in _canon or step == 0:          # 매 에피소드 시작 시 한 번만 계산
        _canon[player] = F.compute_canon(obs)
        
    # [수정 4] extract에서 반환되는 traj 변수 추가 및 _canon을 플레이어별로 접근
    feats, valid, mine, ids, xs, ys, ships, traj = F.extract(obs, canon=_canon[player])
    
    decisions = _net.act(feats, valid, mine, greedy=True, gate_threshold=GATE_THRESHOLD)
    
    # [수정 5] decisions_to_moves에 누락되었던 horizon과 traj 인자 추가 전달
    return _decisions_to_moves(obs, decisions, ids, ships, kmax=K_MAX_GLOBAL, 
                               horizon=F.HORIZON, traj=traj)

def agent(obs):
    return _moves(obs)
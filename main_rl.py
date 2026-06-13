"""Agent that plays with the RL-trained weights (orbit_net_rl.pt).

(Standalone Deployment Version)
Uses the greedy policy exactly matching the rigorous Evaluation phase
of our RL league training. Aiming/collision geometry is shared via physics.py
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

from physics import decisions_to_moves as _decisions_to_moves

# ==============================================================================
# 2. RL 에이전트 구동 로직 (발사 각도/충돌 로직은 physics.py 공유)
# ==============================================================================
_net = OrbitNet()
_net.load_state_dict(torch.load(os.path.join(_HERE, "orbit_net_rl.pt"),
                                map_location="cpu"), strict=False)
_net.eval()
_canon = None   # per-game canonicalisation (rotate my start -> Q1, CCW enemy keys)

def _moves(obs):
    global _canon
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    if _canon is None or step == 0:          # compute once per episode (reset at step 0)
        _canon = F.compute_canon(obs)
    feats, valid, mine, ids, xs, ys, ships = F.extract(obs, canon=_canon)
    decisions = _net.act(feats, valid, mine, greedy=True, gate_threshold=GATE_THRESHOLD)
    # Aiming + launch conversion is shared geometry (physics.decisions_to_moves).
    return _decisions_to_moves(obs, decisions, ids, ships, kmax=K_MAX_GLOBAL)

def agent(obs):
    return _moves(obs)
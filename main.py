"""Orbit Wars agent.

Policy is OrbitNet (model.py), an AlphaStar-inspired attention network that picks,
for each owned planet, a target planet and a ship count. The trained weights
(orbit_net.pt) are mandatory: if PyTorch or the checkpoint is missing/broken the
agent raises loudly rather than silently playing a random-weight net.

Action format: [[from_planet_id, direction_angle_radians, num_ships], ...]
"""

import math
import os
import sys
from collections import namedtuple

# Same fields as kaggle's Planet; we only read attributes, so a local namedtuple
# is equivalent and avoids depending on the env's internal module path.
Planet = namedtuple("Planet", "id owner x y radius ships production")

# Resolve the agent's own directory so `import model` / `import features` resolve
# and the weights are found. On Kaggle the bundle is extracted to
# /kaggle_simulations/agent/ and this file is run via exec() (so __file__ is
# undefined and cwd is NOT the agent dir -- using cwd here breaks the imports).
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
_this_file = globals().get("__file__")   # undefined when run via exec() (kaggle)
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _HERE = _KAGGLE_AGENT_DIR
elif _this_file:
    _HERE = os.path.dirname(os.path.abspath(_this_file))
else:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

WEIGHTS_PATH = os.path.join(_HERE, "orbit_net.pt")
# Launch when the gate's sigmoid prob exceeds this. The gate is calibrated to the
# ~16% launch base rate, so it sits well below 0.5 even for real launches; ~0.25
# matches the human launch frequency (see inspect_episode.py's threshold sweep).
GATE_THRESHOLD = 0.5
K_MAX_GLOBAL = 50  # 50틱 근거리 탐색 상한선

# Shared physics/geometry lives in physics.py so the agent and the replay parser
# model board motion and collisions identically.
from physics import (
    CENTER, ROTATION_LIMIT, SUN_RADIUS as SUN_R, BOARD,
    fleet_speed as _fleet_speed,
    seg_point_dist as _seg_point_dist,
    swept_dist as _swept_dist,
    comet_predictors as _comet_predictors,
    build_world as _build_world,
)

# The policy + its deps. Imported at module top (not lazily) so a missing module
# fails LOUDLY at load -- no hidden errors, no fallback.
import torch
import features as F
from model import OrbitNet


def _path_blocked(sx, sy, angle, v, world_trajectories, radii, target_idx, k_hit):
    """True if, before reaching the target at turn k_hit, the fleet's straight
    path crosses the sun, leaves the board, or hits another planet/comet.
    """
    cosA, sinA = math.cos(angle), math.sin(angle)
    px, py = sx, sy                                   
    
    # 딕셔너리 대신 0번째 틱(현재 위치)을 O(1)로 가져옴
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
                
            q = traj[k]  # 매 틱마다 predict()를 호출하지 않고 캐시 배열 히트
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
    """Launch angle to hit `target`, or None if uncatchable / path blocked.
    """
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
            if not _path_blocked(ox, oy, angle, v, world_trajectories, radii, target_idx, k):
                return angle
    return None

# --------------------------------------------------------------------------- #
# Neural policy (lazy-loaded singleton).
# --------------------------------------------------------------------------- #
_net = None


def _load_net():
    """Load the neural policy (singleton). The weights are MANDATORY: a missing or
    broken checkpoint raises loudly instead of silently running a random-weight
    net (which just fires at random). No heuristic fallback."""
    global _net
    if _net is None:
        net = OrbitNet()
        # strict=False: the SL checkpoint predates the RL value head, so it lacks
        # those keys (unused at inference). torch.load itself still raises if the
        # file is missing, so a bad bundle fails instead of playing randomly.
        net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"), strict=False)
        net.eval()
        _net = net
    return _net


def _neural_moves(obs, net):
    feats, valid_mask, mine_mask, ids, xs, ys, ships = F.extract(obs)
    decisions = net.act(feats, valid_mask, mine_mask, greedy=True,
                        gate_threshold=GATE_THRESHOLD)

    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    planet_by_id = {int(p[0]): p for p in raw}
    world = _build_world(obs)

    # --- [핵심] 스텝당 단 한 번, 모든 행성의 50틱 미래를 전처리 ---
    world_trajectories = []
    radii = []
    for w in world:
        traj = [w["predict"](k) for k in range(K_MAX_GLOBAL + 1)]
        world_trajectories.append(traj)
        radii.append(w["radius"])
        
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    # ---------------------------------------------------------------

    moves = []
    for src, tgt, frac in decisions:
        available = float(ships[src])
        # the net predicts a fraction of the source garrison to send.
        n = int(round(frac * available))
        if n < 1:
            continue
        n = min(n, int(available))
        tplanet = planet_by_id.get(int(ids[tgt]))
        src_row = planet_by_id.get(int(ids[src]))
        if tplanet is None or src_row is None:
            continue
            
        target_id = int(ids[tgt])
        if target_id not in id_to_idx:
            continue
        target_idx = id_to_idx[target_id]
        
        angle = _aim(src_row, world_trajectories, radii, target_idx, _fleet_speed(n), kmax=K_MAX_GLOBAL)
        if angle is None:
            continue  # no clean shot (blocked, sun, or uncatchable)
        moves.append([int(ids[src]), angle, n])
    return moves


# --------------------------------------------------------------------------- #
# Lead-shot test mode: ignore the model and fire at a rotating target with a
# sweep of fleet sizes (1, 10, 20, ... 500), one launch per turn, so you can
# watch in run_game.py / replay.html whether the lead shot hits moving planets.
# Enable by flipping LEADSHOT_TEST to True below.
# --------------------------------------------------------------------------- #
LEADSHOT_TEST = False

_SWEEP_SIZES = [1] + list(range(10, 501, 10))
_sweep_i = 0


def _leadshot_test_moves(obs):
    global _sweep_i
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    av = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)
    raw = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    planets = [Planet(*p) for p in raw]

    mine = [p for p in planets if p.owner == player]
    if not mine:
        return []
    src = max(mine, key=lambda p: p.ships)        # hoard on our biggest planet

    want = _SWEEP_SIZES[_sweep_i % len(_SWEEP_SIZES)]
    if src.ships < want:
        return []  # wait/accumulate until we can fire this sweep size

    comet_pred = _comet_predictors(obs)
    world = _build_world(obs)

    # --- [핵심] 테스트 모드에서도 동일한 전처리 적용 ---
    world_trajectories = []
    radii = []
    for w in world:
        traj = [w["predict"](k) for k in range(K_MAX_GLOBAL + 1)]
        world_trajectories.append(traj)
        radii.append(w["radius"])
        
    id_to_idx = {w["id"]: i for i, w in enumerate(world)}
    # ---------------------------------------------------

    def priority(p):
        if int(p.id) in comet_pred:
            return 0  # comets first, so we can watch comet lead-shots
        r = math.hypot(p.x - CENTER[0], p.y - CENTER[1])
        if (r + p.radius < ROTATION_LIMIT) and abs(av) > 0.0:
            return 1  # orbiting planet
        return 2      # static planet

    # comet > orbiting > static, then nearest.
    cands = sorted((p for p in planets if p.owner != player),
                   key=lambda p: (priority(p), math.hypot(src.x - p.x, src.y - p.y)))
    for tp in cands:
        target_idx = id_to_idx.get(int(tp.id))
        if target_idx is None:
            continue
            
        angle = _aim(src, world_trajectories, radii, target_idx, _fleet_speed(want), kmax=K_MAX_GLOBAL)
        if angle is None:
            continue  # no clean shot; try the next-nearest
        _sweep_i += 1
        kind = "comet" if int(tp.id) in comet_pred else "planet"
        print(f"[leadshot] fire {want} ships from {src.id} at {kind} {tp.id}")
        return [[src.id, angle, want]]
    return []


def make_agent(weights_path):
    """Build an agent function bound to its OWN net loaded from `weights_path`.
    Lets two different checkpoints (e.g. SL vs RL) play in one process without
    sharing the module-global singleton. Weights are mandatory (no fallback)."""
    path = weights_path if os.path.isabs(weights_path) else os.path.join(_HERE, weights_path)
    net = OrbitNet()
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    net.eval()

    def _agent(obs):
        return _neural_moves(obs, net)
    return _agent


def agent(obs):           # kaggle picks the last callable -> keep this last
    if LEADSHOT_TEST:
        return _leadshot_test_moves(obs)
    return _neural_moves(obs, _load_net())
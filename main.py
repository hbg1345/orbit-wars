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
    CENTER, ROTATION_LIMIT,
    fleet_speed as _fleet_speed,
    comet_predictors as _comet_predictors,
    build_world as _build_world,
    aim as _aim,                       # shared aiming/launch geometry (was local here)
)

# The policy + its deps. Imported at module top (not lazily) so a missing module
# fails LOUDLY at load -- no hidden errors, no fallback.
import torch
import features as F
from model import OrbitNet
# Decision -> moves now needs the forecast (min-capture sizing), so it lives in
# forecast.py (which imports physics), not physics.py.
from forecast import decisions_to_moves as _decisions_to_moves


# --------------------------------------------------------------------------- #
# Neural policy (lazy-loaded singleton).
# --------------------------------------------------------------------------- #
_net = None
# Per-game canonicalisation (rotate my start -> Q1, CCW enemy keys), keyed by
# player id. MUST be per-player: kaggle runs both agents in ONE process, so a
# single shared canon gets clobbered at step 0 by whichever player computes it
# last -- the other player then plays in the wrong rotated frame and (since its
# features go out-of-distribution) never launches. Keying by player id keeps each
# perspective's canon separate. See _neural_moves.
_canon = {}     # player_id -> canon dict


def _load_state(path):
    """Load a checkpoint and return its raw parameter state_dict. train_sl.py saves
    {'model_state_dict', 'optimizer_state_dict'}, so the model weights must be
    pulled out of that wrapper -- passing the whole dict to load_state_dict(...,
    strict=False) silently matches ZERO keys and leaves the net at random init
    (the bug that made the agent never attack). Falls back to the raw-state_dict
    format for older checkpoints. torch.load still raises if the file is missing."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _load_net():
    """Load the neural policy (singleton). The weights are MANDATORY: a missing or
    broken checkpoint raises loudly instead of silently running a random-weight
    net (which just fires at random). No heuristic fallback."""
    global _net
    if _net is None:
        net = OrbitNet()
        net.load_state_dict(_load_state(WEIGHTS_PATH), strict=False)
        net.eval()
        _net = net
    return _net


def _neural_moves(obs, net):
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    # Compute once per episode per perspective (step==0 resets for a new game).
    # Keyed by player so P0 and P1 (same process, same module) don't clobber.
    if player not in _canon or step == 0:
        _canon[player] = F.compute_canon(obs)
    feats, valid_mask, mine_mask, ids, xs, ys, ships, traj = F.extract(obs, canon=_canon[player])
    decisions = net.act(feats, valid_mask, mine_mask, greedy=True,
                        gate_threshold=GATE_THRESHOLD)
    # Decisions are (src, tgt, mode); the heuristic sizes + aims each launch.
    # Reuse extract's forecast (traj) so we don't simulate the board twice.
    return _decisions_to_moves(obs, decisions, ids, ships, kmax=K_MAX_GLOBAL,
                               horizon=F.HORIZON, traj=traj)


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
    net.load_state_dict(_load_state(path), strict=False)
    net.eval()

    def _agent(obs):
        return _neural_moves(obs, net)
    return _agent


def agent(obs):           # kaggle picks the last callable -> keep this last
    if LEADSHOT_TEST:
        return _leadshot_test_moves(obs)
    return _neural_moves(obs, _load_net())
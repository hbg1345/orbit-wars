"""Feature extraction for the Orbit Wars neural agent.

Converts a raw observation (list of planets/comets) into fixed-size tensors the
network can consume, plus the masks the policy needs:

  feats       [MAX_PLANETS, FEATURE_DIM]  float32  per-planet features
  valid_mask  [MAX_PLANETS]               bool     True for real planets
  mine_mask   [MAX_PLANETS]               bool     True for planets I own
  ids / xs / ys / ships  [MAX_PLANETS]    int/float for action post-processing

Each planet's feature vector is a STATIC block (instantaneous board state this
turn) followed by a TEMPORAL block: a forward-simulated trajectory of garrison
and owner over the next HORIZON turns (forecast.py), which lets the network see
incoming fleets, captures, and garrison growth before they happen.
"""

import math
import numpy as np

import forecast

# Board geometry (see orbit_star.md).
BOARD = 100.0
CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0  # orbital_radius + planet_radius < 50 -> orbits

# The game is a fixed-horizon episode: it always ends by this many turns. We feed
# the elapsed fraction (step / MAX_TURNS) as a feature so the value head and policy
# can reason about how much game is left (endgame play, comeback feasibility) --
# in a finite-horizon MDP the value of a state genuinely depends on time-to-go.
MAX_TURNS = 500.0

# Board holds at most 40 planets + 4 comets = 44 bodies; pad to exactly that.
MAX_PLANETS = 44

# Owner is encoded relative to me, but distinguishing each opponent (27% of games
# are 4-player). Enemies get one channel each, keyed by their player-id offset from
# me ((owner - me) mod MAX_PLAYERS). Using a fixed modulus -- rather than the live
# player count -- keeps the mapping deterministic from (owner, me) alone, so it is
# identical at train and inference time and stable as players are eliminated.
MAX_PLAYERS = 4                 # Orbit Wars is 2- or 4-player
MAX_ENEMIES = MAX_PLAYERS - 1   # at most 3 opponents
OWNER_CH = 2 + MAX_ENEMIES      # is_mine, is_neutral, enemy_1..enemy_3  = 5

# Per-planet STATIC feature layout (instantaneous, this-turn only).
FEATURE_NAMES = [
    "is_mine",      # owner relative to me
    "is_neutral",
    "enemy_1",      # opponents by cyclic player-id offset from me
    "enemy_2",
    "enemy_3",
    "ships",        # garrison, log-scaled
    "production",   # 1..5 -> /5
    "radius",       # /5
    "x",            # (x-50)/50 -> [-1,1], origin at the sun (center)
    "y",            # (y-50)/50 -> [-1,1]
    "dist_center",  # distance from the sun (center) /50
    "is_comet",
    "time_frac",    # global: elapsed fraction of the game (step / MAX_TURNS), in [0,1]
]
STATIC_DIM = len(FEATURE_NAMES)

# Temporal features: a forward-simulated trajectory per planet (forecast.py).
# For each of the next HORIZON turns we store TEMP_CH channels:
#   [is_mine, is_neutral, enemy_1, enemy_2, enemy_3, ships(log-scaled), x, y]
# flattened turn-major after the static block. The x,y trajectory carries motion
# (comets fly across, planets orbit) -- this replaces a static velocity channel,
# and is the only positional signal for comets, whose path is otherwise unseen.
# Lowering HORIZON shrinks the feature vector (and the parsed cache) proportionally.
HORIZON = 20  # most fleet arrivals (median ETA ~5); trims the long p99 tail
              # (~56 turns) to shrink the feature vector & parsed cache
TEMP_CH = OWNER_CH + 3   # 5 owner channels + log-scaled ships + (x, y)
FEATURE_DIM = STATIC_DIM + HORIZON * TEMP_CH

_LOG1000 = math.log(1000.0)


def _owner_channel(owner, me, quad=None):
    """Owner -> relative channel index into [is_mine, is_neutral, enemy_1..3].
    Enemies are keyed by their COUNTER-CLOCKWISE start-quadrant offset from me
    (quad given), so enemy_1 is always the CCW-adjacent opponent, enemy_2 across,
    enemy_3 CW-adjacent -- consistent with the rotation canonicalisation. Falls
    back to raw player-id offset when no quadrant map is available."""
    if owner == me:
        return 0
    if owner == -1:
        return 1
    if quad is not None and owner in quad and me in quad:
        off = (quad[owner] - quad[me]) % MAX_PLAYERS    # CCW quadrant offset 1..3
    else:
        off = (owner - me) % MAX_PLAYERS                # 1 .. MAX_PLAYERS-1
    return 2 + (off - 1)


def _quadrant(x, y):
    """CCW-ordered quadrant of (x, y) about the sun: Q1=0, Q2=1, Q3=2, Q4=3.
    Starts never sit exactly on an axis (per the spawn layout)."""
    dx, dy = x - CENTER[0], y - CENTER[1]
    if dx >= 0 and dy >= 0:
        return 0
    if dx < 0 and dy >= 0:
        return 1
    if dx < 0 and dy < 0:
        return 2
    return 3


def _rot_ccw(xn, yn, k):
    """Rotate centered/normalised coords CCW by k*90 deg (k in 0..3). Works on
    Python floats and numpy arrays alike."""
    if k == 1:
        return -yn, xn
    if k == 2:
        return -xn, -yn
    if k == 3:
        return yn, -xn
    return xn, yn


def compute_canon(obs):
    """Per-game canonicalisation, computed ONCE from the initial board and reused
    every turn (planets orbit, but this is fixed at t=0):
      rot_k: # of 90 deg CCW rotations that send MY start quadrant to Q1.
      quad:  {player_id: start quadrant} for CCW enemy re-keying.
      me:    my player id.
    Pass the result to extract(..., canon=...). Compute it from the FIRST
    observation of the episode (where each player still owns only its home)."""
    me = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    quad = {}
    for p in planets:
        owner = p[1]
        if owner >= 0:
            quad[owner] = _quadrant(p[2], p[3])
    rot_k = (4 - quad.get(me, 0)) % 4                   # send my quadrant -> Q1
    return {"rot_k": rot_k, "quad": quad, "me": me}


def extract(obs, eta_cache=None, canon=None):
    """Return (feats, valid_mask, mine_mask, ids, xs, ys, ships).

    eta_cache: optional per-episode dict forwarded to forecast.simulate to reuse
    fleet arrival trajectories across steps (see simulate). None at play time.
    canon: optional per-game canonicalisation from compute_canon(initial_obs) --
    rotates all positions so MY start quadrant is Q1 and re-keys enemy channels
    by CCW quadrant offset. None = no transform (raw frame, backward compatible).
    """
    rot_k = canon["rot_k"] if canon else 0
    quad = canon["quad"] if canon else None
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    comet_ids = set(
        obs.get("comet_planet_ids", []) if isinstance(obs, dict)
        else getattr(obs, "comet_planet_ids", [])
    )
    # Global elapsed-time fraction, broadcast to every planet (same value). obs
    # carries "step" at train (replays) and inference (live env) alike.
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    time_frac = min(max(float(step) / MAX_TURNS, 0.0), 1.0)

    feats = np.zeros((MAX_PLANETS, FEATURE_DIM), dtype=np.float32)
    valid_mask = np.zeros(MAX_PLANETS, dtype=bool)
    mine_mask = np.zeros(MAX_PLANETS, dtype=bool)
    ids = np.full(MAX_PLANETS, -1, dtype=np.int64)
    xs = np.zeros(MAX_PLANETS, dtype=np.float32)
    ys = np.zeros(MAX_PLANETS, dtype=np.float32)
    ships = np.zeros(MAX_PLANETS, dtype=np.float32)

    # Forward-simulated garrison/owner/position trajectory per planet (baseline:
    # no new launches), keyed by planet id -> (owner_seq, ships_seq, posx, posy).
    traj = forecast.simulate(obs, HORIZON, eta_cache=eta_cache)

    # CCW quadrant offset lookup for the (vectorised) temporal enemy re-keying.
    q_me = quad.get(player, 0) if quad else 0
    q_lookup = None
    if quad:
        q_lookup = np.zeros(MAX_PLAYERS, dtype=np.int64)
        for _pid, _q in quad.items():
            if 0 <= _pid < MAX_PLAYERS:
                q_lookup[_pid] = _q

    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        pid, owner, x, y, radius, n_ships, production = p

        owner_oh = [0.0] * OWNER_CH
        owner_oh[_owner_channel(owner, player, quad)] = 1.0
        xn, yn = _rot_ccw((x - CENTER[0]) / (BOARD / 2.0),
                          (y - CENTER[1]) / (BOARD / 2.0), rot_k)
        feats[i, :STATIC_DIM] = [
            *owner_oh,                                  # is_mine, is_neutral, enemy_1..3
            math.log1p(max(n_ships, 0.0)) / _LOG1000,
            production / 5.0,
            radius / 5.0,
            xn,                                # rotated so my start quadrant is Q1
            yn,
            math.hypot(x - CENTER[0], y - CENTER[1]) / (BOARD / 2.0),  # dist_center (rot-invariant)
            1.0 if pid in comet_ids else 0.0,
            time_frac,                          # global elapsed-time fraction
        ]

        # Temporal block: [HORIZON, TEMP_CH] flattened, relative to `player`.
        seq = traj.get(int(pid))
        if seq is not None:
            owner_seq, ships_seq, posx_seq, posy_seq = seq
            o = np.asarray(owner_seq)
            sh = np.asarray(ships_seq, dtype=np.float32)
            px = np.asarray(posx_seq, dtype=np.float32)
            py = np.asarray(posy_seq, dtype=np.float32)
            present = o != forecast.GONE
            block = feats[i, STATIC_DIM:].reshape(HORIZON, TEMP_CH)
            block[:, 0] = (present & (o == player))                 # is_mine
            block[:, 1] = (present & (o == -1))                     # is_neutral
            # enemies -> their cyclic-offset channel (2..2+MAX_ENEMIES-1)
            enemy = present & (o != player) & (o != -1)
            ek = np.nonzero(enemy)[0]
            if ek.size:
                if q_lookup is not None:
                    off = (q_lookup[o[ek]] - q_me) % MAX_PLAYERS   # CCW quadrant offset
                else:
                    off = (o[ek] - player) % MAX_PLAYERS
                block[ek, 2 + (off - 1)] = 1.0
            block[:, OWNER_CH] = np.where(present, np.log1p(np.maximum(sh, 0.0)) / _LOG1000, 0.0)
            # Position trajectory, same center-based [-1,1] frame as the static x,y,
            # rotated by the same canonical rot_k (0 where GONE, matching present).
            pxn, pyn = _rot_ccw((px - CENTER[0]) / (BOARD / 2.0),
                                (py - CENTER[1]) / (BOARD / 2.0), rot_k)
            block[:, OWNER_CH + 1] = np.where(present, pxn, 0.0)
            block[:, OWNER_CH + 2] = np.where(present, pyn, 0.0)

        valid_mask[i] = True
        mine_mask[i] = (owner == player)
        ids[i] = pid
        xs[i] = x
        ys[i] = y
        ships[i] = n_ships

    return feats, valid_mask, mine_mask, ids, xs, ys, ships

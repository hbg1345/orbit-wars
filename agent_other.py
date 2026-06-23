"""Head-to-head wrapper for the FROZEN other_submission/ agent. Same isolation as
agent_latest.py (see there for why): load other_submission's same-named modules from
its own directory, capture its `agent`, then restore sys.modules / sys.path so the
other agent in the game imports the current repo's versions.

Usage:
    python run_game.py --games 50 main_rl.py agent_other.py
    python run_game.py --games 50 agent_latest.py agent_other.py   # two frozen subs
"""

import importlib.util
import os
import sys

# kaggle_environments execs this file with no __file__, so fall back to cwd (run_game
# runs from the repo root, where other_submission/ lives).
_this_file = globals().get("__file__")
_ROOT = os.path.dirname(os.path.abspath(_this_file)) if _this_file else os.getcwd()
_SUB = os.path.join(_ROOT, "other_submission")
_NAMES = ("physics", "forecast", "features", "model")

_saved_mods = {n: sys.modules.pop(n, None) for n in _NAMES}
_saved_path = list(sys.path)

sys.path.insert(0, _SUB)
try:
    spec = importlib.util.spec_from_file_location(
        "_other_sub_main", os.path.join(_SUB, "main.py"))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _mod
    spec.loader.exec_module(_mod)
    _other_agent = _mod.agent
    try:
        _in = next(iter(_mod._net.state_dict().values())).shape
        print(f"[agent_other] loaded frozen submission from {_SUB} (first param {tuple(_in)})")
    except Exception:
        pass
finally:
    sys.path[:] = _saved_path
    for n, m in _saved_mods.items():
        if m is not None:
            sys.modules[n] = m
        else:
            sys.modules.pop(n, None)


def agent(obs):
    return _other_agent(obs)

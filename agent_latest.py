"""Head-to-head wrapper: run the FROZEN latest_submission/ agent against the current
repo's agent (main_rl.py) in one process, without their same-named modules colliding.

kaggle_environments loads a file agent with exec() in the SHARED interpreter, so the
submission's `import model` / `import features` / ... would otherwise resolve to the
CURRENT repo's modules (found first on sys.path, then cached in sys.modules) -- both
agents would secretly run on the current code. Here we load latest_submission's
modules FROM its own directory under the bare names, capture its `agent`, then restore
sys.modules / sys.path exactly, so the other agent imports the current repo's versions.

Usage:
    python run_game.py --games 50 main_rl.py agent_latest.py
    # P0 = main_rl.py (current model), P1 = agent_latest.py (frozen submission)
"""

import importlib.util
import os
import sys

# kaggle_environments execs this file with no __file__, so fall back to cwd (run_game
# runs from the repo root, where latest_submission/ lives).
_this_file = globals().get("__file__")
_ROOT = os.path.dirname(os.path.abspath(_this_file)) if _this_file else os.getcwd()
_SUB = os.path.join(_ROOT, "latest_submission")
# Every module the submission's import graph touches (main -> features/model/forecast,
# model -> features, features -> forecast, forecast -> physics).
_NAMES = ("physics", "forecast", "features", "model")

# 1) stash and clear the current-repo versions so the submission load can't reuse them
_saved_mods = {n: sys.modules.pop(n, None) for n in _NAMES}
_saved_path = list(sys.path)

# 2) load the submission FROM its own dir (front of sys.path -> its line-13 `import
#    features` etc. resolve to the submission, not the repo)
sys.path.insert(0, _SUB)
try:
    spec = importlib.util.spec_from_file_location(
        "_latest_sub_main", os.path.join(_SUB, "main.py"))
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _mod          # so the loader sees a registered module
    spec.loader.exec_module(_mod)
    _latest_agent = _mod.agent
    try:
        _in = next(iter(_mod._net.state_dict().values())).shape
        print(f"[agent_latest] loaded frozen submission from {_SUB} (first param {tuple(_in)})")
    except Exception:
        pass
finally:
    # 3) restore global import state EXACTLY -- the submission's agent closed over its
    #    own modules already, so the current-repo agent now imports its own fresh.
    sys.path[:] = _saved_path
    for n, m in _saved_mods.items():
        if m is not None:
            sys.modules[n] = m
        else:
            sys.modules.pop(n, None)


def agent(obs):
    return _latest_agent(obs)

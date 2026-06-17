"""Distribution of fleet travel time (ETA, in turns) and HORIZON coverage.

The temporal features (features.py) only roll the board forward HORIZON turns, so
the forecast "sees" a fleet's arrival only once its remaining ETA <= HORIZON. A
fleet's *total* travel time is fixed at launch, so this measures, per fleet, how
many turns it takes to land (forecast._fleet_eta, traced from the turn the fleet
first appears = its launch). The coverage panel then shows what fraction of
launches are fully covered for HORIZON = 5, 10, 15, ... -- i.e. how much of the
arrival distribution each horizon captures -- to justify the HORIZON choice.

  python horizon_dist.py            # reads data/zips/*.zip (+ data/*.json) -> horizon_dist.png
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import json
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forecast
import physics as P
from features import HORIZON

N_EPISODES = 300
STEP = 5  # report coverage at HORIZON = 5, 10, 15, ...


def iter_episodes(limit):
    """Yield parsed episode dicts from loose data/*.json and data/zips/*.zip
    members (same sources as dataset.py), stopping after `limit` episodes."""
    n = 0
    for path in sorted(glob.glob("data/*.json")):
        try:
            yield json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        n += 1
        if n >= limit:
            return
    for zp in sorted(glob.glob("data/zips/*.zip")):
        try:
            zf = zipfile.ZipFile(zp)
        except Exception:
            continue
        for member in zf.namelist():
            if not member.lower().endswith(".json"):
                continue
            try:
                yield json.load(zf.open(member))
            except Exception:
                continue
            n += 1
            if n >= limit:
                return


def _global_obs(step_entry):
    """Pick a player's observation that carries the (global) fleet/planet list."""
    for p in range(len(step_entry)):
        o = step_entry[p].get("observation", {})
        if o.get("planets"):
            return o
    return None


etas = []
for d in iter_episodes(N_EPISODES):
    steps = d["steps"]
    prev_ids = set()
    for t in range(len(steps)):
        obs = _global_obs(steps[t])
        if obs is None:
            continue
        # Trace only fleets seen for the first time -- that is the launch turn, so
        # the traced ETA is the fleet's full travel time, counted once.
        new = [f for f in (obs.get("fleets") or []) if int(f[0]) not in prev_ids]
        if new:
            at = forecast._track(P.build_world(obs))
            for f in new:
                tid, k = forecast._fleet_eta(f, at, forecast.ETA_SEARCH)
                if tid is not None:
                    etas.append(k)
        prev_ids = {int(f[0]) for f in (obs.get("fleets") or [])}

eta = np.array(etas)
if len(eta) == 0:
    raise SystemExit("no fleet ETAs found -- is data/*.json present?")

print(f"{len(eta)} fleet launches traced to an arrival")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.percentile(eta, q):.0f} turns")
print(f"  mean: {eta.mean():.1f}  max: {eta.max():.0f}")

# Coverage at HORIZON = 5, 10, 15, ... up to just past the p99.5 tail.
h_max = int(np.ceil(np.percentile(eta, 99.5) / STEP) * STEP)
hs = np.arange(STEP, h_max + 1, STEP)
cover = np.array([100.0 * (eta <= h).mean() for h in hs])
print("\ncoverage by HORIZON:")
for h, c in zip(hs, cover):
    mark = "  <- current" if h == HORIZON else ""
    print(f"  H={h:>3}: {c:5.1f}%{mark}")

fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 5))

# Left: ETA histogram.
axh.hist(eta, bins=np.arange(0, eta.max() + 2) - 0.5, color="#4C72B0")
axh.axvline(HORIZON, color="#C44E52", lw=2, label=f"HORIZON={HORIZON}")
axh.set_xlabel("fleet travel time (turns to arrival)")
axh.set_ylabel("# launches")
axh.set_title(f"fleet arrival ETA  (n={len(eta)}, median={np.median(eta):.0f})")
axh.set_xlim(0, min(eta.max(), h_max) + 1)
axh.legend()

# Right: coverage % vs HORIZON, in steps of STEP.
bars = axc.bar(hs, cover, width=STEP * 0.8, color="#55A868")
for h, c in zip(hs, cover):
    axc.text(h, c + 1, f"{c:.0f}", ha="center", va="bottom", fontsize=8)
if HORIZON in hs:
    bars[list(hs).index(HORIZON)].set_color("#C44E52")
axc.axhline(100, color="grey", ls=":", lw=1)
axc.set_xlabel(f"HORIZON (turns ahead, step {STEP})")
axc.set_ylabel("% of launches covered (ETA <= HORIZON)")
axc.set_title("arrival coverage by horizon")
axc.set_xticks(hs)
axc.set_ylim(0, 105)

fig.tight_layout()
fig.savefig("reports/horizon_dist.png", dpi=110)
print("\nsaved horizon_dist.png")

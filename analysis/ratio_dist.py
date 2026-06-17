"""Distribution of ships_sent / source_garrison for winner launches.

If option (c) holds, the heavy 4-decade absolute-count tail should collapse: big
"dump everything" launches pile up near ratio 1, small tactical sends spread at
low ratios -- a bounded [0,1] target that's far easier to model than 1..24000.
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ratios = []
over1 = 0
for path in sorted(glob.glob("data/*.json"))[:300]:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    steps, rewards = d["steps"], d["rewards"]
    winner = int(np.argmax(rewards))
    for t in range(1, len(steps)):
        prev = steps[t - 1][winner].get("observation", {})
        planets = prev.get("planets")
        if not planets:
            continue
        gar = {int(p[0]): float(p[5]) for p in planets}
        for mv in (steps[t][winner].get("action") or []):
            fid, ships = int(mv[0]), float(mv[2])
            g = gar.get(fid)
            if g and g > 0:
                r = ships / g
                if r > 1.0:
                    over1 += 1
                ratios.append(min(r, 1.0))

r = np.array(ratios)
print(f"{len(r)} launches with known source garrison")
print(f"  ratio>1 (clamped): {over1} ({100*over1/len(r):.2f}%)")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q}: {np.percentile(r, q):.2f}")
print(f"  fraction sending >=90% of garrison: {100*(r >= 0.9).mean():.1f}%")
print(f"  fraction sending <=10% of garrison: {100*(r <= 0.1).mean():.1f}%")

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(r, bins=np.linspace(0, 1, 51), color="#4C72B0")
ax.set_xlabel("ships sent / source garrison  (ratio)")
ax.set_ylabel("# launches")
ax.set_title(f"launch size as a fraction of available ships  (n={len(r)}, median={np.median(r):.2f})")
fig.tight_layout()
fig.savefig("reports/ratio_dist.png", dpi=110)
print("saved ratio_dist.png")
"""Full fleet-size distribution including the >60 tail (winner launches)."""
import glob
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ship = Counter()
for path in sorted(glob.glob("data/*.json"))[:300]:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    steps, rewards = d["steps"], d["rewards"]
    winner = int(np.argmax(rewards))
    for t in range(len(steps)):
        for mv in (steps[t][winner].get("action") or []):
            ship[int(mv[2])] += 1

sizes = np.repeat(list(ship.keys()), list(ship.values()))
total = len(sizes)
print(f"{total} launches")
for thr in (60, 100, 200, 500, 1000):
    print(f"  ships > {thr:>4}: {100*(sizes > thr).mean():.2f}%")
for q in (50, 90, 95, 99, 99.9):
    print(f"  p{q}: {np.percentile(sizes, q):.0f}")
print(f"  max: {sizes.max()}")

bins = np.logspace(0, np.log10(sizes.max() + 1), 70)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# log x, linear y -- shape across the whole range
ax1.hist(sizes, bins=bins, color="#55A868")
ax1.set_xscale("log")
ax1.set_xlabel("fleet size (ships, log scale)")
ax1.set_ylabel("# launches")
ax1.set_title("fleet-size distribution (log x)")

# log x AND log y -- reveals the heavy-tail decay
ax2.hist(sizes, bins=bins, color="#C44E52")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("fleet size (ships, log scale)")
ax2.set_ylabel("# launches (log)")
ax2.set_title("same, log y (tail decay)")

for ax in (ax1, ax2):
    for q, c in [(50, "#333"), (90, "#1f77b4"), (99, "#9467bd")]:
        v = np.percentile(sizes, q)
        ax.axvline(v, color=c, ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1], f" p{q}={v:.0f}", color=c, fontsize=8,
                rotation=90, va="top", ha="left")

fig.suptitle(f"winner fleet-size distribution  (n={total}, median={np.median(sizes):.0f}, max={sizes.max()})")
fig.tight_layout()
fig.savefig("ship_tail.png", dpi=110)
print("saved ship_tail.png")
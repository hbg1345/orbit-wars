"""Plot the RL training curves logged by train_rl.py (train_rl_log.csv).

Three panels:
  1. win rates  -- league win, 2p, 4p, and vsSL (the fixed-opponent yardstick)
  2. losses     -- policy and value
  3. entropy    -- policy entropy (exploration)

Usage:
  python plot_rl_log.py                 # reads train_rl_log.csv -> train_rl_curves.png
  python plot_rl_log.py other_log.csv   # a different log
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")          # save to file without a display
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "train_rl_log.csv"
OUT = "train_rl_curves.png"


def series(rows, name):
    """(iters, values) for one column, skipping blank cells."""
    xs, ys = [], []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "None", None):
            xs.append(int(r["iter"]))
            ys.append(float(v))
    return xs, ys


def main():
    with open(LOG, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{LOG} is empty")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: win rates. vsSL is the absolute progress signal; the league rates
    # drift toward parity as the pool strengthens, so mark their baselines.
    ax = axes[0]
    for col, label in [("win", "league win"), ("win2", "2p"),
                       ("win4", "4p"), ("vssl", "vs SL")]:
        xs, ys = series(rows, col)
        if xs:
            ax.plot(xs, ys, marker=".", label=label)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)   # 2p / vsSL baseline
    ax.axhline(0.25, color="gray", ls=":", alpha=0.3)  # 4p baseline
    ax.set_ylim(0, 1)
    ax.set_title("win rates (vs SL = absolute)")
    ax.set_xlabel("iter"); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 2: losses.
    ax = axes[1]
    for col in ("policy", "value"):
        xs, ys = series(rows, col)
        if xs:
            ax.plot(xs, ys, marker=".", label=col)
    ax.axhline(0.0, color="gray", ls=":", alpha=0.5)
    ax.set_title("losses")
    ax.set_xlabel("iter"); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3: entropy (exploration).
    ax = axes[2]
    xs, ys = series(rows, "entropy")
    if xs:
        ax.plot(xs, ys, marker=".", color="tab:green")
    ax.set_title("policy entropy")
    ax.set_xlabel("iter"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}  ({len(rows)} iters)")


if __name__ == "__main__":
    main()

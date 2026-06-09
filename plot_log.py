"""Plot the training curves logged by train_sl.py (train_log.csv).

Draws train vs val for each metric (total loss, target cross-entropy, ship MSE
in log space) and marks the best-val-loss epoch (the one saved to orbit_net.pt).

Usage:
  python plot_log.py                 # reads train_log.csv -> train_curves.png
  python plot_log.py other_log.csv   # a different log
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")          # save to file without a display
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "train_log.csv"
OUT = "train_curves.png"


def series(rows, name):
    """(epochs, values) for one column, skipping blank cells (e.g. no val)."""
    xs, ys = [], []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "None", None):
            xs.append(int(r["epoch"]))
            ys.append(float(v))
    return xs, ys


def main():
    with open(LOG, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{LOG} is empty")
        return

    panels = [
        ("total loss", "train_loss", "val_loss"),
        ("target cross-entropy", "train_ce", "val_ce"),
        ("ship MSE (log1p space)", "train_ship", "val_ship"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Best epoch by val loss (fallback: train loss) to mark on every panel.
    bx, bvals = series(rows, "val_loss")
    if not bx:
        bx, bvals = series(rows, "train_loss")
    best_ep = bx[min(range(len(bvals)), key=lambda i: bvals[i])] if bx else None

    for ax, (title, tr, va) in zip(axes, panels):
        ax.plot(*series(rows, tr), label="train", marker=".")
        vx, vy = series(rows, va)
        if vx:
            ax.plot(vx, vy, label="val", marker=".")
        if best_ep is not None:
            ax.axvline(best_ep, color="gray", ls="--", alpha=0.6,
                       label=f"best @ {best_ep}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}" + (f"  (best val epoch = {best_ep})" if best_ep else ""))


if __name__ == "__main__":
    main()
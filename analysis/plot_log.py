"""Plot the training curves logged by train_sl.py (train_log.csv).

Draws train vs val for each metric (total loss, gate BCE, target cross-entropy,
ship CE over the fraction bins, and the ship-fraction MSE diagnostic) and marks
the best-val-loss epoch (saved to orbit_net.pt).

Usage:
  python plot_log.py                 # reads train_log.csv -> train_curves.png
  python plot_log.py other_log.csv   # a different log
  python plot_log.py --watch         # live auto-refreshing window (tails the csv)
  python plot_log.py --watch --interval 2000
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv

# Parse args BEFORE importing pyplot: the backend (interactive vs file) depends on
# --watch and must be chosen before matplotlib.pyplot is imported.
ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default="reports/train_log.csv")
ap.add_argument("--watch", action="store_true", help="live auto-refreshing window")
ap.add_argument("--interval", type=float, default=5.0, help="watch refresh period (seconds)")
args = ap.parse_args()

import matplotlib
matplotlib.use("TkAgg" if args.watch else "Agg")
import matplotlib.pyplot as plt

OUT = "reports/train_curves.png"

PANELS = [
    ("total loss", "train_loss", "val_loss"),
    ("gate BCE", "train_gate", "val_gate"),
    ("target cross-entropy", "train_ce", "val_ce"),
    ("ship CE (fraction bins)", "train_ship", "val_ship"),
    ("ship MSE (diagnostic)", "train_mse", "val_mse"),
]


def series(rows, name):
    """(epochs, values) for one column, skipping blank cells (e.g. no val)."""
    xs, ys = [], []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "None", None):
            try:
                ys.append(float(v))
                xs.append(int(r["epoch"]))
            except (ValueError, KeyError):
                continue
    return xs, ys


def read_rows(path):
    """All rows of the CSV as dicts; [] if missing/unreadable (e.g. mid-write)."""
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, OSError):
        return []


def draw(axes, rows):
    """Redraw every panel from `rows` onto the given flat list of axes."""
    # Best epoch by val loss (fallback: train loss) to mark on every panel.
    bx, bvals = series(rows, "val_loss")
    if not bx:
        bx, bvals = series(rows, "train_loss")
    best_ep = bx[min(range(len(bvals)), key=lambda i: bvals[i])] if bx else None

    for ax, (title, tr, va) in zip(axes, PANELS):
        ax.clear()
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
    return best_ep


def main():
    if args.watch:
        import matplotlib.animation as animation
        # 2x3 grid reads better in a window than one wide row; hide the unused cell.
        fig, axgrid = plt.subplots(2, 3, figsize=(16, 8))
        axes = axgrid.flatten()
        axes[len(PANELS)].axis("off")

        def update(_frame):
            rows = read_rows(args.log)
            if not rows:
                return
            draw(axes[:len(PANELS)], rows)
            fig.suptitle(f"{args.log}  ({len(rows)} epochs)")

        # Keep a reference (FuncAnimation is GC'd if unbound) -- plt.show() blocks.
        _ani = animation.FuncAnimation(fig, update, interval=args.interval * 1000)
        # Explicit margins (no tight_layout -- it would just fight these each frame).
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.08,
                            hspace=0.35, wspace=0.3)
        plt.show()
        return

    rows = read_rows(args.log)
    if not rows:
        print(f"{args.log} is empty or missing")
        return
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.3 * len(PANELS), 4.5))
    best_ep = draw(axes, rows)
    fig.tight_layout(pad=2.0, w_pad=3.5)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}" + (f"  (best val epoch = {best_ep})" if best_ep else ""))


if __name__ == "__main__":
    main()

"""Live RL dashboard for train_rl_v2.py.

Run alongside training; it tails two files the trainer writes each iter and
auto-refreshes a 2x2 window:
  1. ship-fraction histogram   (launch_dist.jsonl -- the SAMPLED launch fractions)
  2. win rates                 (train 2p/4p + eval gate-2p/4p, from train_rl_log.csv)
  3. losses                    (policy, value)
  4. objective (mean return)

  python watch_launch_dist.py
  python watch_launch_dist.py --file launch_dist.jsonl --csv train_rl_log.csv --interval 1000

The window updates every --interval ms; close it to stop (training is unaffected).
"""

import os as _os, sys as _sys  # noqa: E402  (repo-root shim, added by layout move)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("TkAgg")          # interactive, auto-refreshing window (NOT Agg/file)
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def read_last(path):
    """Return the last COMPLETE json record in the JSONL file, or None. A partial
    final line (trainer mid-write) is skipped, so we never crash on a torn line."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn last line being written; keep the previous
    return last


def read_rows(path):
    """All rows of the CSV log as dicts, or [] if missing/unreadable."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def series(rows, name):
    """(iters, values) for one column, skipping blank/None cells."""
    xs, ys = [], []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "None", None):
            try:
                ys.append(float(v))
                xs.append(int(r["iter"]))
            except (ValueError, KeyError):
                continue
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="launch_dist.jsonl")
    ap.add_argument("--eval-file", default="launch_dist_eval.jsonl")
    ap.add_argument("--csv", default="reports/train_rl_log.csv")
    ap.add_argument("--interval", type=int, default=10*1000, help="refresh period (ms)")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax_hist, ax_win = axes[0]
    ax_loss, ax_ret = axes[1]

    def _bars(rec, color, label):
        """Overlay one normalised ship-fraction histogram (share of launches, so
        train vs eval are comparable despite different launch counts)."""
        if rec is None or not rec.get("n"):
            return None
        counts = np.asarray(rec["counts"], dtype=float)
        total = counts.sum()
        if total <= 0:
            return None
        bins = int(rec.get("bins", len(counts)))
        edges = np.linspace(0.0, 1.0, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_hist.bar(centers, counts / total, width=1.0 / bins, align="center",
                    color=color, alpha=0.55, edgecolor="none",
                    label=f"{label} (it{rec['iter']}, n={rec['n']}, mean={rec.get('mean', 0):.2f})")
        return rec.get("mean")

    def draw_hist():
        ax_hist.clear()
        ax_hist.set_xlabel("ship fraction sent  [0, 1]")
        ax_hist.set_ylabel("share of launches")
        ax_hist.set_xlim(0.0, 1.0)
        train = read_last(args.file)
        ev = read_last(args.eval_file)
        m_tr = _bars(train, "steelblue", "train (sampled)")
        m_ev = _bars(ev, "darkorange", "eval (greedy)")
        for m, c in ((m_tr, "steelblue"), (m_ev, "darkorange")):
            if m is not None:
                ax_hist.axvline(m, color=c, lw=1.3, ls="--", alpha=0.9)
        if train is None and ev is None:
            ax_hist.set_title(f"waiting for {args.file} ...")
        else:
            ax_hist.set_title("ship fraction: train (sampled) vs eval (greedy)")
            ax_hist.legend(loc="upper center", fontsize=8)

    def draw_lines(ax, rows, cols, title, ylabel, baselines=(), baseline_color="gray"):
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("iter"); ax.set_ylabel(ylabel)
        for col, label in cols:
            xs, ys = series(rows, col)
            if xs:
                ax.plot(xs, ys, marker=".", ms=4, label=label)
        for y in baselines:
            ax.axhline(y, color=baseline_color, ls=":", alpha=0.7)
        # iter is integer-valued: force whole-number ticks (no 0.5 steps when the
        # log has only a few rows early on).
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)

    def update(_frame):
        draw_hist()
        rows = read_rows(args.csv)
        draw_lines(ax_win, rows,
                   [("tr_win2", "train 2p"), ("tr_win4", "train 4p"),
                    ("win2", "eval gate(2p)"), ("win4", "eval 4p")],
                   "win rates", "win rate", baselines=(0.5, 0.25),
                   baseline_color="red")
        ax_win.set_ylim(0, 1)
        draw_lines(ax_loss, rows,
                   [("policy", "policy"), ("value", "value")],
                   "losses", "loss", baselines=(0.0,))
        draw_lines(ax_ret, rows,
                   [("mean_ret", "mean return")],
                   "objective: mean return (should rise)", "return")

    # Keep a reference (FuncAnimation is GC'd if unbound) -- plt.show() blocks anyway.
    _ani = animation.FuncAnimation(fig, update, interval=args.interval)
    # Wider gaps between panels so titles/labels don't crowd each other.
    fig.tight_layout(pad=2.5, h_pad=4.0, w_pad=4.0)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()

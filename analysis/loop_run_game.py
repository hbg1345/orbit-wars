"""Run run_game.py as N SEPARATE processes (one game each) and tally P0's result.

This reproduces "running run_game by hand N times" -- each invocation is a fresh
process with a fresh map (the env draws a new random seed per episode), so it's the
honest one-by-one baseline to compare against `run_game.py --games N` (which does the
same draws inside one process). If the two disagree by more than sampling noise,
something is wrong; if they agree, the difference you saw was just small-sample
variance.

Usage:
  python analysis/loop_run_game.py                         # 100x main_rl vs main_snap1
  python analysis/loop_run_game.py 100 main_rl.py main_snap1.py
  python analysis/loop_run_game.py 50  main_snap5.py main_snap0.py main_snap1.py main_snap2.py
"""
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
agents = sys.argv[2:] or ["main_rl.py", "main_snap1.py"]

tally = collections.Counter()
pat = re.compile(r"P0 (WIN|LOSS|DRAW)")

print(f"P0 = {agents[0]}  vs  {agents[1:]}   ({N} separate games)\n")
for i in range(1, N + 1):
    out = subprocess.run(
        [sys.executable, "run_game.py", "--games", "1", *agents],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    m = pat.search(out)
    res = m.group(1) if m else "?"
    tally[res] += 1
    w, done = tally["WIN"], i
    print(f"[{done:3d}/{N}] {res:4s}  running P0 win%={100*w/done:5.1f}")

w, l, d, unk = tally["WIN"], tally["LOSS"], tally["DRAW"], tally["?"]
print(f"\nP0={agents[0]} vs {agents[1:]}")
print(f"  WIN {w}  LOSS {l}  DRAW {d}" + (f"  UNPARSED {unk}" if unk else ""))
print(f"  P0 win% = {100*w/N:.1f}%   (win/total; draws counted as non-win)")
if (w + l):
    print(f"  P0 win% excl. draws = {100*w/(w+l):.1f}%")

"""Render a downloaded Orbit Wars replay JSON to an HTML you can open.

Usage:
  python render_replay.py data/<episode>.json [out.html]
"""

import json
import logging
import sys

logging.disable(logging.WARNING)
from kaggle_environments import make

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "replay_view.html"

with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

env = make("orbit_wars",
           configuration=d.get("configuration", {}),
           info=d.get("info", {}),
           steps=d.get("steps", []))
html = env.render(mode="html", width=900, height=700)
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out} from {path} ({len(d.get('steps', []))} steps)")

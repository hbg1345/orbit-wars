"""Custom replay viewer that highlights the fleets the parser marks as MISSED.

For one episode, we replay the winner's launches through replay._resolve_target.
Fleets that resolve to a planet are drawn green; fleets that resolve to None
("missed" -> sun / out of bounds / still flying) are drawn RED. Watch whether
the red fleets actually fly into empty space (true miss) or graze a planet that
the parser failed to credit.

Usage:
  python render_misses.py data/<episode>.json [out.html]
"""

import json
import sys

import numpy as np

import replay
import features as F

path = sys.argv[1] if len(sys.argv) > 1 else "data/78667187.json"
out = sys.argv[2] if len(sys.argv) > 2 else "misses_view.html"

d = json.load(open(path, encoding="utf-8"))
S = d["steps"]
fleet_idx = replay._index_fleets(S)
W = int(np.argmax(d["rewards"]))

# Classify every fleet the winner launched: hit (resolved) vs missed (None).
missed, hit = set(), set()
for t in range(len(S)):
    obs = S[t][W]["observation"]
    if not obs.get("planets"):
        continue
    ids = {int(p[0]) for p in obs["planets"][:F.MAX_PLANETS]}
    prev = fleet_idx[t - 1].keys() if t > 0 else set()
    for mv in (S[t][W].get("action") or []):
        fid, ang, nsh = int(mv[0]), float(mv[1]), int(mv[2])
        if fid not in ids:
            continue
        new = [k for k, f in fleet_idx[t].items()
               if k not in prev and f[5] == fid and f[6] == nsh and abs(f[4] - ang) < 1e-6]
        if not new:
            continue
        flid = new[0]
        tid = replay._resolve_target(S, fleet_idx, t, fid, ang, nsh)
        (hit if tid is not None else missed).add(flid)

# Compact per-step data: planets [owner,x,y,r], fleets [id,owner,x,y].
frames = []
for t in range(len(S)):
    obs = S[t][0]["observation"]
    planets = [[int(p[1]), round(p[2], 1), round(p[3], 1), round(p[4], 2)]
               for p in obs.get("planets", [])]
    fleets = [[int(f[0]), int(f[1]), round(f[2], 1), round(f[3], 1)]
              for f in obs.get("fleets", [])]
    frames.append({"p": planets, "f": fleets})

data = {"W": W, "missed": sorted(missed), "hit": sorted(hit), "frames": frames}

HTML = """<!doctype html><meta charset=utf-8>
<style>body{background:#111;color:#ddd;font-family:sans-serif;text-align:center}
canvas{background:#000;border:1px solid #333}#bar{margin:8px}</style>
<div id=bar>
 <button id=play>pause</button>
 <input id=slider type=range min=0 value=0 style=width:500px>
 <span id=lbl></span>
</div>
<canvas id=c width=700 height=700></canvas>
<div>RED = parser says MISSED &nbsp; GREEN = winner hit &nbsp; GRAY = enemy fleet</div>
<script>
const D=__DATA__, F=D.frames, missed=new Set(D.missed), hit=new Set(D.hit), W=D.W;
const c=document.getElementById('c'), x=c.getContext('2d'), SC=7;
let t=0, playing=true;
function P(v){return v*SC;}
function draw(){
 x.clearRect(0,0,700,700);
 x.fillStyle='#f5d000'; x.beginPath(); x.arc(P(50),P(50),P(10),0,7); x.fill(); // sun
 const fr=F[t];
 for(const p of fr.p){const[ow,px,py,r]=p;
   x.fillStyle = ow===W?'#3a6':ow===-1?'#888':'#a63';
   x.beginPath(); x.arc(P(px),P(py),Math.max(P(r),2),0,7); x.fill();}
 for(const f of fr.f){const[id,ow,fx,fy]=f;
   let col='#666';
   if(ow===W) col = missed.has(id)?'#f33':hit.has(id)?'#3f6':'#39f';
   x.fillStyle=col; x.beginPath(); x.arc(P(fx),P(fy),3,0,7); x.fill();}
 document.getElementById('lbl').textContent='step '+t+'/'+(F.length-1);
 document.getElementById('slider').value=t;
}
document.getElementById('slider').max=F.length-1;
document.getElementById('slider').oninput=e=>{t=+e.target.value;draw();};
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'pause':'play';};
function loop(){ if(playing){t=(t+1)%F.length; draw();} setTimeout(loop,120);}
draw(); loop();
</script>"""

html = HTML.replace("__DATA__", json.dumps(data))
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out}: {len(missed)} missed fleets, {len(hit)} hit, {len(frames)} steps")

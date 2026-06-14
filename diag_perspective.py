"""Perspective diagnostic: does the gate fire for BOTH player perspectives?

The board in each logged observation is GLOBAL (absolute owners/positions); only
`player` differs per agent. So we replay each logged step twice -- once as
player 0, once as player 1 -- run the net, and count how many owned planets the
gate would fire on. If one perspective never fires while the other does, the bug
is in perspective handling (compute_canon / extract / the model), independent of
any live shared-state. If both fire fine here, the live bug is shared state.

Run in the orbit env:  python diag_perspective.py
"""
import json
import torch
import features as F
from model import OrbitNet

GATE_THRESHOLD = 0.5

net = OrbitNet()
ckpt = torch.load("orbit_net.pt", map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
net.load_state_dict(state, strict=False)
net.eval()

steps = [json.loads(l) for l in open("obs_log.jsonl", encoding="utf-8")]
obs0 = steps[0]
print("home quadrants:",
      {p[1]: F._quadrant(p[2], p[3]) for p in obs0["planets"] if p[1] >= 0})

# Canon is per-episode, computed once from step 0 for each perspective.
canon = {pl: F.compute_canon({**obs0, "player": pl}) for pl in (0, 1)}
print("rot_k:", {pl: c["rot_k"] for pl, c in canon.items()})

fires = {0: 0, 1: 0}
owned_steps = {0: 0, 1: 0}     # steps where this player owned >=1 planet
max_gp = {0: 0.0, 1: 0.0}
for s in steps:
    for pl in (0, 1):
        obs = {**s, "player": pl}
        feats, valid, mine, ids, xs, ys, ships = F.extract(obs, canon=canon[pl])
        with torch.no_grad():
            dbg = net.act_debug(feats, valid, mine, gate_threshold=GATE_THRESHOLD)
        if not dbg:
            continue
        owned_steps[pl] += 1
        gps = [v[0] for v in dbg.values()]
        max_gp[pl] = max(max_gp[pl], max(gps))
        fires[pl] += sum(1 for v in dbg.values() if v[4])

for pl in (0, 1):
    print(f"player={pl}: owned_steps={owned_steps[pl]} "
          f"total_gate_fires={fires[pl]} max_gate_p={max_gp[pl]:.3f}")

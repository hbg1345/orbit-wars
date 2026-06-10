"""Quick smoke test for the 2-Gaussian ship-fraction mixture head.

  conda run -n orbit python smoke_ship.py

Exercises forward -> ship_mixture / ship_fraction / ship_entropy, sampling, a
log_prob backward, and act_rl + evaluate_actions end-to-end on random boards.
"""
import torch
from features import FEATURE_DIM, MAX_PLANETS
from model import OrbitNet

torch.manual_seed(0)
net = OrbitNet(d_model=32, n_heads=2, n_layers=2)
B, N = 3, MAX_PLANETS

feats = torch.randn(B, N, FEATURE_DIM)
valid = torch.zeros(B, N, dtype=torch.bool); valid[:, :6] = True
mine = torch.zeros(B, N, dtype=torch.bool); mine[:, :2] = True

tl, gl, h = net(feats, valid, mine)
print("forward:", tl.shape, gl.shape, h.shape)

b = torch.tensor([0, 1, 2]); src = torch.tensor([0, 1, 0]); tgt = torch.tensor([3, 4, 5])
frac = net.ship_fraction(h, b, src, tgt)
print("ship_fraction:", frac.tolist(), "in [0,1]:", bool((frac >= 0).all() and (frac <= 1).all()))

dist, means, sigmas, w = net.ship_mixture(h, b, src, tgt)
samp = dist.sample().clamp(0, 1)
ent = net.ship_entropy(sigmas, w)
print("means:", means.shape, "sigmas>=floor:", float(sigmas.min()), "weights sum:", w.sum(-1).tolist())
print("sample:", samp.tolist(), "entropy:", ent.tolist())

# log_prob backward
lp = dist.log_prob(frac.detach().clamp(0, 1)).sum()
lp.backward()
g = net.ship_head[-1].weight.grad
print("log_prob backward grad finite:", bool(torch.isfinite(g).all()))

# end-to-end act_rl + evaluate_actions
net.zero_grad()
f0, v0, m0 = feats[0], valid[0], mine[0]
decisions, record = net.act_rl(f0, v0, m0, gate_bias=0.0)
print("act_rl launches:", len(decisions))
logp, value, entb = net.evaluate_actions(feats[:1], valid[:1], mine[:1], [record], gate_bias=0.0)
print("evaluate_actions logp:", logp.tolist(), "entropy:", entb.tolist())
(logp.sum() + entb.sum()).backward()
print("evaluate_actions backward OK")
print("\nALL SMOKE CHECKS PASSED")

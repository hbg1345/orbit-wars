"""Benchmark batch=1 inference (the deployment path) on CPU vs CUDA.

Replicates model.act()'s compute but device-aware, including the per-step
host->device copy of the inputs (that transfer overhead is exactly what makes
GPU questionable for single-sample inference).
"""
import time
import numpy as np
import torch

from features import FEATURE_DIM, MAX_PLANETS
from model import OrbitNet

N = MAX_PLANETS
GATE_THRESHOLD = 0.6
ITERS = 1000
WARMUP = 0

# Realistic-shaped random input (values don't matter for timing).
rng = np.random.default_rng(0)
feats_np = rng.standard_normal((N, FEATURE_DIM)).astype(np.float32)
valid_np = np.ones(N, dtype=bool)
mine_np = np.zeros(N, dtype=bool)
mine_np[:max(1, N // 4)] = True   # ~quarter of bodies are mine


@torch.no_grad()
def act_on(net, device):
    # mirror model.act(greedy=True) but place inputs on `device`
    ft = torch.as_tensor(feats_np, dtype=torch.float32, device=device).unsqueeze(0)
    vm = torch.as_tensor(valid_np, dtype=torch.bool, device=device).unsqueeze(0)
    mm = torch.as_tensor(mine_np, dtype=torch.bool, device=device).unsqueeze(0)
    target_logits, gate_logits, h = net(ft, vm, mm)
    mine_idx = torch.nonzero(mm[0], as_tuple=False).flatten()
    gate_p = torch.sigmoid(gate_logits[0, mine_idx])
    fire = gate_p >= GATE_THRESHOLD
    L = int(fire.sum().item())
    if L > 0:
        src_l = mine_idx[fire]
        tgt_l = target_logits[0, src_l].argmax(dim=-1)
        b0 = torch.zeros(L, dtype=torch.long, device=device)
        s_logits = net.ship_logits(h, b0, src_l, tgt_l)
        _ = s_logits.argmax(dim=-1).tolist()  # force sync / readback like act()


def bench(device):
    net = OrbitNet().to(device).eval()
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    for _ in range(WARMUP):
        act_on(net, device)
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        act_on(net, device)
    sync()
    dt = (time.perf_counter() - t0) / ITERS * 1e3   # ms / step
    return dt


print(f"FEATURE_DIM={FEATURE_DIM}  N(bodies)={N}  iters={ITERS}")
cpu_ms = bench("cpu")
print(f"CPU : {cpu_ms:8.3f} ms/step")
if torch.cuda.is_available():
    gpu_ms = bench("cuda")
    print(f"CUDA: {gpu_ms:8.3f} ms/step  ({torch.cuda.get_device_name(0)})")
    faster = "GPU" if gpu_ms < cpu_ms else "CPU"
    print(f"-> {faster} faster by {max(cpu_ms, gpu_ms) / min(cpu_ms, gpu_ms):.2f}x")
else:
    print("CUDA: not available")

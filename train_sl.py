"""Supervised training for OrbitNet -- pure training only.

Data parsing/caching lives in dataset.py. Build the cache there first, then this
script just loads it and trains, saving the best-val-loss weights to orbit_net.pt
and the latest-epoch weights to orbit_net_last.pt (every epoch).

  python dataset.py            # 1) parse replays -> on-disk cache (once)
  python train_sl.py           # 2) train on the cached dataset (fresh)
  python train_sl.py --resume  # 2') continue from orbit_net.pt
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import OrbitNet
from dataset import load_cached

EPOCHS = 300
LR = 5e-4
WEIGHT_DECAY = 5e-5       # L2 regularization (cf. supervised.py)
SHIP_LOSS_WEIGHT = 0.5    # ship loss is the NLL of the human fraction under the
                          # 2-Gaussian mixture head, so SL fits the full (bimodal)
                          # launch-fraction distribution, not just its mean. NLL is
                          # unbounded below (can go negative for a sharp fit), so
                          # tune this down if the ship term dominates gate/target.
# Launches are only ~16% of owned-planet decisions, so up-weight the positive
# class in the gate BCE (~ n_neg / n_pos = 84.3 / 15.7). This both pushes launch
# probabilities up (so a ~0.5 threshold works) and gives the rare launch class
# more gradient, sharpening the gate. Set to 1.0 to disable.
GATE_POS_WEIGHT = 5.4
BATCH = 256
OUT_PATH = "orbit_net.pt"        # best-val-loss checkpoint
LAST_PATH = "orbit_net_last.pt"  # latest-epoch checkpoint (saved every epoch)
LOG_CSV = "train_log.csv"   # per-epoch metrics, plotted by plot_log.py


def compute_losses(net, batch, device):
    """Forward one batch and return (loss, gate_bce, target_ce, ship_ce, ship_mse).
    ship_mse is a diagnostic only (MSE of the categorical mode vs the human fraction)."""
    feats = batch["feats"].to(device)
    valid = batch["valid"].to(device)
    mine = batch["mine"].to(device)
    launch = batch["launch"].to(device)
    target = batch["target"].to(device)
    ships = batch["ships"].to(device)          # fraction of source garrison [0,1]
    has_ship = batch["has_ship"].to(device)

    h = net(feats, valid)                         # [B,N,D] 인코더 출력
    kh = net.k_proj(h)                            # target attention 키
    B = feats.shape[0]

    if not mine.any():
        z0 = torch.zeros((), device=device)
        return z0, z0, z0, z0, z0

    # 내 행성을 보드별 index 오름차순(= AR 디코더 처리 순서)으로 모은다.
    bidx, pidx = torch.nonzero(mine, as_tuple=True)
    gate_label = launch[bidx, pidx].float()       # gate BCE 라벨 (off-board 발사 포함)
    act_launched = has_ship[bidx, pidx].bool()    # 유효한 on-board target으로 실제 발사한 행성
    human_tgt = target[bidx, pidx].clamp(min=0)   # -1(미발사)은 0으로; noop 처리되므로 무방
    frac = ships[bidx, pidx].clamp(0.0, 1.0)
    bin_all = (frac * (net.SHIP_BINS - 1)).round().long().clamp(0, net.SHIP_BINS - 1)

    # 보드별 시퀀스 위치 pos 계산 후 [B,K_max]로 scatter.
    counts = mine.sum(1).long()
    K_max = int(counts.max())
    start = torch.zeros(B, dtype=torch.long, device=device)
    start[1:] = counts.cumsum(0)[:-1]
    pos = torch.arange(bidx.numel(), device=device) - start[bidx]

    def scat(vals, dtype):
        t = torch.zeros(B, K_max, dtype=dtype, device=device)
        t[bidx, pos] = vals.to(dtype)
        return t

    # 디코더 메모리(teacher forcing)는 '사람이 실제 한 행동' 기준. on-board 발사만 행동으로,
    # 나머지는 noop으로 인코딩한다(act_launched).
    planet_pad = scat(pidx, torch.long)
    launched_pad = scat(act_launched, torch.float32)
    target_pad = scat(human_tgt, torch.long)
    bin_pad = scat(bin_all, torch.long)
    seq_mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
    seq_mask[bidx, pos] = True

    z, _ = net._ar_decode(h, planet_pad, launched_pad, target_pad, bin_pad, seq_mask)
    z_flat = z[bidx, pos]                         # [M,D] 각 결정의 AR context

    # Launch gate: BCE over owned planets. gate는 (z_k, target) 조건.
    # 발사 행성은 사람 target, 미발사 행성은 모델 argmax target으로 조건을 잡는다
    # (추론 시 gate가 보게 될 target 분포와 일치).
    tl_all = net._target_logits_from(z_flat, kh, valid, bidx, pidx)   # [M,N]
    argmax_tgt = tl_all.argmax(dim=-1)
    cond_tgt = torch.where(act_launched, human_tgt, argmax_tgt)
    gate_logit, _ = net._gate_ship_from(z_flat, h[bidx, cond_tgt])
    gate_loss = Fn.binary_cross_entropy_with_logits(
        gate_logit, gate_label,
        pos_weight=torch.tensor(GATE_POS_WEIGHT, device=device))

    # Target CE + Ship-fraction CE over resolved launches
    if act_launched.any():
        m = act_launched
        tgt_loss = Fn.cross_entropy(tl_all[m], human_tgt[m])

        # 사람 target을 조건으로 ship bin(SHIP_BINS 클래스) CE
        _, s_logits = net._gate_ship_from(z_flat[m], h[bidx[m], human_tgt[m]])
        ship_loss = Fn.cross_entropy(s_logits, bin_all[m])

        # Diagnostic: argmax bin을 실수 비율로 변환한 MSE
        with torch.no_grad():
            pred_frac = net.bin_to_fraction(s_logits.argmax(dim=-1)).clamp(0.0, 1.0)
            ship_mse = Fn.mse_loss(pred_frac, frac[m])
    else:
        z0 = torch.zeros((), device=device)
        tgt_loss = ship_loss = ship_mse = z0

    loss = gate_loss + tgt_loss + SHIP_LOSS_WEIGHT * ship_loss
    return loss, gate_loss, tgt_loss, ship_loss, ship_mse


@torch.no_grad()
def evaluate(net, loader, device):
    """Mean (loss, gate_bce, target_ce, ship_nll, ship_mse) over a loader, eval mode."""
    was_training = net.training
    net.eval()
    tot, tot_gate, tot_tgt, tot_ship, tot_mse, n_seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for batch in loader:
        loss, gate_loss, tgt_loss, ship_loss, ship_mse = compute_losses(net, batch, device)
        bs = batch["feats"].shape[0]
        tot += loss.item() * bs
        tot_gate += gate_loss.item() * bs
        tot_tgt += tgt_loss.item() * bs
        tot_ship += ship_loss.item() * bs
        tot_mse += ship_mse.item() * bs
        n_seen += bs
    if was_training:
        net.train()
    return (tot / n_seen, tot_gate / n_seen, tot_tgt / n_seen,
            tot_ship / n_seen, tot_mse / n_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", nargs="?", const=OUT_PATH, default=None,
                    help="resume from a checkpoint (default: orbit_net.pt) instead of fresh")
    ap.add_argument("--start_epoch", type=int, default=0,
                    help="epoch to resume from (e.g., --start_epoch 10 resumes from epoch 11)")
    args = ap.parse_args()

    try:
        train_ds, val_ds = load_cached()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"training examples (steps): {len(train_ds)}  "
          f"(val {len(val_ds) if val_ds else 0})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    net = OrbitNet().to(device)
    
    # [수정] 옵티마이저를 먼저 정의해야 state를 불러올 수 있어!
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            # 새로운 방식: 모델 + 옵티마이저가 딕셔너리로 묶여 있는 경우
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"resumed weights and optimizer state from {args.resume} (perfect resume)")
            # 예전 방식: 모델 가중치만 저장되어 있던 경우 (호환성 유지)
            else:
                net.load_state_dict(checkpoint, strict=False)
                print(f"resumed weights from {args.resume} (old format, optimizer state not restored)")
        else:
            print(f"--resume given but {args.resume} not found; starting fresh")

    # persistent_workers: keep workers (and their lazily-opened memmaps) alive
    # across epochs instead of re-spawning every epoch -- less spawn churn and
    # memory thrash on Windows, which was crashing workers mid-run.
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=BATCH,
                                shuffle=False, num_workers=4,
                                pin_memory=(device == "cuda"),
                                persistent_workers=True)

    mode = "a" if args.start_epoch > 0 else "w"
    log_f = open(LOG_CSV, mode, newline="", encoding="utf-8")
    log = csv.writer(log_f)
    
    if args.start_epoch == 0:
        log.writerow(["epoch", "train_loss", "train_gate", "train_ce", "train_ship", "train_mse",
                      "val_loss", "val_gate", "val_ce", "val_ship", "val_mse"])

    best_val = float("inf")
    net.train()
    
    for epoch in range(args.start_epoch, EPOCHS):
        tot, tot_gate, tot_tgt, tot_ship, tot_mse, n_seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch in pbar:
            loss, gate_loss, tgt_loss, ship_loss, ship_mse = compute_losses(net, batch, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()

            bs = batch["feats"].shape[0]
            tot += loss.item() * bs
            tot_gate += gate_loss.item() * bs
            tot_tgt += tgt_loss.item() * bs
            tot_ship += ship_loss.item() * bs
            tot_mse += ship_mse.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{tot/n_seen:.3f}", gate=f"{tot_gate/n_seen:.3f}",
                             ce=f"{tot_tgt/n_seen:.3f}", ship=f"{tot_ship/n_seen:.3f}",
                             mse=f"{tot_mse/n_seen:.3f}")

        tr_loss, tr_gate, tr_ce, tr_ship, tr_mse = (tot/n_seen, tot_gate/n_seen,
                                                    tot_tgt/n_seen, tot_ship/n_seen, tot_mse/n_seen)
        line = (f"epoch {epoch+1:3d}  loss={tr_loss:.4f}  gate={tr_gate:.4f}  "
                f"target_ce={tr_ce:.4f}  ship_nll={tr_ship:.4f}  ship_mse={tr_mse:.4f}")

        v_loss = v_gate = v_tgt = v_ship = v_mse = None
        
        # [수정] 저장을 위한 딕셔너리 생성 (모델 + 옵티마이저 묶기)
        save_dict = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }

        if val_loader is not None:
            v_loss, v_gate, v_tgt, v_ship, v_mse = evaluate(net, val_loader, device)
            line += (f"  ||  val_loss={v_loss:.4f}  val_gate={v_gate:.4f}  "
                     f"val_ce={v_tgt:.4f}  val_ship_ce={v_ship:.4f}  val_ship_mse={v_mse:.4f}")
            if v_loss < best_val:
                best_val = v_loss
                # [수정] 모델 대신 통째로 묶은 딕셔너리를 저장
                torch.save(save_dict, OUT_PATH)
                line += f"  <- best, saved {OUT_PATH}"
        # Always keep the latest-epoch checkpoint (for resume / inspecting the most
        # recent weights), independent of the best-val gate above.
        torch.save(save_dict, LAST_PATH)
        print(line)

        log.writerow([epoch + 1,
                      f"{tr_loss:.6f}", f"{tr_gate:.6f}", f"{tr_ce:.6f}", f"{tr_ship:.6f}", f"{tr_mse:.6f}",
                      "" if v_loss is None else f"{v_loss:.6f}",
                      "" if v_gate is None else f"{v_gate:.6f}",
                      "" if v_tgt is None else f"{v_tgt:.6f}",
                      "" if v_ship is None else f"{v_ship:.6f}",
                      "" if v_mse is None else f"{v_mse:.6f}"])
        log_f.flush()

    log_f.close()
    print(f"metrics logged -> {LOG_CSV}")

    if val_loader is None:
        # [수정] 여기서도 딕셔너리로 저장
        torch.save(save_dict, OUT_PATH)
        print(f"saved final-epoch weights & optimizer state -> {OUT_PATH}")
    else:
        print(f"best val_loss={best_val:.4f}  ({OUT_PATH})")

if __name__ == "__main__":
    main()
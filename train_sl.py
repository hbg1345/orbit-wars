"""Supervised training for OrbitNet -- pure training only.

Data parsing/caching lives in dataset.py. Build the cache there first, then this
script just loads it and trains, saving the best-val-loss weights to orbit_net.pt
and the latest-epoch weights to orbit_net_last.pt (every epoch).

  python dataset.py            # 1) parse replays -> on-disk cache (once)
  python train_sl.py           # 2) train; AUTO-resumes from orbit_net_last.pt if present
  python train_sl.py --fresh   # 2') force a fresh run (ignore any checkpoint)
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
SHIP_LOSS_WEIGHT = 0.5    # ship loss is the cross-entropy of the human launch
                          # fraction binned into SHIP_BINS classes (discrete ship
                          # head), so SL fits the full launch-fraction distribution,
                          # not just its mean. Tune down if the ship term dominates
                          # gate/target.
# Launches are only ~16% of owned-planet decisions, so up-weight the positive
# class in the gate BCE (~ n_neg / n_pos = 84.3 / 15.7). This both pushes launch
# probabilities up (so a ~0.5 threshold works) and gives the rare launch class
# more gradient, sharpening the gate. Set to 1.0 to disable.
GATE_POS_WEIGHT = 5.4
BATCH = 512
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

    # Current OrbitNet: encoder + (gate_head, q/k target attention, ship_head).
    # forward -> target_logits [B,N,N], gate_logits [B,N], h [B,N,D].
    target_logits, gate_logits, h = net(feats, valid, mine)

    if not mine.any():
        z0 = torch.zeros((), device=device)
        return z0, z0, z0, z0, z0

    # 내 행성들을 (board, planet) 인덱스로 평탄화.
    bidx, pidx = torch.nonzero(mine, as_tuple=True)
    gate_label = launch[bidx, pidx].float()       # gate BCE 라벨 (off-board 발사 포함)
    act_launched = has_ship[bidx, pidx].bool()    # 유효한 on-board target으로 실제 발사한 행성
    human_tgt = target[bidx, pidx].clamp(min=0)   # -1(미발사)은 0으로; noop 처리되므로 무방
    frac = ships[bidx, pidx].clamp(0.0, 1.0)
    # 사람 비율 -> bin index (model.bin_to_fraction의 역변환):
    # bin 0~99 = (b+0.5)/100 (0.5%..99.5%), bin 100 = 100%. frac>=99.5%는 bin 100.
    denom = net.SHIP_BINS - 1                      # = 100
    bin_all = torch.where(
        frac >= 1.0 - 0.5 / denom,
        torch.full_like(frac, float(denom)),
        (frac * denom - 0.5).round().clamp(0, denom - 1),
    ).long()

    # Launch gate: 내 모든 행성에 대한 BCE (gate_head는 target 조건 없이 행성 임베딩만 사용).
    gate_loss = Fn.binary_cross_entropy_with_logits(
        gate_logits[bidx, pidx], gate_label,
        pos_weight=torch.tensor(GATE_POS_WEIGHT, device=device))

    # Target CE + Ship-fraction CE over resolved launches
    if act_launched.any():
        m = act_launched
        mb, mp, mtgt = bidx[m], pidx[m], human_tgt[m]

        # Target: 해당 source 행성의 target_logits [N]에 대한 CE
        tgt_loss = Fn.cross_entropy(target_logits[mb, mp], mtgt)

        # Ship bin: (source, human_target) 쌍 임베딩 -> ship_head [SHIP_BINS] CE.
        # ship_logits(h, b_idx, src, tgt)는 h[b_idx, src] 식으로 배치 인덱싱한다.
        s_logits = net.ship_logits(h, mb, mp, mtgt)
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
    ap.add_argument("--resume", nargs="?", const=LAST_PATH, default=None,
                    help=f"checkpoint to resume from. Omit entirely: auto-resume from "
                         f"{LAST_PATH} if it exists. Bare --resume: also {LAST_PATH}. "
                         f"--resume PATH: that file.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start from scratch "
                         "(disables auto-resume).")
    ap.add_argument("--start_epoch", type=int, default=None,
                    help="manual override for the resume epoch (0-based count of "
                         "completed epochs). Default: auto from the checkpoint's "
                         "saved epoch (0 if fresh / old-format checkpoint).")
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
    
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Resume source resolution: --fresh forces scratch; an explicit --resume PATH
    # wins; otherwise auto-resume from LAST_PATH if it's there (no flag needed).
    if args.fresh:
        resume_path = None
    elif args.resume is not None:
        resume_path = args.resume
    elif os.path.exists(LAST_PATH):
        resume_path = LAST_PATH
        print(f"auto-resuming from {LAST_PATH} (use --fresh to start over)")
    else:
        resume_path = None

    # Resume state: epoch (0-based # of completed epochs) and best val loss are
    # restored from the checkpoint so you don't have to pass --start_epoch by hand.
    start_epoch = 0
    best_val = float("inf")
    if resume_path:
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            # 새로운 방식: 모델 + 옵티마이저가 딕셔너리로 묶여 있는 경우
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = int(checkpoint.get('epoch', 0))      # epochs already done
                best_val = float(checkpoint.get('best_val', float("inf")))
                print(f"resumed from {resume_path} at epoch {start_epoch} "
                      f"(optimizer + best_val={best_val:.4f} restored)")
            # 예전 방식: 모델 가중치만 저장되어 있던 경우 (호환성 유지)
            else:
                net.load_state_dict(checkpoint, strict=False)
                print(f"resumed weights from {resume_path} (old format, no epoch/optimizer state)")
        else:
            print(f"--resume given but {resume_path} not found; starting fresh")

    # Manual override always wins over the checkpoint's saved epoch.
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    # persistent_workers: keep workers (and their lazily-opened memmaps) alive
    # across epochs instead of re-spawning every epoch -- less spawn churn and
    # memory thrash on Windows, which was crashing workers mid-run.
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        num_workers=1, pin_memory=(device == "cuda"),
                        persistent_workers=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=BATCH,
                                shuffle=False, num_workers=4,
                                pin_memory=(device == "cuda"),
                                persistent_workers=True)

    mode = "a" if start_epoch > 0 else "w"
    log_f = open(LOG_CSV, mode, newline="", encoding="utf-8")
    log = csv.writer(log_f)

    if start_epoch == 0:
        log.writerow(["epoch", "train_loss", "train_gate", "train_ce", "train_ship", "train_mse",
                      "val_loss", "val_gate", "val_ce", "val_ship", "val_mse"])

    net.train()
    
    for epoch in range(start_epoch, EPOCHS):
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
        is_best = False
        if val_loader is not None:
            v_loss, v_gate, v_tgt, v_ship, v_mse = evaluate(net, val_loader, device)
            line += (f"  ||  val_loss={v_loss:.4f}  val_gate={v_gate:.4f}  "
                     f"val_ce={v_tgt:.4f}  val_ship_ce={v_ship:.4f}  val_ship_mse={v_mse:.4f}")
            if v_loss < best_val:
                best_val = v_loss
                is_best = True

        # Checkpoint bundle: weights + optimizer + resume state. 'epoch' is the
        # count of COMPLETED epochs (= epoch+1), i.e. the value to resume at next.
        save_dict = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch + 1,
            'best_val': best_val,
        }
        if is_best:
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
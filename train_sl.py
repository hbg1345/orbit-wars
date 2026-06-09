"""Supervised training for OrbitNet -- pure training only.

Data parsing/caching lives in dataset.py. Build the cache there first, then this
script just loads it and trains, saving the best-val-loss weights to orbit_net.pt.

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

EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5       # L2 regularization (cf. supervised.py)
SHIP_LOSS_WEIGHT = 1.0    # ship loss is now Beta NLL (a log-density, can be <0), so it
                          # sits on the same scale as the target cross-entropy
# Launches are only ~16% of owned-planet decisions, so up-weight the positive
# class in the gate BCE (~ n_neg / n_pos = 84.3 / 15.7). This both pushes launch
# probabilities up (so a ~0.5 threshold works) and gives the rare launch class
# more gradient, sharpening the gate. Set to 1.0 to disable.
GATE_POS_WEIGHT = 5.4
BATCH = 256
OUT_PATH = "orbit_net.pt"
LOG_CSV = "train_log.csv"   # per-epoch metrics, plotted by plot_log.py


def compute_losses(net, batch, device):
    """Forward one batch and return (loss, gate_bce, target_ce, ship_nll). Shared
    by the train and validation loops so they score identically."""
    feats = batch["feats"].to(device)
    valid = batch["valid"].to(device)
    mine = batch["mine"].to(device)
    launch = batch["launch"].to(device)
    target = batch["target"].to(device)
    ships = batch["ships"].to(device)          # fraction of source garrison [0,1]
    has_ship = batch["has_ship"].to(device)

    target_logits, gate_logits, h = net(feats, valid, mine)   # [B,N,N],[B,N],[B,N,D]

    # Launch gate: BCE over owned planets -- did this planet launch at all?
    # pos_weight up-weights the rare launch class (see GATE_POS_WEIGHT).
    if mine.any():
        gate_loss = Fn.binary_cross_entropy_with_logits(
            gate_logits[mine], launch[mine].float(),
            pos_weight=torch.tensor(GATE_POS_WEIGHT, device=device))
    else:
        gate_loss = torch.zeros((), device=device)

    # Target CE + ship-fraction Beta NLL over resolved launches (has_ship), using the
    # (source, human-chosen target) pair.
    if has_ship.any():
        b_idx, src_idx = torch.nonzero(has_ship, as_tuple=True)
        tgt_idx = target[b_idx, src_idx]
        tgt_loss = Fn.cross_entropy(target_logits[b_idx, src_idx], tgt_idx)
        # Ship fraction: NLL of the (now unified) Beta head on the human fraction,
        # so SL pre-trains the very distribution RL later samples from.
        a, b = net.ship_beta(h, b_idx, src_idx, tgt_idx)
        fr = ships[b_idx, src_idx].clamp(1e-4, 1 - 1e-4)        # keep in (0,1)
        ship_loss = -torch.distributions.Beta(a, b).log_prob(fr).mean()
    else:
        tgt_loss = torch.zeros((), device=device)
        ship_loss = torch.zeros((), device=device)

    loss = gate_loss + tgt_loss + SHIP_LOSS_WEIGHT * ship_loss
    return loss, gate_loss, tgt_loss, ship_loss


@torch.no_grad()
def evaluate(net, loader, device):
    """Mean (loss, gate_bce, target_ce, ship_nll) over a loader, in eval mode."""
    was_training = net.training
    net.eval()
    tot, tot_gate, tot_tgt, tot_ship, n_seen = 0.0, 0.0, 0.0, 0.0, 0
    for batch in loader:
        loss, gate_loss, tgt_loss, ship_loss = compute_losses(net, batch, device)
        bs = batch["feats"].shape[0]
        tot += loss.item() * bs
        tot_gate += gate_loss.item() * bs
        tot_tgt += tgt_loss.item() * bs
        tot_ship += ship_loss.item() * bs
        n_seen += bs
    if was_training:
        net.train()
    return tot / n_seen, tot_gate / n_seen, tot_tgt / n_seen, tot_ship / n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", nargs="?", const=OUT_PATH, default=None,
                    help="resume from a checkpoint (default: orbit_net.pt) instead of fresh")
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
    if args.resume:
        if os.path.exists(args.resume):
            net.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
            print(f"resumed weights from {args.resume} (optimizer state not restored)")
        else:
            print(f"--resume given but {args.resume} not found; starting fresh")
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=BATCH,
                                shuffle=False, num_workers=0,
                                pin_memory=(device == "cuda"))

    log_f = open(LOG_CSV, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "train_gate", "train_ce", "train_ship",
                  "val_loss", "val_gate", "val_ce", "val_ship"])

    best_val = float("inf")
    net.train()
    for epoch in range(EPOCHS):
        tot, tot_gate, tot_tgt, tot_ship, n_seen = 0.0, 0.0, 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch in pbar:
            loss, gate_loss, tgt_loss, ship_loss = compute_losses(net, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = batch["feats"].shape[0]
            tot += loss.item() * bs
            tot_gate += gate_loss.item() * bs
            tot_tgt += tgt_loss.item() * bs
            tot_ship += ship_loss.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{tot/n_seen:.3f}", gate=f"{tot_gate/n_seen:.3f}",
                             ce=f"{tot_tgt/n_seen:.3f}", ship=f"{tot_ship/n_seen:.3f}")

        tr_loss, tr_gate, tr_ce, tr_ship = (tot/n_seen, tot_gate/n_seen,
                                            tot_tgt/n_seen, tot_ship/n_seen)
        line = (f"epoch {epoch+1:3d}  loss={tr_loss:.4f}  gate={tr_gate:.4f}  "
                f"target_ce={tr_ce:.4f}  ship_nll={tr_ship:.4f}")

        v_loss = v_gate = v_tgt = v_ship = None
        if val_loader is not None:
            v_loss, v_gate, v_tgt, v_ship = evaluate(net, val_loader, device)
            line += (f"  ||  val_loss={v_loss:.4f}  val_gate={v_gate:.4f}  "
                     f"val_ce={v_tgt:.4f}  val_ship_nll={v_ship:.4f}")
            if v_loss < best_val:
                best_val = v_loss
                torch.save(net.state_dict(), OUT_PATH)
                line += f"  <- best, saved {OUT_PATH}"
        print(line)

        log.writerow([epoch + 1,
                      f"{tr_loss:.6f}", f"{tr_gate:.6f}", f"{tr_ce:.6f}", f"{tr_ship:.6f}",
                      "" if v_loss is None else f"{v_loss:.6f}",
                      "" if v_gate is None else f"{v_gate:.6f}",
                      "" if v_tgt is None else f"{v_tgt:.6f}",
                      "" if v_ship is None else f"{v_ship:.6f}"])
        log_f.flush()   # so plot_log.py can read partial runs live

    log_f.close()
    print(f"metrics logged -> {LOG_CSV}")

    if val_loader is None:
        # No validation set (too few replays); keep the final-epoch weights.
        torch.save(net.state_dict(), OUT_PATH)
        print(f"saved final-epoch weights -> {OUT_PATH}")
    else:
        print(f"best val_loss={best_val:.4f}  ({OUT_PATH})")


if __name__ == "__main__":
    main()

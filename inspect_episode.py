"""Inspect how orbit_net.pt predicts target + ship count across episodes.

For every step where the winner owned planets, this prints the model's decision.
Can evaluate a single episode or aggregate results across MULTIPLE episodes.

Usage:
  python inspect_episode.py                     # evaluates ALL data/*.json files
  python inspect_episode.py data/x.json         # evaluates that specific replay
  python inspect_episode.py data/x.json --all   # show every step, not just launches
  python inspect_episode.py --weights ckpt/epoch_30.pt
  python inspect_episode.py --limit 20          # evaluate only the first 20 episodes
"""

import glob
import json
import sys
import zipfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import features as F
import physics as P
import replay
from model import OrbitNet

GATE_THRESHOLD = 0.25


def _resolve_human(steps, fleet_idx, t, winner, ids, id2slot, mine):
    # (기존 함수 동일)
    action = steps[t][winner].get("action") or []
    by_src = {}
    for mv in action:
        fid, ang, nsh = int(mv[0]), float(mv[1]), int(mv[2])
        si = id2slot.get(fid)
        if si is None or not mine[si]:
            continue
        if si not in by_src or nsh > by_src[si][1]:
            by_src[si] = (ang, nsh)

    out = {}
    if by_src:
        body_at = replay._body_track(P.build_world(steps[t][0]["observation"]))
        for si, (ang, nsh) in by_src.items():
            fid = int(ids[si])
            tgt_id = replay._resolve_target(fleet_idx, t, fid, ang, nsh, body_at)
            out[si] = (id2slot.get(tgt_id), nsh)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    show_all = "--all" in args
    args = [a for a in args if a != "--all"]
    weights = "orbit_net.pt"
    if "--weights" in args:
        i = args.index("--weights")
        weights = args[i + 1]
        del args[i:i + 2]

    # --limit / -n N: cap how many episodes to evaluate (default: all).
    limit = None
    for flag in ("--limit", "-n"):
        if flag in args:
            i = args.index(flag)
            limit = int(args[i + 1])
            del args[i:i + 2]
            break

    # 인자가 없으면 data 폴더의 loose json + zip 안의 에피소드를 모두 가져옴.
    # zip은 dataset.py처럼 멤버를 메모리에서 읽음(디스크에 풀지 않음). 각 item은
    # .json 경로이거나 (zip_path, member) 튜플.
    inputs = args if args else sorted(glob.glob("data/*.json") + glob.glob("data/zips/*.zip"))
    paths = []
    for p in inputs:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                paths += [(p, n) for n in zf.namelist() if n.lower().endswith(".json")]
        else:
            paths.append(p)
    # Default order is newest first: zip filenames are ISO-dated, so a reverse
    # sort puts the latest days first (matching dataset.py / download_latest.sh).
    # An explicit file list keeps the order the user gave.
    if not args:
        paths.sort(key=lambda it: it if isinstance(it, str) else f"{it[0]}::{it[1]}",
                   reverse=True)
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        print("No JSON replay files found.")
        return

    print(f"Found {len(paths)} episodes. Loading weights from {weights}...\n")

    net = OrbitNet()
    ckpt = torch.load(weights, map_location="cpu")
    # train_sl.py now saves {'model_state_dict', 'optimizer_state_dict'};
    # fall back to the old raw-state_dict format for older checkpoints.
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()

    all_records = []
    is_multi_file = len(paths) > 1

    for e_idx, item in enumerate(paths):
        try:
            if isinstance(item, tuple):
                zip_path, member = item
                with zipfile.ZipFile(zip_path) as zf:
                    d = json.loads(zf.read(member))
                path = f"{zip_path}::{member}"        # display label only
            else:
                path = item
                d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {item} due to error: {e}")
            continue
            
        steps, rewards = d["steps"], d["rewards"]
        winner = int(np.argmax(rewards))
        fleet_idx = replay._index_fleets(steps)
        # Canonicalisation MUST match training (replay.py) and live (main.py):
        # rotate the winner's start quadrant -> Q1. Computed once from the initial
        # board. Without this, the model was evaluated on a raw frame it never
        # trained on, which is why the sweep looked good but live play didn't.
        obs0 = steps[0][winner].get("observation", {})
        canon = F.compute_canon(obs0) if obs0.get("planets") else None
        
        # 파일이 여러 개일 때는 진행 상황만 간단히 출력
        if is_multi_file and not show_all:
            print(f"Processing [{e_idx+1}/{len(paths)}] {path} ...")
        elif not is_multi_file:
            print(f"replay {path}  winner=player{winner}  steps={len(steps)}\n")

        for t in range(1, len(steps)):
            obs = steps[t - 1][winner].get("observation", {})
            if not obs.get("planets"):
                continue

            feats, valid, mine, ids, xs, ys, ships = F.extract(obs, canon=canon)
            id2slot = {int(pid): i for i, pid in enumerate(ids) if pid >= 0}
            human = _resolve_human(steps, fleet_idx, t, winner, ids, id2slot, mine)

            if not human and not show_all:
                continue

            ft = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
            vm = torch.as_tensor(valid, dtype=torch.bool).unsqueeze(0)
            mm = torch.as_tensor(mine, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                tgt_logits, gate_logits, h = net(ft, vm, mm)
            gate_p = torch.sigmoid(gate_logits[0])
            tprobs = torch.softmax(tgt_logits[0], dim=-1)

            # 단일 파일이거나 --all 옵션이 있을 때만 상세 스텝 출력
            if not is_multi_file or show_all:
                print(f"=== step {t}  (input = pre-action board obs[{t-1}]) ===")
                print(f"  {'src':>5} {'gar':>5} | {'gate':>5} {'MODEL tgt':>10} {'p':>5} {'ships':>6} | "
                      f"{'HUMAN tgt':>10} {'ships':>6} | match")

            for i in np.nonzero(mine)[0]:
                g = float(gate_p[i].item())
                tp = tprobs[i]
                m_tgt = int(torch.argmax(tp).item())
                frac = float(net.ship_fraction(h, torch.tensor([0]), torch.tensor([i]),
                                               torch.tensor([m_tgt])).item())
                m_ships = frac * float(ships[i])

                launched = g >= GATE_THRESHOLD
                if launched:
                    m_str = f"P{int(ids[m_tgt])}"
                    mp_str, ms_str = f"{float(tp[m_tgt]):.2f}", f"{m_ships:.0f}"
                else:
                    m_str, mp_str, ms_str = "NOOP", "-", "-"

                h_entry = human.get(i)
                if h_entry is None:
                    h_tgt_slot, h_ships = None, None
                    h_str, hs_str = ("-" if not show_all else "noop"), "-"
                else:
                    h_tgt_slot, h_ships = h_entry
                    h_str = f"P{int(ids[h_tgt_slot])}" if h_tgt_slot is not None else "off-board"
                    hs_str = f"{h_ships}"

                human_launched = h_entry is not None and h_tgt_slot is not None
                
                # 글로벌 레코드 배열에 저장
                all_records.append((g, human_launched, h_tgt_slot if human_launched else -1,
                                m_tgt, m_ships, h_ships if human_launched else 0.0))

                mark = ""
                if human_launched:
                    mark = "OK" if (launched and m_tgt == h_tgt_slot) else "x"
                
                if not is_multi_file or show_all:
                    print(f"  {int(ids[i]):>5} {ships[i]:>5.0f} | {g:>5.2f} {m_str:>10} {mp_str:>5} "
                          f"{ms_str:>6} | {h_str:>10} {hs_str:>6} | {mark}")
            if not is_multi_file or show_all:
                print()

    # 글로벌 요약본 (Global Summary) 계산
    if not all_records:
        print("\nNo valid records extracted from the provided files.")
        return

    g = np.array([r[0] for r in all_records])
    hl = np.array([r[1] for r in all_records], dtype=bool)
    htgt = np.array([r[2] for r in all_records])
    mtgt = np.array([r[3] for r in all_records])
    mship = np.array([r[4] for r in all_records])
    hship = np.array([r[5] for r in all_records])

    n_owned = len(all_records)
    n_hl = int(hl.sum())
    
    print("\n================ GLOBAL SUMMARY ================")
    print(f"Total Episodes Analyzed: {len(paths)}")
    print(f"Total owned-planet decisions: {n_owned}   Total human launches: {n_hl}")
    
    if n_hl:
        print(f"gate prob (mean): launch planets={g[hl].mean():.3f}  "
              f"idle planets={g[~hl].mean():.3f}   (higher gap = better discrimination)")
        correct = hl & (mtgt == htgt)
        print(f"target top-1 (when human launched, ignoring gate): "
              f"{int(correct.sum())}/{n_hl} ({100*correct.sum()/n_hl:.1f}%)")
        se = np.abs(mship[correct] - hship[correct])
        if se.size:
            print(f"ship |error| on correct targets: median={np.median(se):.1f} "
                  f"mean={se.mean():.1f} p90={np.percentile(se,90):.1f}")

        # ROC AUC 계산
        if 0 < n_hl < n_owned:
            auc_score = roc_auc_score(hl, g)
            print(f"\nLaunch Gate ROC AUC: {auc_score:.4f}")
        else:
            print("\nLaunch Gate ROC AUC: N/A (requires both positive and negative samples)")

        # F1-Score가 포함된 통합 Threshold Sweep
        print("\ngate-threshold sweep:")
        print(f"  {'thr':>5} {'fires':>7} {'recall':>8} {'prec':>8} {'f1-score':>9} {'tgt-match':>10} {'false-fire':>11}")
        for thr in [0.05 * (i+1) for i in range(20)]:
            fired = g >= thr
            tp = (fired & hl).sum()
            fp = (fired & ~hl).sum()
            
            recall = tp / n_hl if n_hl > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            tmatch = (fired & correct).sum() / n_hl if n_hl > 0 else 0.0
            false_fire = int(fp)
            
            print(f"  {thr:>5.2f} {int(fired.sum()):>7} {100*recall:>7.0f}% "
                  f"{100*precision:>7.0f}% {100*f1:>8.1f}% {100*tmatch:>9.0f}% {false_fire:>11}")

if __name__ == "__main__":
    main()
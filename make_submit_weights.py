"""Strip the optimizer state out of a training checkpoint, leaving only the model
weights, for bundling into the Kaggle submission.

train_sl.py saves {'model_state_dict', 'optimizer_state_dict', 'epoch', 'best_val'}.
The optimizer state (Adam's two moment buffers) is ~2x the size of the weights, so
the full orbit_net.pt is ~146MB while the weights alone are ~50MB (fp32) / ~25MB
(fp16). The submission only needs the weights, so we write a slim checkpoint and
leave the full orbit_net.pt untouched for resuming training.

main.py._load_state() accepts either the wrapped dict or a raw state_dict, so the
slim file loads with no agent-side change. With --half the weights are stored as
float16; load_state_dict() casts them back to the model's fp32 params on load.

Usage:
  python make_submit_weights.py                       # orbit_net.pt -> orbit_net_submit.pt (fp32)
  python make_submit_weights.py --half                # ~25MB instead of ~50MB
  python make_submit_weights.py --src X.pt --dst Y.pt
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="orbit_net.pt", help="full training checkpoint")
    ap.add_argument("--dst", default="orbit_net_submit.pt", help="slim weights-only output")
    ap.add_argument("--half", action="store_true",
                    help="store float weights as float16 (~half the size)")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        raise SystemExit(f"error: source checkpoint not found: {args.src}")

    ckpt = torch.load(args.src, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt  # already a raw state_dict (older format)
    else:
        raise SystemExit(f"error: unexpected checkpoint format in {args.src}: {type(ckpt)}")

    if args.half:
        sd = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}

    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    torch.save(sd, args.dst)

    src_mb = os.path.getsize(args.src) / 1e6
    dst_mb = os.path.getsize(args.dst) / 1e6
    print(f"{args.src} ({src_mb:.1f}MB) -> {args.dst} ({dst_mb:.1f}MB), "
          f"{n:,} params{' [fp16]' if args.half else ''}")


if __name__ == "__main__":
    main()

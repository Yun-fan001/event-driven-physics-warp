"""对指定 checkpoint 跑转移锚定评估 (--ckpt --name)。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from event_wm import (build_model, evaluate, intervention,
                      load_trajectories, split_by_friction)
from anchored_eval import make_anchored_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizons", default="16,32,64,96,128")
    ap.add_argument("--data-dir", default="data/stick_slip")
    ap.add_argument("--out-dir", default="checkpoints_wm")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"ckpt 不存在: {ckpt}")
    model_name = args.name.replace("_anchored", "")
    model = build_model(model_name)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    trajs = load_trajectories(args.data_dir)
    _, test_id, test_ood = split_by_friction(trajs)

    res = {}
    for hz in horizons:
        for split, tr in [("id", test_id), ("ood", test_ood)]:
            xs, ym, yf, fr = make_anchored_windows(tr, hz)
            if len(xs) == 0:
                res[f"{split}_h{hz}"] = {"n": 0}
                continue
            r, _, _ = evaluate(model, xs, ym, yf, friction=None)
            iv = intervention(model, xs, ym, yf, fr, fr * 1.5)
            r["intervention"] = iv
            r["n"] = len(xs)
            res[f"{split}_h{hz}"] = r
            print(f"  [{args.name}] {split} h={hz}: n={len(xs)} "
                  f"FF={r['first_failure_mean']:.1f} evF1={r['event_f1']:.3f} "
                  f"iv_rate={iv.get('direction_rate')}")
    out = Path(args.out_dir) / f"{args.name}_anchored_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

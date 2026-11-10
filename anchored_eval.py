"""转移锚定评估: GT 滑移转移前 horizon 帧开始 roll-out,
直接量化"世界模型能否提前 Δ 预测滑移事件" (ID vs OOD + 干预)。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from event_wm import (H, K, build_model, evaluate, intervention,
                      load_trajectories, split_by_friction, threshold_events,
                      ForceWM, ImplicitWM, EventWM)


def find_transitions(mode):
    """全局 1→2 转移帧索引"""
    out = []
    for t in range(1, len(mode)):
        if mode[t - 1] == 1 and mode[t] == 2:
            out.append(t)
    return out


def make_anchored_windows(trajs, horizon, rng=None, max_per_traj=4):
    """转移锚定: roll-out 起点 = t* - horizon - K; GT 转移位于 roll-out 的第 horizon 帧"""
    xs, ym, yf, fric = [], [], [], []
    for t in trajs:
        T = len(t["mode"])
        feat = np.concatenate([
            t["force"],
            t["pusher"][:, None],
            np.full((T, 1), t["friction"], dtype=np.float32),
        ], axis=1)
        n_traj = 0
        for t_star in find_transitions(t["mode"]):
            s = t_star - horizon - K
            if s < 0 or s + K + H > T:
                continue
            xs.append(feat[s:s + K])
            ym.append(t["mode"][s + K:s + K + H])
            yf.append(t["force"][s + K:s + K + H])
            fric.append(t["friction"])
            n_traj += 1
            if n_traj >= max_per_traj:
                break
    if not xs:
        return (np.zeros((0, K, 5), np.float32), np.zeros((0, H), np.int64),
                np.zeros((0, H, 3), np.float32), np.zeros(0, np.float32))
    return (np.stack(xs), np.stack(ym), np.stack(yf),
            np.array(fric, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="event_wm,force_wm,implicit_wm")
    ap.add_argument("--horizons", default="16,32,64,96,128")
    ap.add_argument("--data-dir", default="data/stick_slip")
    ap.add_argument("--ckpt-dir", default="checkpoints_wm")
    args = ap.parse_args()
    models = args.models.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    # h>=H-1 为退化视界 (GT 转移在 roll-out 窗口外), 自动剔除
    horizons = [h for h in horizons if h < H]

    trajs = load_trajectories(args.data_dir)
    _, test_id, test_ood = split_by_friction(trajs)

    summary = {}
    for name in models:
        ckpt = Path(args.ckpt_dir) / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt} 不存在")
            continue
        model = build_model(name)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
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
                print(f"  [{name}] {split} h={hz}: n={len(xs)} "
                      f"FF={r['first_failure_mean']:.1f} "
                      f"evF1={r['event_f1']:.3f} iv_rate={iv.get('direction_rate')}")
        out = Path(args.ckpt_dir) / f"{name}_anchored_results.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        summary[name] = res
        print(f"saved -> {out}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()

"""③ 检测器条件化 EventWM (DetEventWM):
把已训练的 Force-TCN 检测器 (事件级 F1 0.543, 判别式最佳) 作为冻结的感知前端,
其对 K 帧输入历史的逐帧事件 logits 拼入世界模型输入 (5 维 → 8 维)。
检验: 显式检测特征能否提升事件生成时刻精度 (锚定 evF1)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from models.event_tcn import EventTCN
from event_wm import (Encoder, EventWM, evaluate, intervention, K, H,
                      load_trajectories, make_anchored_train, make_windows,
                      split_by_friction, train_model)
from anchored_eval import make_anchored_windows


def load_detector(ckpt_path, freeze=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    det = EventTCN(
        num_classes=args.get("num_classes", 3),
        vision_feat_dim=0,                       # Force-TCN
        force_dim=args.get("force_dim", 3),
        proprio_dim=args.get("proprio_dim", 1),
        hidden_dim=args.get("hidden_dim", 128),
        tcn_layers=args.get("tcn_layers", 4),
    )
    det.load_state_dict(
        {k.replace("model.", ""): v
         for k, v in ckpt["model_state_dict"].items()})
    if freeze:
        det.eval()
        for p in det.parameters():
            p.requires_grad_(False)
    return det


@torch.no_grad()
def augment(xs, det, batch=64):
    """xs (N,K,5) → (N,K,8): 追加检测器逐帧事件 logits"""
    out = []
    for i in range(0, len(xs), batch):
        xb = torch.from_numpy(xs[i:i + batch])
        force = xb[:, :, :3]
        proprio = xb[:, :, 3:4]
        logits = det(None, force, proprio)       # (B,K,3)
        out.append(torch.cat([xb, logits], dim=-1))
    return torch.cat(out, 0).numpy()


def det_logits(det, xb):
    """可微版: 检测器 logits (联合训练用), xb (B,K,5) → (B,K,3)"""
    return det(None, xb[:, :, :3], xb[:, :, 3:4])


class DetEventWM(EventWM):
    def __init__(self, num_modes=3, hdim=128):
        super().__init__(num_modes, hdim)
        self.enc = Encoder(8)                    # 5 原始特征 + 3 检测 logits
class JointDetEventWM(EventWM):
    """联合优化版: 检测器可训练, 与 WM 一起由生成损失端到端优化"""
    def __init__(self, det, num_modes=3, hdim=128):
        super().__init__(num_modes, hdim)
        self.enc = Encoder(8)
        self.det = det                       # trainable

    def forward(self, x, gt_modes=None):
        x8 = torch.cat([x, det_logits(self.det, x)], dim=-1)
        return super().forward(x8, gt_modes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-ckpt",
                    default="checkpoints_wm_det/stick_slip/force_tcn/best.pt")
    ap.add_argument("--joint", action="store_true",
                    help="联合训练: 检测器与 WM 端到端 (否则冻结前端)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    det = load_detector(args.detector_ckpt, freeze=not args.joint)
    trajs = load_trajectories("data/stick_slip")
    train_t, test_id, test_ood = split_by_friction(trajs)

    rng = np.random.default_rng(args.seed)
    xs, ym, yf = make_windows(train_t, rng, n_per_traj=5)
    xa, ya, yfa = make_anchored_train(train_t, rng)
    xs = np.concatenate([xs, xa])
    ym = np.concatenate([ym, ya])
    yf = np.concatenate([yf, yfa])
    counts = np.bincount(ym.reshape(-1), minlength=3).astype(float)
    counts[counts == 0] = 1e-6
    w = counts.sum() / (3 * counts)
    class_weights = torch.from_numpy(w.astype(np.float32))
    print(f"windows: {len(xs)}, class weights: {w}")

    name = "det_joint" if args.joint else "det_event_wm"
    if args.joint:
        model = JointDetEventWM(det)
        xs_in = xs
    else:
        model = DetEventWM()
        xs_in = augment(xs, det)
    train_model(model, xs_in, ym, yf, epochs=args.epochs, seed=args.seed,
                class_weights=class_weights)

    res = {"model": name, "seed": args.seed,
           "detector": args.detector_ckpt}
    for hz in [16, 32, 64, 96]:
        xw, yw, yfw, fr = make_anchored_windows(test_ood, hz)
        if len(xw) == 0:
            continue
        xw_in = xw if args.joint else augment(xw, det)
        r, _, _ = evaluate(model, xw_in, yw, yfw, friction=None)
        iv = intervention(model, xw_in, yw, yfw, fr, fr * 1.5)
        r["intervention"] = iv
        r["n"] = len(xw)
        res[f"ood_h{hz}"] = r
        print(f"  [ood h{hz}] n={len(xw)} FF={r['first_failure_mean']:.1f} "
              f"evF1={r['event_f1']:.3f} iv_rate={iv.get('direction_rate')} "
              f"shift={iv.get('mean_shift_frames')}")

    out = Path("checkpoints_wm") / f"{name}_s{args.seed}_anchored.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    torch.save(model.state_dict(),
               Path("checkpoints_wm") / f"{name}_s{args.seed}.pt")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

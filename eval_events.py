"""
事件级评估脚本 (2026-11 清理版新增)
对训练好的 checkpoint 在测试集上做【事件级】F1/P/R (±5 帧容差),
补充 train_step2.py 只报帧级 macro-F1 的不足。

关键: 帧级 F1 由随机窗口拼接而成, 窗口边界会产生虚假的状态转移,
无法用于事件级指标。本脚本对每条测试轨迹【整条顺序分块】前向,
在整条轨迹上计算事件转移 (从 GT 与预测的状态序列)。

用法:
  python eval_events.py --task stick_slip --model force_tcn \
      --checkpoint checkpoints/stick_slip/force_tcn/best.pt --data-dir data/stick_slip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import ContactEventDataset
from train_step2 import ModelWrapper, force_to_events, TASK_CONFIG

TRANSITIONS = {
    "stick_slip": [(1, 2)],
    "peg_in_hole": [(0, 1)],
    "stack_block": [(0, 1), (0, 2)],
}


def transitions_of(states, transitions):
    s = np.asarray(states)
    idx = [np.where((s[:-1] == f) & (s[1:] == t))[0] + 1 for f, t in transitions]
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=int)


def event_metrics(gt_states, pred_states, transitions, tolerance=5):
    gt = transitions_of(gt_states, transitions)
    pd = transitions_of(pred_states, transitions)
    n_gt, n_pd = len(gt), len(pd)
    if n_gt == 0:
        return {"gt_events": 0, "pred_events": n_pd,
                "precision": 1.0 if n_pd == 0 else 0.0, "recall": 1.0, "f1": 0.0,
                "time_errors": []}
    used = np.zeros(n_pd, dtype=bool)
    matched = 0
    time_errors = []
    for g in gt:
        best = None
        for j, p in enumerate(pd):
            if used[j]:
                continue
            d = abs(int(p) - int(g))
            if d <= tolerance and (best is None or d < best[1]):
                best = (j, d)
        if best is not None:
            matched += 1
            used[best[0]] = True
            time_errors.append(best[1])
    recall = matched / n_gt
    precision = int(used.sum()) / n_pd if n_pd else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"gt_events": n_gt, "pred_events": n_pd,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "time_errors": [int(e) for e in time_errors]}


def _proprio(d, task):
    if task == "stick_slip":
        return np.stack([d["pusher_x"].astype(np.float32)], axis=-1)
    if task == "peg_in_hole":
        return np.stack([d["peg_z"].astype(np.float32)], axis=-1)
    return np.stack([d["gripper_x"].astype(np.float32),
                     d["gripper_z"].astype(np.float32)], axis=-1)



@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIG.keys()))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--tolerance", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--tcn-layers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelWrapper(args.model, args.task,
                         hidden_dim=args.hidden_dim, tcn_layers=args.tcn_layers).to(device)
    if args.model != "rule_detector":
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 复现训练时的 70/15/15 划分 (与 train_step2.py 一致)
    dataset = ContactEventDataset(args.data_dir, task=args.task, seq_len=None, cache=False)
    n = len(dataset)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n, generator=g).tolist()
    test_idx = idx[n_tr + n_va:]
    test_files = [dataset.files[i] for i in test_idx]

    aggr = {"gt_events": 0, "pred_events": 0, "matched_gt": 0, "matched_pd": 0, "te": []}
    frame_correct = frame_total = 0

    for f in test_files:
        d = np.load(f)
        T = d["forces"].shape[0]
        ri = int(d.get("render_interval", 1))
        fr = d["frames"]
        if fr.shape[0] < T and ri > 1:
            frames = np.repeat(fr, ri, axis=0)[:T]
        else:
            frames = fr[:T]
        frames = np.transpose(frames, (0, 3, 1, 2)).astype(np.float32) / 255.0
        force = d["forces"].astype(np.float32)
        vel = d["velocities"].astype(np.float32)
        proprio = _proprio(d, args.task)
        gt = d["contact_states_gt"].astype(np.int64)

        preds = []
        for s in range(0, T, args.seq_len):
            e = min(s + args.seq_len, T)
            vis = torch.from_numpy(frames[s:e]).unsqueeze(0).to(device)
            fo = torch.from_numpy(force[s:e]).unsqueeze(0).to(device)
            pr = torch.from_numpy(proprio[s:e]).unsqueeze(0).to(device)
            if args.model == "force_mlp":
                _, fp = model(vis, fo, pr)
                out = force_to_events(fp, args.task)
            elif args.model == "rule_detector":
                ve = torch.from_numpy(vel[s:e]).unsqueeze(0).to(device)
                out = force_to_events(fo, args.task, velocity=ve)
            else:
                logits, _ = model(vis, fo, pr)
                out = logits.argmax(dim=-1)
            preds.append(out[0].cpu().numpy())
        pred = np.concatenate(preds)

        m = event_metrics(gt, pred, TRANSITIONS[args.task], args.tolerance)
        frame_correct += int((pred == gt).sum())
        frame_total += int(len(gt))
        aggr["gt_events"] += m["gt_events"]
        aggr["pred_events"] += m["pred_events"]
        aggr["matched_gt"] += int(m["recall"] * m["gt_events"])
        aggr["matched_pd"] += int(m["precision"] * m["pred_events"])
        aggr["te"].extend(m["time_errors"])

    n_gt = aggr["gt_events"]
    n_pd = aggr["pred_events"]
    rec = aggr["matched_gt"] / n_gt if n_gt else 1.0
    pre = aggr["matched_pd"] / n_pd if n_pd else 1.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    te = np.array(aggr["te"]) if aggr["te"] else np.array([])

    summary = {
        "task": args.task, "model": args.model, "seed": args.seed,
        "num_test_trajectories": len(test_files),
        "frame_accuracy": frame_correct / frame_total if frame_total else 0.0,
        "event_f1": float(f1), "event_precision": float(pre), "event_recall": float(rec),
        "gt_events_total": int(n_gt), "pred_events_total": int(n_pd),
        "time_error_median_frames": float(np.median(te)) if len(te) else None,
        "time_error_max_frames": float(te.max()) if len(te) else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.checkpoint).parent / "event_metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

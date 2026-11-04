"""
标注质量报告脚本 (2026-11 新增, 对应《实验规划》验收标准)
对已采集的 .npz 轨迹批量评估"自动标注 vs GT"的:
  1. 帧级状态准确率 (参考指标)
  2. 事件级 F1 / Precision / Recall (允许 ±tolerance 帧误差, 默认 5 帧)
  3. 事件时刻误差分布 (GT事件与其最近Auto事件的时间差, 中位数要求 <= 2 帧)

用法:
  python annotation_report.py --task stick_slip --data-dir data/stick_slip
  python annotation_report.py --task all --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 事件转移对 (from_state, to_state) 按任务
TASK_TRANSITIONS = {
    "stick_slip": [(1, 2)],                      # Stick -> Slip
    "peg_in_hole": [(0, 1)],                     # NoContact -> Contact
    "stack_block": [(0, 1), (0, 2)],             # NoContact->Grip (抓取), NoContact->Release (释放完成)
}


def transition_events(states: np.ndarray, transition: tuple[int, int]) -> np.ndarray:
    """返回发生 transition 的帧索引 (事件发生帧 = 目标帧)."""
    frm, to = transition
    idx = np.where((states[:-1] == frm) & (states[1:] == to))[0] + 1
    return idx


def event_metrics(
    gt_states: np.ndarray,
    pred_states: np.ndarray,
    transitions: list[tuple[int, int]],
    tolerance: int = 5,
) -> dict:
    """事件级 F1/P/R (GT事件附近±tolerance帧内是否有Auto事件) + 时间误差分布."""
    gt_events = np.concatenate([transition_events(gt_states, tr) for tr in transitions])
    pred_events = np.concatenate([transition_events(pred_states, tr) for tr in transitions])
    gt_events = np.sort(gt_events)
    pred_events = np.sort(pred_events)

    n_gt = len(gt_events)
    n_pred = len(pred_events)

    if n_gt == 0:
        # GT 无事件: 无假阳则视为完美
        return {
            "gt_events": 0, "pred_events": n_pred,
            "recall": 1.0, "precision": 1.0 if n_pred == 0 else 0.0,
            "f1": 0.0, "time_errors": [],
        }

    # 召回: 每个 GT 事件是否在 ±tolerance 内存在 Auto 事件 (贪心一对一匹配)
    matched_gt = 0
    time_errors = []
    used = np.zeros(len(pred_events), dtype=bool)
    for g in gt_events:
        best = None
        for j, p in enumerate(pred_events):
            if used[j]:
                continue
            d = abs(int(p) - int(g))
            if d <= tolerance:
                if best is None or d < best[1]:
                    best = (j, d)
        if best is not None:
            matched_gt += 1
            used[best[0]] = True
            time_errors.append(best[1])

    recall = matched_gt / n_gt
    # 精确率: Auto 事件中与某 GT 事件 ±tolerance 内匹配的比例
    matched_pred = int(used.sum())
    precision = matched_pred / n_pred if n_pred > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "gt_events": int(n_gt), "pred_events": int(n_pred),
        "recall": float(recall), "precision": float(precision), "f1": float(f1),
        "time_errors": [int(e) for e in time_errors],
    }


def report_dir(data_dir: str | Path, task: str, tolerance: int = 5) -> dict:
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files in {data_dir}")

    frame_accs = []
    aggr = {"gt_events": 0, "pred_events": 0, "matched_gt": 0, "matched_pred": 0,
            "time_errors": []}
    per_file = []

    for f in files:
        d = np.load(f)
        gt = d["contact_states_gt"]
        auto = d["contact_states_auto"]
        m = event_metrics(gt, auto, TASK_TRANSITIONS[task], tolerance)
        frame_accs.append(float((gt == auto).mean()))
        aggr["gt_events"] += m["gt_events"]
        aggr["pred_events"] += m["pred_events"]
        aggr["matched_gt"] += int(m["recall"] * m["gt_events"])
        aggr["matched_pred"] += int(m["precision"] * m["pred_events"])
        aggr["time_errors"].extend(m["time_errors"])
        per_file.append({"file": f.name, "frame_acc": frame_accs[-1], **m})

    n_gt = aggr["gt_events"]
    n_pred = aggr["pred_events"]
    recall = aggr["matched_gt"] / n_gt if n_gt else 1.0
    precision = aggr["matched_pred"] / n_pred if n_pred else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    te = np.array(aggr["time_errors"]) if aggr["time_errors"] else np.array([])

    summary = {
        "task": task, "num_trajectories": len(files), "tolerance_frames": tolerance,
        "frame_accuracy_mean": float(np.mean(frame_accs)),
        "frame_accuracy_min": float(np.min(frame_accs)),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "gt_events_total": int(n_gt),
        "pred_events_total": int(n_pred),
        "time_error_median_frames": float(np.median(te)) if len(te) else None,
        "time_error_max_frames": float(te.max()) if len(te) else None,
        "per_file": per_file,
    }
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation quality report")
    parser.add_argument("--task", type=str, required=True,
                        choices=["stick_slip", "peg_in_hole", "stack_block", "all"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tasks = list(TASK_TRANSITIONS.keys()) if args.task == "all" else [args.task]
    all_summaries = []
    for task in tasks:
        base = Path(args.data_dir)
        s = report_dir(base / task if args.task == "all" else base, task, args.tolerance)
        all_summaries.append(s)
        print(f"[{task}] trajs={s['num_trajectories']} "
              f"frame_acc(mean/min)={s['frame_accuracy_mean']:.4f}/{s['frame_accuracy_min']:.4f} "
              f"event F1={s['event_f1']:.4f} P={s['event_precision']:.4f} R={s['event_recall']:.4f} "
              f"gt_events={s['gt_events_total']} "
              f"time_err(median/max)={s['time_error_median_frames']}/{s['time_error_max_frames']} frames")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {out}")


if __name__ == "__main__":
    main()

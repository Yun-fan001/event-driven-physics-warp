"""
轨迹可视化脚本 (Step 1)
核心功能: 绘制第一条轨迹的力-时间曲线、速度曲线、接触状态，并标记Stick→Slip事件
输出: PNG图片，用于论文插图和调参参考
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="可视化Stick-Slip轨迹")
    parser.add_argument("--traj", type=str, default="data/seed/traj_0000.npz", help="输入的.npz轨迹文件路径")
    parser.add_argument("--output", type=str, default="data/seed/traj_0000_force_events.png",help="输出图片路径")
    args = parser.parse_args()

    # 加载数据
    data = np.load(args.traj)
    forces = data["forces"]
    velocities = data["velocities"]
    gt_states = data["contact_states_gt"]
    auto_states = data["contact_states_auto"]
    slip_events = data["slip_events"].astype(bool)

    t = np.arange(len(forces))
    force_mag = np.linalg.norm(forces[:, :2], axis=1)
    vx = velocities[:, 0]

    # --- 创建三子图 ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 子图1: 力-时间曲线
    axes[0].plot(t, force_mag, label="|F_xy|")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 速度-时间曲线
    axes[1].plot(t, vx, label="v_x", color="green", linewidth=1.5)
    axes[1].axhline(0.0, color="k", linewidth=0.8, linestyle="-")  # 零速线
    # 标注阈值
    axes[1].axhline(0.008, color="orange", linewidth=0.8, linestyle="--", alpha=0.5, label="stick_thresh")
    axes[1].axhline(0.012, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="slip_thresh")
    axes[1].set_ylabel("Velocity (m/s)", fontsize=11)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # 子图3: 接触状态阶梯图
    axes[2].step(t, gt_states, where="post", label="GT state")
    axes[2].step(t, auto_states, where="post", label="Auto state", alpha=0.7)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["NoContact", "Stick", "Slip"])
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

     # --- 标记Stick→Slip事件 (红色虚线) ---
    event_idx = np.where(slip_events)[0]
    for ax in axes[:2]:  # 在力、速度图上画竖线
        for idx in event_idx:
            ax.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=1.2)

    if len(event_idx) > 0:
        axes[0].text(
            0.01, 0.95,
            f"Detected Stick->Slip events: {event_idx.tolist()}",
            transform=axes[0].transAxes,
            va="top",
            fontsize=10,
            color="red",
        )

    fig.suptitle("Stick-Slip Trajectory: Force / Velocity / Contact States")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"✅ 已保存可视化图片: {out}")
    # 如果GT(黑线)和Auto(红线)偏差大，需调整AnnotationConfig阈值
    mismatch = np.sum(gt_states != auto_states)
    if mismatch > 0:
        print(f"⚠️ 发现 {mismatch} 帧标注不一致，建议检查阈值")

if __name__ == "__main__":
    main()
"""
Step 2 统一可视化脚本
支持 Task 1 (Stick-Slip), Task 2 (Peg-in-Hole), Task 3 (Stack-Block)
输出力-时间/速度曲线 + 事件标记、状态阶梯图
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def vis_stick_slip(data: dict, output: str) -> None:
    """Task 1: 力-速度-状态图 + Stick→Slip 事件标记"""
    forces = data["forces"]
    velocities = data["velocities"]
    gt_states = data["contact_states_gt"]
    auto_states = data["contact_states_auto"]
    slip_events = data.get("slip_events", np.zeros(len(gt_states), dtype=bool)).astype(bool)

    t = np.arange(len(forces))
    force_mag = np.linalg.norm(forces[:, :2], axis=1)
    vx = velocities[:, 0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, force_mag, label="|F_xy|")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, vx, label="v_x", color="green", linewidth=1.5)
    axes[1].axhline(0.0, color="k", linewidth=0.8, linestyle="-")
    axes[1].axhline(0.008, color="orange", linewidth=0.8, linestyle="--", alpha=0.5, label="stick_thresh=0.008")
    axes[1].axhline(0.012, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="slip_thresh=0.012")
    axes[1].set_ylabel("Velocity (m/s)", fontsize=11)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, gt_states, where="post", label="GT state", linewidth=1.2)
    axes[2].step(t, auto_states, where="post", label="Auto state", alpha=0.7)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["NoContact", "Stick", "Slip"])
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    event_idx = np.where(slip_events)[0]
    for ax in axes[:2]:
        for idx in event_idx:
            ax.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=1.2)

    fig.suptitle("Stick-Slip: Force / Velocity / Contact States")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"  ✅ 已保存 : {output}")


def vis_peg_in_hole(data: dict, output: str) -> None:
    """Task 2: 力-位置图 + Contact 事件标记"""
    forces = data["forces"]
    peg_z = data.get("peg_z", np.linspace(0.15, 0.05, len(forces)))
    gt_states = data["contact_states_gt"]
    auto_states = data["contact_states_auto"]
    contact_events = data.get("contact_events", np.zeros(len(gt_states), dtype=bool)).astype(bool)

    t = np.arange(len(forces))
    force_mag = np.linalg.norm(forces, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, force_mag, label="|F|", color="blue")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="contact_thresh=1.0N")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, peg_z, label="peg_z", color="green")
    axes[1].set_ylabel("Peg Z (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, gt_states, where="post", label="GT state", linewidth=1.2)
    axes[2].step(t, auto_states, where="post", label="Auto state", alpha=0.7)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["NoContact", "Contact"])
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    event_idx = np.where(contact_events)[0]
    for ax in axes[:2]:
        for idx in event_idx:
            ax.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=1.2)

    fig.suptitle("Peg-in-Hole: Force / Position / Contact States")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"  ✅ 已保存 : {output}")


def vis_stack_block(data: dict, output: str) -> None:
    """Task 3: 力-位置图 + Grip/Release 事件标记"""
    forces = data["forces"]
    gripper_z = data.get("gripper_z", np.arange(len(forces), dtype=np.float32))
    block_pos = data.get("block_pos", np.zeros((len(forces), 3), dtype=np.float32))
    gt_states = data["contact_states_gt"]
    auto_states = data["contact_states_auto"]
    grip_events = data.get("grip_events", np.zeros(len(gt_states), dtype=bool)).astype(bool)
    release_events = data.get("release_events", np.zeros(len(gt_states), dtype=bool)).astype(bool)

    t = np.arange(len(forces))
    force_mag = np.linalg.norm(forces, axis=1)
    bz = block_pos[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, force_mag, label="|F|", color="blue")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, gripper_z, label="gripper_z", color="green")
    axes[1].plot(t, bz, label="block_z", color="orange", alpha=0.7)
    axes[1].set_ylabel("Z Position (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, gt_states, where="post", label="GT state", linewidth=1.2)
    axes[2].step(t, auto_states, where="post", label="Auto state", alpha=0.7)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["NoContact", "Grip", "Release"])
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Grip 事件 (绿色) / Release 事件 (红色)
    for ax in axes[:2]:
        for idx in np.where(grip_events)[0]:
            ax.axvline(idx, color="green", linestyle="--", alpha=0.8, linewidth=1.2)
        for idx in np.where(release_events)[0]:
            ax.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=1.2)

    fig.suptitle("Stack-Block: Force / Position / Contact States")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"  ✅ 已保存 : {output}")


TASK_VIS = {
    "stick_slip": vis_stick_slip,
    "peg_in_hole": vis_peg_in_hole,
    "stack_block": vis_stack_block,
}


def main():
    parser = argparse.ArgumentParser(description="Step 2 轨迹可视化")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_VIS.keys()))
    parser.add_argument("--traj", type=str, required=True,
                        help="输入的 .npz 轨迹文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径 (默认与输入同名 .png)")
    args = parser.parse_args()

    data = np.load(args.traj)
    output = args.output or str(Path(args.traj).with_suffix(".png"))

    vis_fn = TASK_VIS[args.task]
    vis_fn(data, output)

    mismatch = int((data["contact_states_gt"] != data["contact_states_auto"]).sum())
    if mismatch > 0:
        print(f"  ⚠️  {mismatch} 帧标注不一致，建议检查阈值")


if __name__ == "__main__":
    main()
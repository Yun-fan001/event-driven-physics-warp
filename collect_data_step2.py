"""
Step 2 统一数据采集脚本
支持 Task 1(Stick-Slip), Task 2(Peg-in-Hole), Task 3(Stack-Block)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 动态导入对应环境
from env_stick_slip import StickSlipEnv, EpisodeConfig as StickSlipConfig
from env_peg_in_hole import PegInHoleEnv, EpisodeConfig as PegConfig
from env_stack_block import StackBlockEnv, EpisodeConfig as StackConfig
from event_annotation_v2 import (
    AnnotationConfig,
    annotate_stick_slip,
    annotate_peg_in_hole,
    annotate_stack_block,
    compare_with_ground_truth,
)


# ============================================================
# 各任务采集函数
# ============================================================
def collect_stick_slip(env: StickSlipEnv, seed: int, friction_scale: float) -> dict:
    obs, info = env.reset(seed=seed, options={"friction_scale": friction_scale})

    frames, forces, velocities, gt_states, pusher_x = [], [], [], [], []
    action = np.array([1.0], dtype=np.float32)

    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        forces.append(info["force"])
        velocities.append(info["velocity"])
        gt_states.append(info["contact_state"])
        pusher_x.append(info["pusher_x"])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    force_xy = np.stack(forces)[:, :2]
    vel_x = np.stack(velocities)[:, 0]
    pred_states, slip_events = annotate_stick_slip(force_xy, vel_x)
    gt_states_arr = np.array(gt_states, dtype=np.int64)
    metrics = compare_with_ground_truth(pred_states, gt_states_arr)

    return {
        "frames": np.stack(frames).astype(np.uint8),
        "forces": np.stack(forces).astype(np.float32),
        "velocities": np.stack(velocities).astype(np.float32),
        "pusher_x": np.array(pusher_x, dtype=np.float32),
        "contact_states_gt": gt_states_arr,
        "contact_states_auto": pred_states,
        "slip_events": slip_events.astype(np.int8),
        "metrics": metrics,
        "seed": seed,
        "friction_scale": friction_scale,
    }


def collect_peg_in_hole(env: PegInHoleEnv, seed: int, friction_scale: float) -> dict:
    obs, info = env.reset(seed=seed, options={"friction_scale": friction_scale})

    frames, forces, velocities, gt_states, peg_z = [], [], [], [], []
    action = np.array([1.0], dtype=np.float32)  # 最大速度插入

    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        forces.append(info["force"])
        velocities.append(info["velocity"])
        gt_states.append(info["contact_state"])
        peg_z.append(info["peg_z"])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    force = np.stack(forces)
    pred_states, contact_events = annotate_peg_in_hole(force)
    gt_states_arr = np.array(gt_states, dtype=np.int64)
    metrics = compare_with_ground_truth(pred_states, gt_states_arr)

    return {
        "frames": np.stack(frames).astype(np.uint8),
        "forces": force.astype(np.float32),
        "velocities": np.stack(velocities).astype(np.float32),
        "peg_z": np.array(peg_z, dtype=np.float32),
        "contact_states_gt": gt_states_arr,
        "contact_states_auto": pred_states,
        "contact_events": contact_events.astype(np.int8),
        "metrics": metrics,
        "seed": seed,
        "friction_scale": friction_scale,
    }


def collect_stack_block(env: StackBlockEnv, seed: int, friction_scale: float) -> dict:
    obs, info = env.reset(seed=seed, options={"friction_scale": friction_scale})

    frames, forces, velocities, gt_states = [], [], [], []
    gripper_x, gripper_z, block_pos = [], [], []

    action = np.array([1.0], dtype=np.float32)

    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        forces.append(info["force"])
        velocities.append(info["velocity"])
        gt_states.append(info["contact_state"])
        gripper_x.append(info["gripper_x"])
        gripper_z.append(info["gripper_z"])
        block_pos.append(info["block_pos"])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    force = np.stack(forces)
    gz = np.array(gripper_z, dtype=np.float32)
    bp = np.stack(block_pos).astype(np.float32)

    pred_states, grip_events, release_events = annotate_stack_block(force, gz, bp)
    gt_states_arr = np.array(gt_states, dtype=np.int64)
    metrics = compare_with_ground_truth(pred_states, gt_states_arr)

    return {
        "frames": np.stack(frames).astype(np.uint8),
        "forces": force.astype(np.float32),
        "velocities": np.stack(velocities).astype(np.float32),
        "gripper_x": np.array(gripper_x, dtype=np.float32),
        "gripper_z": gz,
        "block_pos": bp,
        "contact_states_gt": gt_states_arr,
        "contact_states_auto": pred_states,
        "grip_events": grip_events.astype(np.int8),
        "release_events": release_events.astype(np.int8),
        "metrics": metrics,
        "seed": seed,
        "friction_scale": friction_scale,
    }


# ============================================================
# 主函数
# ============================================================
TASK_MAP = {
    "stick_slip": (StickSlipEnv, StickSlipConfig, collect_stick_slip),
    "peg_in_hole": (PegInHoleEnv, PegConfig, collect_peg_in_hole),
    "stack_block": (StackBlockEnv, StackConfig, collect_stack_block),
}


def main():
    parser = argparse.ArgumentParser(description="Step 2 统一数据采集")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    EnvClass, ConfigClass, collect_fn = TASK_MAP[args.task]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvClass(config=ConfigClass())
    all_acc = []

    for i in tqdm(range(args.num_episodes), desc=f"采集 {args.task}"):
        seed = args.seed_start + i
        rng = np.random.default_rng(seed)  # ✅ 用当前种子生成独立随机数生成器
        friction_scale = rng.uniform(0.7, 1.3)
        episode = collect_fn(env, seed=seed, friction_scale=friction_scale)

        # 根据任务保存不同字段
        save_dict = {
            "frames": episode["frames"],
            "forces": episode["forces"],
            "velocities": episode["velocities"],
            "contact_states_gt": episode["contact_states_gt"],
            "contact_states_auto": episode["contact_states_auto"],
            "seed": episode["seed"],
            "friction_scale": episode["friction_scale"],
            "metrics_accuracy": episode["metrics"]["state_accuracy"],
        }

        if args.task == "stick_slip":
            save_dict.update({
                "pusher_x": episode["pusher_x"],
                "slip_events": episode["slip_events"],
            })
        elif args.task == "peg_in_hole":
            save_dict.update({
                "peg_z": episode["peg_z"],
                "contact_events": episode["contact_events"],
            })
        elif args.task == "stack_block":
            save_dict.update({
                "gripper_x": episode["gripper_x"],
                "gripper_z": episode["gripper_z"],
                "block_pos": episode["block_pos"],
                "grip_events": episode["grip_events"],
                "release_events": episode["release_events"],
            })

        np.savez_compressed(out_dir / f"traj_{i:04d}.npz", **save_dict)
        all_acc.append(episode["metrics"]["state_accuracy"])

    env.close()

    print(f"\n✅ {args.task}: 已保存 {args.num_episodes} 条轨迹到 {out_dir}")
    print(f"平均自动标注精度: {np.mean(all_acc):.4f}")
    print(f"最低自动标注精度: {np.min(all_acc):.4f}")
    if np.min(all_acc) < 0.98:
        print("⚠️ 警告: 部分轨迹标注精度低于 98%，请调整阈值后重试")


if __name__ == "__main__":
    main()
"""
种子数据采集脚本 (Step 1)
核心功能: 批量运行仿真，采集100条轨迹，自动标注，保存为.npz
输出格式: 视觉帧 + 力觉序列 + 速度 + 事件标签 (与PDF要求一致)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from env_stick_slip import StickSlipEnv, EpisodeConfig
from event_annotation import AnnotationConfig, annotate_contact_states, compare_with_ground_truth

# ============================================================
# 采集单条轨迹
# action固定为[1.0]，即推杆以较快速度持续前推
# 如需随机策略，可改为 action = env._rng.uniform(-1, 1, size=(1,))
# ============================================================
def collect_one_episode(env: StickSlipEnv, seed: int, friction_scale: float) -> dict:
    obs, info = env.reset(seed=seed, options={"friction_scale": friction_scale})

    frames = []
    forces = []
    velocities = []
    gt_states = []
    pusher_x = []

    action = np.array([1.0], dtype=np.float32)# 推杆最大速度

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

    # --- 后处理: 自动标注 ---
    force_xy = np.stack(forces)[:, :2]
    vel_x = np.stack(velocities)[:, 0]

    pred_states, slip_events = annotate_contact_states(force_xy, vel_x)
    gt_states = np.array(gt_states, dtype=np.int64)
    metrics = compare_with_ground_truth(pred_states, gt_states)

    return {
        "frames": np.stack(frames).astype(np.uint8),            # (T, H, W, 3) 视觉数据
        "forces": np.stack(forces).astype(np.float32),         # (T, 3) 三维力
        "velocities": np.stack(velocities).astype(np.float32), # (T, 3) 三维速度
        "pusher_x": np.array(pusher_x, dtype=np.float32),      # (T,) 推杆位置
        "contact_states_gt": gt_states,                        # (T,) 仿真Ground Truth
        "contact_states_auto": pred_states,                    # (T,) 自动标注结果
        "slip_events": slip_events.astype(np.int8),            # (T,) Stick→Slip事件标记
        "metrics": metrics,                                    # 标注精度指标
        "seed": seed,
        "friction_scale": friction_scale,
    }

# ============================================================
# 主函数: 批量采集
#   --num-episodes: 轨迹数量 (Step1要求100条)
#   --output-dir: 输出目录
#   --seed-start: 随机种子起始值 (保证可复现)
# 【修改场景】如需固定摩擦系数(而非随机)，注释掉 random.uniform 行，改为固定值
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/seed")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = StickSlipEnv(config=EpisodeConfig())
    all_acc = []

    for i in tqdm(range(args.num_episodes)):
        seed = args.seed_start + i
        rng = np.random.default_rng(seed)  # ✅ 用当前种子生成独立随机数生成器
        friction_scale = rng.uniform(0.7, 1.3)
        episode = collect_one_episode(env, seed=seed, friction_scale=friction_scale)

        np.savez_compressed(
            out_dir / f"traj_{i:04d}.npz",
            frames=episode["frames"],
            forces=episode["forces"],
            velocities=episode["velocities"],
            pusher_x=episode["pusher_x"],
            contact_states_gt=episode["contact_states_gt"],
            contact_states_auto=episode["contact_states_auto"],
            slip_events=episode["slip_events"],
            seed=episode["seed"],
            friction_scale=episode["friction_scale"],
        )
        all_acc.append(episode["metrics"]["state_accuracy"])

    env.close()

    print(f"\n✅ 已保存 {args.num_episodes} 条轨迹到 {out_dir}")
    print(f"平均自动标注精度: {np.mean(all_acc):.4f}")
    print(f"最低自动标注精度: {np.min(all_acc):.4f}")
    if np.min(all_acc) < 0.98:
        print("⚠️ 警告: 部分轨迹标注精度低于98%，请调整 AnnotationConfig 阈值后重试")


if __name__ == "__main__":
    main()
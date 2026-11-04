"""
多任务数据集加载器
支持 Task 1/2/3 的 .npz 数据文件
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class ContactEventDataset(Dataset):
    """
    统一数据集接口，支持三种任务。
    
    数据格式 (与 collect_data_step2.py 输出一致):
        frames:     (T, H, W, 3)  uint8
        forces:     (T, 3)        float32
        velocities: (T, 3)        float32
        contact_states_gt: (T,)   int64
        task_id:    int           0=stick_slip, 1=peg_in_hole, 2=stack_block
        proprio:    (T, P)        float32  (任务相关的本体感知)
    """

    TASK_NUM_CLASSES = {
        "stick_slip": 3,      # NO_CONTACT, STICK, SLIP
        "peg_in_hole": 2,     # NO_CONTACT, CONTACT
        "stack_block": 3,     # NO_CONTACT, GRIP, RELEASE
    }

    def __init__(
        self,
        data_dir: str | Path,
        task: str,
        transform: Callable | None = None,
        seq_len: int | None = None,  # 若指定，统一截断/填充到该长度
        cache: bool = True,           # 预加载全部 npz 到内存 (200条x3.5MB≈700MB), 大幅加速训练
    ):
        super().__init__()
        self.task = task
        self.num_classes = self.TASK_NUM_CLASSES[task]
        self.transform = transform
        self.seq_len = seq_len

        data_dir = Path(data_dir)
        self.files = sorted(data_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise ValueError(f"在 {data_dir} 中未找到 .npz 文件")

        self._cache = {}
        if cache:
            for f in self.files:
                self._cache[str(f)] = np.load(f)
            print(f"[Dataset] {task}: cached {len(self.files)} trajectories in memory")

        print(f"[Dataset] {task}: 加载 {len(self.files)} 条轨迹")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = self._cache.get(str(self.files[idx]))
        if d is None:
            d = np.load(self.files[idx])

        # 视觉帧: (T', H, W, 3) uint8 → 按 render_interval 展开 → (T, 3, H, W) float32 [0,1]
        frames = d["frames"].astype(np.float32) / 255.0
        ri = int(d.get("render_interval", 1))
        if ri > 1 and frames.shape[0] < d["forces"].shape[0]:
            frames = np.repeat(frames, ri, axis=0)[: d["forces"].shape[0]]
        frames = np.transpose(frames, (0, 3, 1, 2))  # T,H,W,C → T,C,H,W

        forces = d["forces"].astype(np.float32)
        velocities = d["velocities"].astype(np.float32)
        states = d["contact_states_gt"].astype(np.int64)

        # 本体感知: 任务相关
        if self.task == "stick_slip":
            # pusher_x + state 作为 proprio
            proprio = np.stack([
                d["pusher_x"].astype(np.float32),
            ], axis=-1)  # (T, 1)
        elif self.task == "peg_in_hole":
            proprio = np.stack([
                d["peg_z"].astype(np.float32),
            ], axis=-1)  # (T, 1)
        elif self.task == "stack_block":
            gripper_x = d.get("gripper_x", np.zeros(len(states), dtype=np.float32))
            gripper_z = d.get("gripper_z", np.zeros(len(states), dtype=np.float32))
            proprio = np.stack([gripper_x, gripper_z], axis=-1)  # (T, 2)
        else:
            proprio = np.zeros((len(states), 1), dtype=np.float32)

        # 统一序列长度 (截断或填充)
        if self.seq_len is not None:
            T = frames.shape[0]
            if T >= self.seq_len:
                start = np.random.randint(0, T - self.seq_len + 1)
                frames = frames[start : start + self.seq_len]
                forces = forces[start : start + self.seq_len]
                velocities = velocities[start : start + self.seq_len]
                states = states[start : start + self.seq_len]
                proprio = proprio[start : start + self.seq_len]
            else:
                # 填充
                pad = self.seq_len - T
                frames = np.pad(frames, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
                forces = np.pad(forces, ((0, pad), (0, 0)), mode="edge")
                velocities = np.pad(velocities, ((0, pad), (0, 0)), mode="edge")
                # 修复(缺陷7): 标签必须用 constant 填充 -100 (与 ignore_index 对齐),
                # 原 mode="edge" 会复制末帧标签导致填充帧被错误训练
                states = np.pad(states, (0, pad), mode="constant", constant_values=-100)
                proprio = np.pad(proprio, ((0, pad), (0, 0)), mode="edge")

        # 修复(缺陷5): 模型输入去除速度通道 —— 标签(GT)由速度阈值定义,
        # 若速度拼入输入则"标签≈输入的确定性函数", 数据效率实验无区分度。
        # force_input 仅为力 (3维)。速度仍保留在 npz 中用于标注/分析/可视化。
        force_input = forces

        sample = {
            "vision": torch.from_numpy(frames).float(),
            "force": torch.from_numpy(force_input).float(),
            # velocity 仅 RuleDetector 基线使用 (力+速度手调阈值自动标注器),
            # 学习模型 (Event-TCN/Force-MLP/...) 不接收速度通道 (缺陷5修复)
            "velocity": torch.from_numpy(velocities).float(),
            "proprio": torch.from_numpy(proprio).float(),
            "label": torch.from_numpy(states).long(),
            "task": self.task,
        }
        return sample
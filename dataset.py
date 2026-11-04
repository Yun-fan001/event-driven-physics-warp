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
                with np.load(f) as d:
                    # 复制为纯 ndarray 并关闭文件句柄 (可 pickle, 避免 spawn 失败)
                    self._cache[str(f)] = {k: d[k] for k in d.files}
            print(f"[Dataset] {task}: cached {len(self.files)} trajectories in memory")

        print(f"[Dataset] {task}: 加载 {len(self.files)} 条轨迹")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = self._cache.get(str(self.files[idx]))
        if d is None:
            d = np.load(self.files[idx])

        frames_raw = d["frames"]                     # (Traw, H, W, 3) uint8
        ri = int(d.get("render_interval", 1))
        forces = d["forces"].astype(np.float32)
        velocities = d["velocities"].astype(np.float32)
        states = d["contact_states_gt"].astype(np.int64)
        T = forces.shape[0]

        # 本体感知: 任务相关
        if self.task == "stick_slip":
            proprio = np.stack([d["pusher_x"].astype(np.float32)], axis=-1)
        elif self.task == "peg_in_hole":
            proprio = np.stack([d["peg_z"].astype(np.float32)], axis=-1)
        elif self.task == "stack_block":
            proprio = np.stack([d["gripper_x"].astype(np.float32),
                                d["gripper_z"].astype(np.float32)], axis=-1)
        else:
            proprio = np.zeros((T, 1), dtype=np.float32)

        # --- 统一窗口 (先切片后展开, 避免展开全序列的中间开销) ---
        if self.seq_len is not None and T >= self.seq_len:
            start = np.random.randint(0, T - self.seq_len + 1)
            L = self.seq_len
        else:
            start = 0
            L = T

        forces = forces[start:start + L]
        velocities = velocities[start:start + L]
        states = states[start:start + L]
        proprio = proprio[start:start + L]

        # 视觉帧窗口 (原始渲染帧空间对齐)
        if frames_raw.shape[0] < T and ri > 1:
            # 原始渲染帧: 窗口映射到 raw 索引再展开
            raw_start = start // ri
            raw_end = (start + L - 1) // ri + 1
            fr = np.repeat(frames_raw[raw_start:raw_end], ri, axis=0)
            fr = fr[start % ri : start % ri + L]
        else:
            # 已展开格式 (ri=1 或旧数据)
            fr = frames_raw[start:start + L]

        # --- 统一填充到 seq_len ---
        if self.seq_len is not None and L < self.seq_len:
            pad = self.seq_len - L
            fr = np.pad(fr, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
            forces = np.pad(forces, ((0, pad), (0, 0)), mode="edge")
            velocities = np.pad(velocities, ((0, pad), (0, 0)), mode="edge")
            # 修复(缺陷7): 标签必须用 constant 填充 -100 (与 ignore_index 对齐)
            states = np.pad(states, (0, pad), mode="constant", constant_values=-100)
            proprio = np.pad(proprio, ((0, pad), (0, 0)), mode="edge")

        # 视觉: uint8 → (T, 3, H, W) float32 [0,1]
        frames = np.transpose(fr, (0, 3, 1, 2)).astype(np.float32) / 255.0

        # 修复(缺陷5): 模型输入去除速度通道 —— 标签(GT)由速度阈值定义,
        # 若速度拼入输入则"标签≈输入的确定性函数", 数据效率实验无区分度。
        # force_input 仅为力 (3维)。速度仍保留用于标注/分析/可视化。
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
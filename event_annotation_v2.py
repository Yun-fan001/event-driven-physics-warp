"""
事件自动标注模块 V2 (Step 2)
支持 Task 1(Stick-Slip), Task 2(Peg-in-Hole), Task 3(Stack-Block)

状态值定义 (按任务):
  Task 1 (stick_slip):  NO_CONTACT=0 / STICK=1 / SLIP=2
  Task 2 (peg_in_hole): NO_CONTACT=0 / CONTACT=1
  Task 3 (stack_block): NO_CONTACT=0 / GRIP=1  / RELEASE=2
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ============================================================
# 任务级状态常量 (避免 IntEnum 别名冲突 & 跨任务交叉引用)
# ============================================================
# Task 1
NO_CONTACT = 0
STICK = 1
SLIP = 2
# Task 2
CONTACT = 1
# Task 3
GRIP = 1
RELEASE = 2


@dataclass
class AnnotationConfig:
    # Task 1 (Stick-Slip) — 必须与 env_stick_slip.py 保持一致 (0.5)
    stick_vel_threshold: float = 0.01
    slip_vel_threshold: float = 0.02
    force_delta_threshold: float = 2.0
    stick_contact_force_threshold: float = 0.5   # 专门给 Task1

    # Task 2 (Peg-in-Hole)
    peg_contact_force_threshold: float = 1.0

    # Task 3 (Stack-Block)
    grip_force_threshold: float = 2.0
    release_pos_threshold: float = 0.05  # 目标位置容差


# ============================================================
# Task 1: Stick-Slip 自动标注
# 输入: force_xy (T,2), velocity_x (T,)
# 输出: contact_states (T,) ∈ {0,1,2}, slip_events (T,) bool
# ============================================================
def annotate_stick_slip(
    force_xy: np.ndarray,
    velocity_x: np.ndarray,
    cfg: AnnotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or AnnotationConfig()
    T = len(force_xy)
    contact_states = np.zeros(T, dtype=np.int64)
    slip_events = np.zeros(T, dtype=bool)

    force_mag = np.linalg.norm(force_xy, axis=1)
    in_contact = force_mag > cfg.stick_contact_force_threshold

    # 阶段1: 逐帧分类
    for t in range(T):
        if not in_contact[t]:
            contact_states[t] = NO_CONTACT                          # 0
        elif abs(velocity_x[t]) < cfg.stick_vel_threshold:
            contact_states[t] = STICK                                # 1
        else:
            contact_states[t] = SLIP                                 # 2

    # 阶段2: 力跳变 + 速度跳变 双重确认
    delta_f = np.zeros(T)
    delta_f[1:] = force_mag[1:] - force_mag[:-1]

    for t in range(1, T):
        prev = contact_states[t - 1]
        curr = contact_states[t]

        stick_to_slip = (prev == STICK and curr == SLIP)
        force_jump = abs(delta_f[t]) > cfg.force_delta_threshold
        vel_jump = (
            abs(velocity_x[t - 1]) < cfg.stick_vel_threshold
            and abs(velocity_x[t]) > cfg.slip_vel_threshold
        )

        if stick_to_slip or (in_contact[t] and force_jump and vel_jump):
            slip_events[t] = True
            contact_states[t] = SLIP

    return contact_states, slip_events


# ============================================================
# Task 2: Peg-in-Hole 自动标注
# 输入: force (T,3)
# 输出: contact_states (T,) ∈ {0,1}, contact_events (T,) bool
# 事件: contact_events[t]=True 表示 NoContact→Contact 的跳变
# ============================================================
def annotate_peg_in_hole(
    force: np.ndarray,
    cfg: AnnotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or AnnotationConfig()
    T = len(force)
    contact_states = np.zeros(T, dtype=np.int64)
    contact_events = np.zeros(T, dtype=bool)

    force_mag = np.linalg.norm(force, axis=1)
    in_contact = force_mag > cfg.peg_contact_force_threshold

    for t in range(T):
        contact_states[t] = CONTACT if in_contact[t] else NO_CONTACT

    # 检测 NoContact → Contact 的跳变
    for t in range(1, T):
        if contact_states[t - 1] == NO_CONTACT and contact_states[t] == CONTACT:
            contact_events[t] = True

    return contact_states, contact_events


# ============================================================
# Task 3: Stack-Block 自动标注
# 输入: force (T,3), gripper_z (T,), block_pos (T,3)
# 输出: contact_states (T,) ∈ {0,1,2}, grip_events (T,), release_events (T,)
# 注意: GT 来自 env_stack_block.py 的 _get_contact_state()，
#       该函数部分依赖控制器阶段 (_phase) 而非纯物理量。
#       在仿真中可接受，但在论文中需说明局限性。
# ============================================================
def annotate_stack_block(
    force: np.ndarray,
    gripper_z: np.ndarray,
    block_pos: np.ndarray,
    cfg: AnnotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or AnnotationConfig()
    T = len(force)
    contact_states = np.zeros(T, dtype=np.int64)
    grip_events = np.zeros(T, dtype=bool)
    release_events = np.zeros(T, dtype=bool)

    force_mag = np.linalg.norm(force, axis=1)

    for t in range(T):
        fz = force_mag[t]
        gz = gripper_z[t]
        bz = block_pos[t, 2]
        bx = block_pos[t, 0]

        # 1. 判断是否在目标平台上方（候选 RELEASE）
        on_target = (bx > 0.12) and (bz < 0.06)
        gripper_high = gz > 0.10

        # 2. 判断是否被夹持（候选 GRIP）：力大 或  gripper 压得很低（压紧阶段）
        is_gripped = (fz > cfg.grip_force_threshold) or (gz < 0.08 and fz > 1.0)

        # 3. 状态判定（优先级：RELEASE > GRIP > NO_CONTACT）
        if on_target and gripper_high:
            contact_states[t] = RELEASE
        elif is_gripped:
            contact_states[t] = GRIP
        else:
            contact_states[t] = NO_CONTACT

    # 检测事件跳变
    for t in range(1, T):
        prev = contact_states[t - 1]
        curr = contact_states[t]
        if prev == NO_CONTACT and curr == GRIP:
            grip_events[t] = True
        if prev == GRIP and curr == RELEASE:
            release_events[t] = True

    return contact_states, grip_events, release_events


# ============================================================
# 通用评估函数
# ============================================================
def compare_with_ground_truth(
    pred_states: np.ndarray,
    gt_states: np.ndarray,
) -> dict[str, float]:
    assert pred_states.shape == gt_states.shape
    acc = (pred_states == gt_states).mean()
    return {
        "state_accuracy": float(acc),
        "num_frames": int(len(pred_states)),
        "mismatch_frames": int((pred_states != gt_states).sum()),
    }
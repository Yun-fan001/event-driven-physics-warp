"""
事件自动标注模块 (Step 1)
核心功能: 从力/速度序列中自动检测 Stick→Slip 转换事件
与仿真GT对比，验证自动标注精度
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from env_stick_slip import ContactState


@dataclass
class AnnotationConfig:
    force_delta_threshold: float = 2.0# 力跳变临界值判断，漏则小
    slip_vel_threshold: float = 0.02# 滑动临界值判断，漏则小
    stick_vel_threshold: float = 0.01# 黏着临界值判断
    contact_force_threshold: float = 0.5# 接触临界值判断，与 env 里的 contact_force_threshold 保持一致
    min_stick_duration: int = 5# AI加的，最小Stick持续帧数 (当前未使用，预留去噪)，可过滤短时抖动导致的伪Stick段

# ============================================================
# 自动标注接触状态 + 检测Stick→Slip事件
# 输入: 水平力序列 force_xy (T,2), 水平速度 velocity_x (T,)
# 输出: 状态序列 contact_states (T,), 事件标记 slip_events (T,)
# 
# 算法逻辑 (两阶段):
# 1. 逐帧分类 → NoContact / Stick / Slip
# 2. 检测转换 → 找出 Stick→Slip 的跳变时刻
# ============================================================

def annotate_contact_states(
    force_xy: np.ndarray,# 输入: (T, 2) 平面力序列
    velocity_x: np.ndarray,# 输入: (T,)   x 方向速度序列
    cfg: AnnotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or AnnotationConfig()
    """
    输出:
        contact_states: (T,)  每帧状态 {0,1,2}
        slip_events:    (T,)  True 表示该帧发生 Stick→Slip 事件
    """
    T = len(force_xy)
    contact_states = np.zeros(T, dtype=np.int64)
    slip_events = np.zeros(T, dtype=bool)

    # --- 阶段1: 逐帧状态分类 ---
    force_mag = np.linalg.norm(force_xy, axis=1)
    in_contact = force_mag > cfg.contact_force_threshold

    # 2. 逐帧判定 NoContact / Stick / Slip
    for t in range(T):
        if not in_contact[t]:
            contact_states[t] = ContactState.NO_CONTACT
        elif abs(velocity_x[t]) < cfg.stick_vel_threshold:
            contact_states[t] = ContactState.CONTACT_STICK
        else:
            contact_states[t] = ContactState.CONTACT_SLIP

    # 3. 力的一阶差分：检测跳变
    delta_f = np.zeros(T)
    delta_f[1:] = force_mag[1:] - force_mag[:-1]

    # 4. 满足"力跳变+速度跳变"则强制标为 Slip
    for t in range(1, T):
        prev = contact_states[t - 1]
        curr = contact_states[t]

        # 状态从 Stick 变 Slip
        stick_to_slip = (
            prev == ContactState.CONTACT_STICK
            and curr == ContactState.CONTACT_SLIP
        )

        # 力突降（>阈值）且速度突增
        force_jump = abs(delta_f[t]) > cfg.force_delta_threshold
        vel_jump = (
            abs(velocity_x[t - 1]) < cfg.stick_vel_threshold
            and abs(velocity_x[t]) > cfg.slip_vel_threshold
        )

        if stick_to_slip or (in_contact[t] and force_jump and vel_jump):
            slip_events[t] = True
            contact_states[t] = ContactState.CONTACT_SLIP

    return contact_states, slip_events

# ============================================================
# 评估函数: 对比自动标注与仿真Ground Truth
# state_accuracy > 0.98 (即98%以上帧标注正确)
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
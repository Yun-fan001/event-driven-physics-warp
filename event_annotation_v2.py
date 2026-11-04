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
    # Task 1 (Stick-Slip) — 自动标注 = 力+速度 传感器信号 (对应 RuleDetector 基线定义)
    # GT 用运动学真值(几何接触 + 速度阈值 0.01/迟滞退出 0.005), 标注器用不同阈值独立推导
    # 注意: 连续滑移阶段物块速度≈推杆速度(0.015~0.03 m/s),
    #       标注滑移阈值须介于 GT 阈值(0.01) 与最慢推速(0.015) 之间
    stick_vel_threshold: float = 0.008         # 迟滞下界: 低于此速度判定 Stick
    slip_vel_threshold: float = 0.011          # 迟滞上界: 高于此速度判定 Slip
                                                # (GT 进入阈值 0.01; 实测慢起始轨迹穿越滞后 3-9 帧,
                                                #  该滞后为手调阈值的固有局限, 论文中作为学习模型动机)
    stick_contact_force_threshold: float = 0.5   # 接触判定力阈值 (进入)
    stick_contact_exit_ratio: float = 0.6        # 接触退出迟滞: 力 < 0.6*阈值 才退出接触

    # Task 2 (Peg-in-Hole)
    peg_contact_force_threshold: float = 1.0
    peg_contact_exit_ratio: float = 0.6

    # Task 3 (Stack-Block)
    # 抓取保持期接触力 ≈ 吸力(1.5N) - 块重(0.78N) ≈ 0.7N, 压紧期可达 2.6N
    # 阈值取 0.6 覆盖整个保持期, 且与释放期(0N)有清晰分界
    grip_force_threshold: float = 0.6
    release_pos_threshold: float = 0.05  # 目标位置容差


# ============================================================
# Task 1: Stick-Slip 自动标注 (力+速度 + 迟滞带状态机)
# 输入: force_xy (T,2), velocity_x (T,)
# 输出: contact_states (T,) ∈ {0,1,2}, slip_events (T,) bool
# 设计依据: 《实验规划》模型表中 RuleDetector 基线 = "力+速度(手调阈值自动标注器本身)",
#   即标注器用 力+速度 传感器信号。GT 用运动学真值(几何接触 + 速度阈值 0.01),
#   标注器用不同的阈值与迟滞带(0.008/0.012)独立推导, 采集后才与 GT 比对。
# 实机对应: 力=腕部力传感器, 速度=视觉跟踪估计(此处为仿真速度传感器)。
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
    vx = np.abs(velocity_x)

    enter_thr = cfg.stick_contact_force_threshold
    exit_thr = enter_thr * cfg.stick_contact_exit_ratio

    in_contact = False
    state = NO_CONTACT

    for t in range(T):
        # --- 接触迟滞 ---
        if in_contact:
            in_contact = force_mag[t] > exit_thr
        else:
            in_contact = force_mag[t] > enter_thr

        if not in_contact:
            state = NO_CONTACT
            contact_states[t] = state
            continue

        # 进入接触: 初始化为 STICK (修复: 原状态机从 NO_CONTACT 起步导致永不判 Stick)
        if state == NO_CONTACT:
            state = STICK

        # --- Stick/Slip 迟滞 ---
        if state == SLIP:
            if vx[t] < cfg.stick_vel_threshold:
                state = STICK
        else:
            if vx[t] > cfg.slip_vel_threshold:
                state = SLIP
                slip_events[t] = True

        contact_states[t] = state

    return contact_states, slip_events


# ============================================================
# Task 2: Peg-in-Hole 自动标注 (接触迟滞)
# 输入: force (T,3)
# 输出: contact_states (T,) ∈ {0,1}, contact_events (T,) bool
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
    enter_thr = cfg.peg_contact_force_threshold
    exit_thr = enter_thr * cfg.peg_contact_exit_ratio

    in_contact = False
    for t in range(T):
        if in_contact:
            in_contact = force_mag[t] > exit_thr
        else:
            in_contact = force_mag[t] > enter_thr

        contact_states[t] = CONTACT if in_contact else NO_CONTACT
        if t > 0 and contact_states[t - 1] == NO_CONTACT and contact_states[t] == CONTACT:
            contact_events[t] = True

    return contact_states, contact_events


# ============================================================
# Task 3: Stack-Block 自动标注 (传感器信号: 力 + gripper_z + block_pos)
# 输入: force (T,3), gripper_z (T,), block_pos (T,3)
# 输出: contact_states (T,) ∈ {0,1,2}, grip_events (T,), release_events (T,)
# 注意: GT 已改为纯运动学判据 (见 env_stack_block.py), 标注器用传感器信号独立推导
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

    gripped = False  # 抓取锁存 (迟滞防抖动): 保持期接触力在 0.3~0.8N 间波动

    for t in range(T):
        fz = force_mag[t]
        gz = gripper_z[t]
        bz = block_pos[t, 2]
        bx = block_pos[t, 0]

        # 1. 判断是否在目标平台上方（候选 RELEASE）
        # 平台顶面 z=0.04, 物块半高 0.025 → 落位时 block_z=0.065
        on_target = (bx > 0.12) and (bz < 0.075)
        # gripper 明显高于物块顶面 (与 GT 同一边界: 尖端高于顶面 12mm)
        gripper_clear = gz > bz + 0.045

        # 2. 抓取判定 (迟滞锁存)
        # 进入: 力 > 0.6N 或 压紧阶段(gz<0.08 且力>0.5N)
        # 保持: 力 > 0.3N 即维持 GRIP (吸力-块重 ≈ 0.7N, 防止波动误判)
        if gripped:
            gripped = fz > 0.3
        else:
            gripped = (fz > cfg.grip_force_threshold) or (gz < 0.08 and fz > 0.5)

        # 3. 状态判定（优先级：RELEASE > GRIP > NO_CONTACT）
        # RELEASE = 在平台上方 + gripper 已明显离开 + 抓取锁存已释放(力归零)
        if on_target and gripper_clear and not gripped:
            contact_states[t] = RELEASE
        elif gripped:
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
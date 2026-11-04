"""
Slip-Onset 仿真环境 (Step 1 重设计版, 2026-11 修复)
核心功能: 在 MuJoCo 中模拟"推杆慢速三角波推动物块"的 Stick-Slip 现象。

重设计要点 (对应缺陷 1/3/4/5):
  - 慢推: push_speed 约 0.02 m/s (原 0.0008 m/步 = 0.4 m/s 快推, 500 帧仅 1 次转换)
  - 三角波/变幅推杆轨迹: 每周期产生 NO_CONTACT -> STICK -> SLIP -> NO_CONTACT 序列
  - GT 接触判据 = touch 传感器 (几何真值), 与力信号完全解耦
  - GT Stick/Slip = 物块速度阈值 (运动学真值, 仅作标签, 不进模型输入)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# 事件 0=无接触 1=粘着 2=滑动；与 event_annotation.py 共用
class ContactState(IntEnum):
    NO_CONTACT = 0
    CONTACT_STICK = 1
    CONTACT_SLIP = 2


@dataclass
class EpisodeConfig:
    max_steps: int = 3000            # 3000 步 x 0.002s = 6.0 s (约 3 个三角波半周期)
    push_speed: float = 0.02         # 慢推速度 (m/s)。原 0.0008 m/步 = 0.4 m/s 快推, 已废弃
    push_amplitude: float = 0.04     # 三角波推程 (m), 变幅随机化范围 [0.03, 0.05]
    push_gap: float = 0.005          # 初始推杆尖端与物块左面的间隙 (m), 保证 NO_CONTACT 起始段
    friction_scale: float = 1.0
    image_width: int = 128
    image_height: int = 128
    stick_vel_threshold: float = 0.01  # GT: 区分 Stick/Slip 的物块速度阈值 (运动学真值)


class StickSlipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | None = None,
        render_mode: str | None = "rgb_array",
        config: EpisodeConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EpisodeConfig()

        if xml_path is None:
            xml_path = str(Path(__file__).resolve().parent / "assets" / "stick_slip.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self._step_count = 0
        self._rng = np.random.default_rng(seed)
        # --- 预计算传感器/关节地址，加速每步访问 ---
        # MuJoCo中通过名称查找地址较慢，初始化时缓存
        self._vel_adr = self.model.sensor("block_vel").adr[0]
        self._pos_adr = self.model.sensor("block_pos").adr[0]
        self._touch_adr = self.model.sensor("pusher_touch").adr[0]
        self._pusher_qpos_adr = self.model.joint("pusher_x").qposadr[0]
        self._pusher_act_id = self.model.actuator("pusher_pos").id
        self._slider_qpos_adr = self.model.joint("slider_free").qposadr[0]

        # 推杆几何 (尖端世界坐标 = pusher body x + qpos + tip site x)
        self._pusher_body_x = float(self.model.body_pos[self.model.body("pusher").id][0])
        self._tip_site_x = float(self.model.site("pusher_tip").pos[0])
        self._pusher_geom_id = self.model.geom("pusher_geom").id

        # 三角波轨迹内部状态 (reset 时初始化)
        self._dt = float(self.model.opt.timestep)
        self._push_base = 0.0       # 三角波下界
        self._push_top = 0.0        # 三角波上界
        self._push_dir = 1.0        # +1 前进 / -1 后退
        self._pusher_target = 0.0
        self._gt_slip_active = False  # GT 迟滞状态 (防标签抖动)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)

        # --- 多样性随机化 ---
        friction_scale = float(options.get("friction_scale", self.config.friction_scale))
        self.model.geom("floor").friction[0] = 0.9 * friction_scale
        self.model.geom("block").friction[0] = 0.9 * friction_scale

        # 物块初始位置扰动 (±3 mm)
        block_x0 = float(options.get("block_x0", self._rng.uniform(-0.003, 0.003)))
        self.data.qpos[self._slider_qpos_adr] = block_x0
        self.data.qpos[self._slider_qpos_adr + 1] = 0.0
        self.data.qpos[self._slider_qpos_adr + 2] = 0.05
        self.data.qpos[self._slider_qpos_adr + 3 : self._slider_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 推杆起点: 尖端位于物块左面 - gap 处
        #   物块左面(世界) = block_x0 - 0.05
        #   尖端(世界)     = pusher_body_x + qpos + tip_site_x
        #   → qpos_base = block_x0 - 0.05 - gap - pusher_body_x - tip_site_x
        gap = float(options.get("push_gap", self.config.push_gap))
        self._push_base = float(np.clip(
            block_x0 - 0.05 - gap - self._pusher_body_x - self._tip_site_x,
            -0.10, 0.35,
        ))

        # 推杆速度: 3 档随机 (对应规划"速度 3-4 档"), 可用 options 覆盖
        push_speed = float(options.get(
            "push_speed", self._rng.choice([0.015, 0.02, 0.03])))
        self.config.push_speed = push_speed

        # 三角波幅度 (变幅随机化)
        amp = float(options.get("push_amplitude", self._rng.uniform(0.03, 0.05)))
        self._push_top = self._push_base + amp
        self._push_dir = 1.0
        self._pusher_target = self._push_base

        self.data.qpos[self._pusher_qpos_adr] = self._push_base
        self.data.qvel[:] = 0.0
        self.data.ctrl[self._pusher_act_id] = self._push_base
        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # --- 三角波推杆轨迹 (速度恒定, 边界折返) ---
        # action 可作速度倍率 (采集脚本固定传 1.0)
        speed_factor = float(np.clip(action[0], 0.0, 1.0)) if action is not None else 1.0
        speed = speed_factor * self.config.push_speed
        self._pusher_target += self._push_dir * speed * self._dt

        if self._pusher_target >= self._push_top:
            self._pusher_target = self._push_top
            self._push_dir = -1.0
        elif self._pusher_target <= self._push_base:
            self._pusher_target = self._push_base
            self._push_dir = 1.0

        self.data.ctrl[self._pusher_act_id] = self._pusher_target
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        terminated = self._step_count >= self.config.max_steps
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self.renderer.update_scene(self.data, camera="side_cam")
        return self.renderer.render()

    def _get_contact_force(self) -> np.ndarray:
        """真实接触力 (世界系, 3维): 由 data.contact + mj_contactForce 计算。

        修复(缺陷1): 原 <force> 传感器测的是 body 与父级的约束力
        (= 自重 + 位置伺服弹簧力), 非接触力。此处直接提取接触求解器给出的
        推杆-物块接触力, 与真实腕部 F/T 传感器(经零偏标定)语义一致。
        """
        f = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._pusher_geom_id or c.geom2 == self._pusher_geom_id:
                tmp = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, tmp)
                f += tmp[:3]
        return f.astype(np.float32)

    def _get_block_velocity(self) -> np.ndarray:
        return self.data.sensordata[self._vel_adr : self._vel_adr + 3].copy()

    def _get_block_position(self) -> np.ndarray:
        return self.data.sensordata[self._pos_adr : self._pos_adr + 3].copy()

    def _get_touch(self) -> float:
        return float(self.data.sensordata[self._touch_adr])

    # ============================================================
    # GT 接触状态判定 —— 运动学真值, 与传感器信号完全解耦 (缺陷4修复)
    #   接触 = 几何间距判据 (推杆尖端与物块左面距离 < 1mm)
    #   Stick/Slip = 物块速度阈值 (运动学真值, 仅作标签)
    # ============================================================
    def _is_in_contact(self) -> bool:
        # 推杆尖端世界 x = pusher_body_x + qpos + tip_site_x
        tip_x = (self._pusher_body_x
                 + float(self.data.qpos[self._pusher_qpos_adr])
                 + self._tip_site_x)
        block_x = float(self._get_block_position()[0])
        block_left = block_x - 0.05  # 物块半宽 0.05
        block_right = block_x + 0.05
        margin = 0.001
        return (tip_x >= block_left - margin) and (tip_x <= block_right + margin)

    def _get_contact_state(self) -> ContactState:
        if not self._is_in_contact():
            self._gt_slip_active = False
            return ContactState.NO_CONTACT

        vx = abs(self._get_block_velocity()[0])
        # GT 迟滞: 进入 Slip 需速度 > 0.01, 退出需速度 < 0.005 (防标签抖动)
        if self._gt_slip_active:
            if vx < 0.005:
                self._gt_slip_active = False
        else:
            if vx > self.config.stick_vel_threshold:
                self._gt_slip_active = True
        return ContactState.CONTACT_SLIP if self._gt_slip_active else ContactState.CONTACT_STICK

    def _get_obs(self) -> np.ndarray:
        force = self._get_contact_force()
        vel = self._get_block_velocity()
        pusher_x = self.data.qpos[self._pusher_qpos_adr]
        state = float(self._get_contact_state())
        return np.concatenate([force, vel, [pusher_x, state]]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        force = self._get_contact_force()
        vel = self._get_block_velocity()
        state = self._get_contact_state()
        block_pos = self._get_block_position()
        return {
            "force": force.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "contact_state": int(state),
            "contact_state_name": state.name,
            "pusher_x": float(self.data.qpos[self._pusher_qpos_adr]),
            "pusher_target": float(self._pusher_target),
            "block_pos": block_pos.astype(np.float32),
            "in_contact": bool(self._is_in_contact()),
            "touch": self._get_touch(),
        }

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()      # ← 先释放底层资源
            self.renderer = None       # ← 再解除引用

# 【工厂函数】方便 gymnasium.make() 注册；目前直接调用也行
def make_env(**kwargs) -> StickSlipEnv:
    return StickSlipEnv(**kwargs)

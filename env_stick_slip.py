"""
Stick-Slip 仿真环境 (Step 1)
核心功能: 在MuJoCo中模拟"推杆推动物块"的Stick-Slip现象，自动标注接触状态
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
    max_steps: int = 500# 单条最大500 步 × 0.002s = 1.0 
    push_speed: float = 0.0008# 0.0008 m/步；越大越快 Stick 阶段越短越少
    friction_scale: float = 1.0# 摩擦缩放
    image_width: int = 128
    image_height: int = 128# 分辨率
    stick_vel_threshold: float = 0.01# 运动临界值判断
    contact_force_threshold: float = 0.5# 接触临界值判断


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
        self._pusher_target = -0.05
        self._rng = np.random.default_rng(seed)
        # --- 预计算传感器/关节地址，加速每步访问 ---
        # MuJoCo中通过名称查找地址较慢，初始化时缓存
        self._force_adr = self.model.sensor("pusher_force").adr[0]
        self._vel_adr = self.model.sensor("block_vel").adr[0]
        self._pos_adr = self.model.sensor("block_pos").adr[0]
        self._pusher_qpos_adr = self.model.joint("pusher_x").qposadr[0]
        self._pusher_act_id = self.model.actuator("pusher_pos").id
        self._slider_qvel_adr = self.model.joint("slider_free").dofadr[0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        friction_scale = options.get("friction_scale", 1.0) if options else 1.0
        # 随机摩擦系数， 如果换固定摩擦：options={"friction_scale": 1.0}
        self.model.geom("floor").friction[0] = 0.9 * friction_scale
        self.model.geom("block").friction[0] = 0.9 * friction_scale

        self.data.qpos[self._pusher_qpos_adr] = -0.05
        self.data.qvel[:] = 0.0
        self._pusher_target = -0.05
        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        # 想改成可后退 → 去掉 abs()，并调整 clip 下限
        delta = float(action[0]) * self.config.push_speed
        self._pusher_target = np.clip(
            self._pusher_target + delta,
            -0.05,
            0.30,
        )
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
        return self.data.sensordata[self._force_adr : self._force_adr + 3].copy()

    def _get_block_velocity(self) -> np.ndarray:
        return self.data.sensordata[self._vel_adr : self._vel_adr + 3].copy()

    # ============================================================
    # 接触状态判定 —— 论文的Ground Truth来源
    # 逻辑: 无接触→NO_CONTACT; 有接触且速度小→STICK; 有接触且速度大→SLIP
    # stick_vel_threshold是区分Stick/Slip的唯一阈值，需根据物理合理性设定
    # ============================================================
    def _is_in_contact(self) -> bool:
        force = self._get_contact_force()
        return np.linalg.norm(force[:2]) > self.config.contact_force_threshold

    def _get_contact_state(self) -> ContactState:
        if not self._is_in_contact():
            return ContactState.NO_CONTACT

        vx = abs(self._get_block_velocity()[0])
        if vx < self.config.stick_vel_threshold:
            return ContactState.CONTACT_STICK
        return ContactState.CONTACT_SLIP

    def _get_obs(self) -> np.ndarray:
        force = self._get_contact_force()
        vel = self._get_block_velocity()
        pusher_x = self.data.qpos[self._pusher_qpos_adr]
        state = float(self._get_contact_state())
        return np.concatenate([force, vel, [pusher_x, state]]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        # 存放额外信息，不进入观察空间，但用于记录和标注
        force = self._get_contact_force()
        vel = self._get_block_velocity()
        state = self._get_contact_state()
        return {
            "force": force.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "contact_state": int(state),
            "contact_state_name": state.name,
            "pusher_x": float(self.data.qpos[self._pusher_qpos_adr]),
            "in_contact": bool(self._is_in_contact()),
        }

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()      # ← 先释放底层资源
            self.renderer = None       # ← 再解除引用

# 【工厂函数】方便 gymnasium.make() 注册；目前直接调用也行
def make_env(**kwargs) -> StickSlipEnv:
    return StickSlipEnv(**kwargs)

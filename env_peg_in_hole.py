"""
Peg-in-Hole 仿真环境 (Step 2 - Task 2)
核心功能: 圆柱垂直插入孔洞，监测 Contact / NoContact 状态转换
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


class ContactState(IntEnum):
    NO_CONTACT = 0
    CONTACT = 1  # 圆柱底部与 socket 底部接触


@dataclass
class EpisodeConfig:
    max_steps: int = 400          # 单条最大 400 步 × 0.002s = 0.8s
    insert_speed: float = 0.0008  # 每步下降距离 (m/step)
    image_width: int = 128
    image_height: int = 128
    contact_force_threshold: float = 1.0  # 接触力阈值 (N)


class PegInHoleEnv(gym.Env):
    """
    Gymnasium 接口的 Peg-in-Hole 环境。
    Action: 标量 [0,1]，控制插入速度倍率（实际总是向下插入）。
    Observation: [Fx, Fy, Fz, Vx, Vy, Vz, peg_z, contact_state]
    """
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
            xml_path = str(Path(__file__).resolve().parent / "assets" / "peg_in_hole.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        # action: 速度倍率 [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # obs: 力(3) + 速度(3) + peg_z(1) + state(1) = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self._step_count = 0
        self._peg_target_z = 0.15
        self._rng = np.random.default_rng(seed)

        # 预计算传感器/关节地址，加速每步访问
        self._force_adr = self.model.sensor("peg_force").adr[0]
        self._vel_adr = self.model.sensor("peg_vel").adr[0]
        self._peg_qpos_adr = self.model.joint("peg_z").qposadr[0]
        self._peg_act_id = self.model.actuator("peg_pos").id

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # 随机化初始高度和摩擦系数
        init_z = options.get("init_z", 0.15) if options else 0.15
        friction_scale = options.get("friction_scale", 1.0) if options else 1.0

        self.model.geom("socket_bottom").friction[0] = 0.6 * friction_scale
        self.model.geom("peg_geom").friction[0] = 0.6 * friction_scale

        self.data.qpos[self._peg_qpos_adr] = init_z
        self.data.qvel[:] = 0.0
        self._peg_target_z = init_z
        self._step_count = 0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # action 控制速度倍率，实际运动方向固定为向下插入
        speed = 0.5 + 0.5 * float(np.clip(action[0], 0.0, 1.0))
        delta = speed * self.config.insert_speed
        self._peg_target_z = np.clip(self._peg_target_z - delta, 0.02, 0.20)
        self.data.ctrl[self._peg_act_id] = self._peg_target_z

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

    def _get_peg_velocity(self) -> np.ndarray:
        return self.data.sensordata[self._vel_adr : self._vel_adr + 3].copy()

    # ============================================================
    # 接触状态判定 —— 论文 Ground Truth
    # 逻辑: 接触力模值超过阈值 → CONTACT，否则 NO_CONTACT
    # ============================================================
    def _is_in_contact(self) -> bool:
        force = self._get_contact_force()
        return np.linalg.norm(force) > self.config.contact_force_threshold

    def _get_contact_state(self) -> ContactState:
        if self._is_in_contact():
            return ContactState.CONTACT
        return ContactState.NO_CONTACT

    def _get_obs(self) -> np.ndarray:
        force = self._get_contact_force()
        vel = self._get_peg_velocity()
        peg_z = self.data.qpos[self._peg_qpos_adr]
        state = float(self._get_contact_state())
        return np.concatenate([force, vel, [peg_z, state]]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        force = self._get_contact_force()
        vel = self._get_peg_velocity()
        state = self._get_contact_state()
        return {
            "force": force.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "contact_state": int(state),
            "contact_state_name": state.name,
            "peg_z": float(self.data.qpos[self._peg_qpos_adr]),
            "in_contact": bool(self._is_in_contact()),
        }

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_env(**kwargs) -> PegInHoleEnv:
    return PegInHoleEnv(**kwargs)
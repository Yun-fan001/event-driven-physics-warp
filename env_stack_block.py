"""
Stack Block 仿真环境 (Step 2 - Task 3)
核心功能: 吸盘式抓取方块 → 平移 → 放置到目标平台，监测 Grip / Release 事件
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
    NO_CONTACT = 0      # 初始: 方块在地面，gripper 未接触
    CONTACT_GRIP = 1    # Gripper 压紧并提起方块（抓取中）
    CONTACT_RELEASE = 2 # 方块已放置到目标平台，gripper 已离开


@dataclass
class EpisodeConfig:
    max_steps: int = 600          # 抓取-平移-放置需要更多步数
    move_speed: float = 0.0008    # 每步移动距离 (m/step)
    image_width: int = 128
    image_height: int = 128
    grip_force_threshold: float = 2.0   # 判定"夹住"的力阈值 (N)
    release_height: float = 0.12        # 提起后保持的高度


class StackBlockEnv(gym.Env):
    """
    Gymnasium 接口的 Stack-Block 环境。
    Action: 标量 [0,1]，控制整个抓取-放置序列的速度倍率。
    内部通过阶段状态机(phase)自动执行: 下降→压紧→上升→平移→下降放置→上升离开
    Observation: [Fx, Fy, Fz, Vx, Vy, Vz, gripper_x, gripper_z, block_z, state]
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
            xml_path = str(Path(__file__).resolve().parent / "assets" / "stack_block.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # obs: 力(3) + 速度(3) + gripper_x(1) + gripper_z(1) + block_z(1) + state(1) = 11
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._step_count = 0
        self._phase = 0          # 内部阶段: 0~5
        self._phase_counter = 0  # 当前阶段已持续步数
        self._rng = np.random.default_rng(seed)

        # 预计算地址
        self._force_adr = self.model.sensor("gripper_force").adr[0]
        self._vel_adr = self.model.sensor("block_vel").adr[0]
        self._block_pos_adr = self.model.sensor("block_pos").adr[0]
        self._gripper_x_act = self.model.actuator("gripper_x_pos").id
        self._gripper_z_act = self.model.actuator("gripper_z_pos").id
        self._gripper_x_qpos = self.model.joint("gripper_x").qposadr[0]
        self._gripper_z_qpos = self.model.joint("gripper_z").qposadr[0]

        # 目标位置
        self._gripper_target_x = 0.0
        self._gripper_target_z = 0.15

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        friction_scale = options.get("friction_scale", 1.0) if options else 1.0
        # 随机化摩擦，影响"吸起"成功率
        self.model.geom("gripper_geom").friction[0] = 1.5 * friction_scale
        self.model.geom("block_geom").friction[0] = 1.5 * friction_scale

        # 重置方块位置和姿态（轻微随机扰动，增加多样性）
        block_qpos = self.model.joint("block_free").qposadr[0]
        self.data.qpos[block_qpos : block_qpos + 3] = [
            self._rng.uniform(-0.01, 0.01),
            self._rng.uniform(-0.01, 0.01),
            0.025,
        ]
        self.data.qpos[block_qpos + 3 : block_qpos + 7] = [1.0, 0.0, 0.0, 0.0]  # 四元数
        self.data.qvel[:] = 0.0

        # 重置 gripper
        self.data.qpos[self._gripper_x_qpos] = 0.0
        self.data.qpos[self._gripper_z_qpos] = 0.15
        self._gripper_target_x = 0.0
        self._gripper_target_z = 0.15

        self._step_count = 0
        self._phase = 0
        self._phase_counter = 0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        speed = 0.5 + 0.5 * float(np.clip(action[0], 0.0, 1.0))
        delta = speed * self.config.move_speed

        # ============================================================
        # 阶段状态机: 自动执行抓取-放置序列
        # Phase 0: 下降接触方块
        # Phase 1: 压紧 (保持低位若干步，产生足够正压力)
        # Phase 2: 上升提起
        # Phase 3: 平移到目标平台上方
        # Phase 4: 下降放置
        # Phase 5: 上升离开 (结束)
        # ============================================================
        if self._phase == 0:
            self._gripper_target_z = np.clip(self._gripper_target_z - delta, 0.055, 0.20)
            if self._gripper_target_z <= 0.058:
                self._phase = 1
                self._phase_counter = 0

        elif self._phase == 1:
            self._phase_counter += 1
            # 保持压紧位置，让物理稳定
            self._gripper_target_z = 0.055
            if self._phase_counter >= 30:  # 压紧 30 步 (60ms)
                self._phase = 2
                self._phase_counter = 0

        elif self._phase == 2:
            self._gripper_target_z = np.clip(self._gripper_target_z + delta, 0.04, 0.20)
            if self._gripper_target_z >= self.config.release_height:
                self._phase = 3
                self._phase_counter = 0

        elif self._phase == 3:
            self._gripper_target_x = np.clip(self._gripper_target_x + delta, -0.05, 0.15)
            if self._gripper_target_x >= 0.14:
                self._phase = 4
                self._phase_counter = 0

        elif self._phase == 4:
            self._gripper_target_z = np.clip(self._gripper_target_z - delta, 0.055, 0.20)
            if self._gripper_target_z <= 0.065:
                self._phase = 5
                self._phase_counter = 0

        elif self._phase == 5:
            self._gripper_target_z = np.clip(self._gripper_target_z + delta, 0.04, 0.20)

        self.data.ctrl[self._gripper_x_act] = self._gripper_target_x
        self.data.ctrl[self._gripper_z_act] = self._gripper_target_z

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

    def _get_gripper_force(self) -> np.ndarray:
        return self.data.sensordata[self._force_adr : self._force_adr + 3].copy()

    def _get_block_velocity(self) -> np.ndarray:
        return self.data.sensordata[self._vel_adr : self._vel_adr + 3].copy()

    def _get_block_position(self) -> np.ndarray:
        return self.data.sensordata[self._block_pos_adr : self._block_pos_adr + 3].copy()

    # ============================================================
    # 接触状态判定 —— 论文 Ground Truth
    # 混合策略: 以阶段状态机为主，物理量为辅
    #   NO_CONTACT: 初始等待阶段 (phase 0 早期) 或 最终释放后 (phase 5 且 gripper 已远离)
    #   CONTACT_GRIP: phase 1~3 (压紧、提起、平移)
    #   CONTACT_RELEASE: phase 4~5 且方块已接近目标平台
    # ============================================================
    def _get_contact_state(self) -> ContactState:
        force = self._get_gripper_force()
        force_mag = np.linalg.norm(force)
        block_pos = self._get_block_position()
        block_z = block_pos[2]
        gripper_z = self.data.qpos[self._gripper_z_qpos]

        # Phase 1~3: 压紧、提起、平移 → 必定为 GRIP
        if self._phase in (1, 2, 3):
            return ContactState.CONTACT_GRIP

        # Phase 4~5: 下降放置 / 上升离开
        if self._phase in (4, 5):
            # 如果 gripper 已上升到足够高，且方块在平台附近 → RELEASE
            if gripper_z > 0.10 and block_z < 0.06 and abs(block_pos[0] - 0.15) < 0.05:
                return ContactState.CONTACT_RELEASE
            # 否则仍在放置过程中，视为 GRIP
            if force_mag > self.config.grip_force_threshold:
                return ContactState.CONTACT_GRIP
            return ContactState.CONTACT_RELEASE

        # Phase 0: 下降中
        if force_mag > self.config.grip_force_threshold and block_z > 0.04:
            # 已经产生足够接触力，提前视为 GRIP
            return ContactState.CONTACT_GRIP

        return ContactState.NO_CONTACT

    def _get_obs(self) -> np.ndarray:
        force = self._get_gripper_force()
        vel = self._get_block_velocity()
        gripper_x = self.data.qpos[self._gripper_x_qpos]
        gripper_z = self.data.qpos[self._gripper_z_qpos]
        block_z = self._get_block_position()[2]
        state = float(self._get_contact_state())
        return np.concatenate([force, vel, [gripper_x, gripper_z, block_z, state]]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        force = self._get_gripper_force()
        vel = self._get_block_velocity()
        state = self._get_contact_state()
        block_pos = self._get_block_position()
        return {
            "force": force.astype(np.float32),
            "velocity": vel.astype(np.float32),
            "contact_state": int(state),
            "contact_state_name": state.name,
            "gripper_x": float(self.data.qpos[self._gripper_x_qpos]),
            "gripper_z": float(self.data.qpos[self._gripper_z_qpos]),
            "block_pos": block_pos.astype(np.float32),
            "phase": self._phase,
        }

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_env(**kwargs) -> StackBlockEnv:
    return StackBlockEnv(**kwargs)
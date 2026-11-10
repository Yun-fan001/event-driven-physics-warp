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
    max_steps: int = 1200         # 抓取-平移-放置需要更多步数 (慢速 0.2 m/s)
    move_speed: float = 0.0004    # 每步移动距离 (m/step) = 0.2 m/s
    image_width: int = 128
    image_height: int = 128
    grip_force_threshold: float = 2.0   # 自动标注用的力阈值参考 (GT 不再使用)
    release_height: float = 0.12        # 提起后保持的高度
    place_height: float = 0.098         # 放置下降停止高度 (物块落平台后 gripper 停止, 避免压穿)
    suction_force: float = 1.5          # 吸盘吸力 (N), 作用于 block_z, 模拟真空吸盘
                                        # (块重 0.08kg×9.81≈0.78N, 1.5N 有足够裕量且不过冲)


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
        self._vel_adr = self.model.sensor("block_vel").adr[0]
        self._block_pos_adr = self.model.sensor("block_pos").adr[0]
        self._gripper_x_act = self.model.actuator("gripper_x_pos").id
        self._gripper_z_act = self.model.actuator("gripper_z_pos").id
        self._suction_act = self.model.actuator("suction").id
        self._gripper_x_qpos = self.model.joint("gripper_x").qposadr[0]
        self._gripper_z_qpos = self.model.joint("gripper_z").qposadr[0]
        self._block_x_qpos = self.model.joint("block_x").qposadr[0]
        self._block_z_qpos = self.model.joint("block_z").qposadr[0]
        self._gripper_geom_id = self.model.geom("gripper_geom").id

        # 目标位置
        self._gripper_target_x = 0.0
        self._gripper_target_z = 0.15

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        friction_scale = options.get("friction_scale", 1.0) if options else 1.0
        # 随机化摩擦
        self.model.geom("gripper_geom").friction[0] = 1.5 * friction_scale
        self.model.geom("block_geom").friction[0] = 1.5 * friction_scale

        # 重置方块 (x/z 滑块, 轻微位置扰动)
        self.data.qpos[self._block_x_qpos] = float(self._rng.uniform(-0.01, 0.01))
        self.data.qpos[self._block_z_qpos] = 0.025
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
            # 下降到放置高度: 物块落在平台(block_z=0.065)时 gripper 尖端在 0.065+0.033=0.098
            self._gripper_target_z = np.clip(self._gripper_target_z - delta, 0.055, 0.20)
            if self._gripper_target_z <= self.config.place_height:
                self._phase = 5
                self._phase_counter = 0

        elif self._phase == 5:
            self._gripper_target_z = np.clip(self._gripper_target_z + delta, 0.04, 0.20)

        self.data.ctrl[self._gripper_x_act] = self._gripper_target_x
        self.data.ctrl[self._gripper_z_act] = self._gripper_target_z
        # 吸力执行器: Phase 1~4 抓取保持期间开启 (模拟真空吸盘),
        # Phase 0 未接触 / Phase 5 已释放 时关闭。
        if self._phase in (1, 2, 3, 4):
            self.data.ctrl[self._suction_act] = self.config.suction_force
        else:
            self.data.ctrl[self._suction_act] = 0.0

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
        """真实接触力 (世界系, 3维): data.contact + mj_contactForce。

        修复(缺陷1): 原 <force> 传感器测的是 body 与父级约束力
        (= 自重 + 位置伺服弹簧力), 非接触力。
        """
        f = np.zeros(3)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._gripper_geom_id or c.geom2 == self._gripper_geom_id:
                tmp = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, tmp)
                f += tmp[:3]
        return f.astype(np.float32)

    def _get_block_velocity(self) -> np.ndarray:
        return self.data.sensordata[self._vel_adr : self._vel_adr + 3].copy()

    def _get_block_position(self) -> np.ndarray:
        return self.data.sensordata[self._block_pos_adr : self._block_pos_adr + 3].copy()

    # ============================================================
    # 接触状态判定 —— 物理真值 (2026-11 清理: 去 gripper_z 位置泄漏)
    #   GRIP    = 真实接触力 > 0.5 N (夹爪-物块接触, 吸力保持期约 0.7N)
    #   RELEASE = 物块在目标平台 (block_pos 运动学, 不进模型输入) 且接触力 < 0.5 N
    #   NO_CONTACT = 其他
    # (原判据 gripper_z < block_z+0.035 用了 gripper_z, 而 gripper_z 是模型
    #  本体输入 → 标签泄漏。改为接触力后与模型本体通道解耦。)
    # ============================================================
    def _get_contact_state(self) -> ContactState:
        force_mag = float(np.linalg.norm(self._get_gripper_force()))
        block_pos = self._get_block_position()
        block_z = float(block_pos[2])
        block_x = float(block_pos[0])

        on_target = (abs(block_x - 0.15) < 0.03) and (block_z < 0.07)

        if on_target and force_mag < 0.5:
            return ContactState.CONTACT_RELEASE
        if force_mag > 0.5:
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
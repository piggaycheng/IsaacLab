# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence, Tuple

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class FourLegsPMTGAction(ActionTerm):
    cfg: actions_cfg.FourLegsPMTGActionCfg

    ik_action_cfgs: list[actions_cfg.DifferentialInverseKinematicsActionCfg]
    """List of IK configurations for the four legs."""

    def __init__(self, cfg: actions_cfg.FourLegsPMTGActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._phases = torch.zeros(self.num_envs, 4, device=self.device)  # phase for each leg

        self.ik_action_cfgs = cfg.ik_action_cfgs
        self.ik_action_terms = [MyDifferentialInverseKinematicsAction(ik_cfg, env) for ik_cfg in self.ik_action_cfgs]
        for term in self.ik_action_terms:
            term.set_gain(self.cfg.gain)
            term.set_residuals_scale(self.cfg.residuals_scale)

        self.trajectory_generators = [
            HybridFourDimTrajectoryGenerator(
                trajectory_generator_params=self.cfg.trajectory_generator_params,
                phase_offset=phase,
                leg_hip_position=leg_hip_position,
                default_foot_height=foot_default_height,
                default_y_offset=leg_y_offset,
                default_x_offset=leg_x_offset,
            ) for (phase, leg_hip_position, foot_default_height, leg_y_offset, leg_x_offset) in zip(
                self.cfg.phase_offsets,
                self.cfg.leg_hip_positions,
                self.cfg.foot_default_heights,
                self.cfg.leg_y_offsets,
                self.cfg.leg_x_offsets,
            )
        ]

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def phases(self) -> torch.Tensor:
        # 將每個phase轉換成sin及cos形式
        phases_sin = torch.sin(2 * torch.pi * self._phases)
        phases_cos = torch.cos(2 * torch.pi * self._phases)
        phase_sin_cos = torch.stack([phases_sin, phases_cos], dim=-1)
        return phase_sin_cos.flatten(start_dim=1)  # shape (num_envs, 8)

    def process_actions(self, actions: torch.Tensor):
        """16-D action space: first 4 are for trajectory generator, last 12 are joint position residuals."""

        # apply action smoothing
        self._raw_actions[:] = actions
        self._processed_actions = self.cfg.action_smoothing_alpha * actions + (1 - self.cfg.action_smoothing_alpha) * self.processed_actions
        # The first 4 actions are shared trajectory generator parameters
        tg_args = self.processed_actions[:, :4]
        # # FIXME: For debug only, fix the step height to a constant value, others are zero
        # tg_args[:, 0] = 0.0  # 前進速度
        # tg_args[:, 1] = 0.0  # 側向速度
        # tg_args[:, 2] = 0.0  # 轉向角速度
        # tg_args[:, 3] = 0.15  # 固定抬腿高度為 0.1 m

        # Generate foot target positions for all legs
        # The result is a list of tensors, where each tensor is for a leg.
        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(self.trajectory_generators):
            foot_target_position, phase = trajectory_generator.generate(tg_args, self._env.step_dt)
            foot_target_positions.append(foot_target_position)
            # Save the phase for each leg
            self._phases[:, trajectory_generator_idx] = phase

        # Process actions for each leg
        for i, ik_term in enumerate(self.ik_action_terms):
            # Set the residual for the current leg's joints
            residual = self.processed_actions[:, 4 + i * 3 : 7 + i * 3]
            ik_term.set_residuals(residual)
            # Set the foot target position for the current leg
            ik_term.process_actions(foot_target_positions[i])

    def apply_actions(self):
        for term in self.ik_action_terms:
            term.apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


class MyDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    @property
    def ik_controller(self) -> DifferentialIKController:
        return self._ik_controller

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des_full_step = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            delta_joint_pos = joint_pos_des_full_step - joint_pos
            joint_pos_des = joint_pos + self._gain * delta_joint_pos
        else:
            joint_pos_des = joint_pos.clone()

        if self._residuals is not None:
            joint_pos_des += self._residuals * self._residuals_scale

        self._joint_pos_des = joint_pos_des

    def apply_actions(self):
        # apply the desired joint positions
        self._asset.set_joint_position_target(self._joint_pos_des, self._joint_ids)

    def set_residuals(self, residuals: torch.Tensor):
        self._residuals = residuals

    def set_gain(self, gain: float):
        self._gain = gain

    def set_residuals_scale(self, scale: float):
        self._residuals_scale = scale


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器 (批次處理版本)。

    它接收一個 2 維的完整動作張量, shape 為 (batch_size, 4), 4個維度分別是:

    - 前進速度 (stance_vx)
    - 側向速度 (stance_vy)
    - 轉向角速度 (yaw_rotation_rate)
    - 抬腿高度 (step_height)

    步頻 (frequency) 會根據期望速度自動調整，而擺動相占空比 (swing_duty_cycle) 則固定。
    """

    def __init__(self,
                 trajectory_generator_params: actions_cfg.FourLegsPMTGActionCfg.TrajectoryGeneratorCfg,
                 phase_offset: float = 0.0,
                 leg_hip_position: Sequence[float] | torch.Tensor | None = None,
                 default_foot_height: float = 0.0,
                 default_y_offset: float = 0.0,
                 default_x_offset: float = 0.0,
                 # --- 可配置的內部參數 ---
                 base_frequency: float = 1.5,
                 velocity_to_freq_gain: float = 0.8,
                 default_swing_duty_cycle: float = 0.5,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-6,
                 ):
        """
        初始化單腿軌跡生成器。

        Args:
            phase_offset (float): 此腿的初始相位 (0~1)。
            leg_hip_position (np.ndarray | None): shape (3,) 髖關節在機身座標系下的位置。
            base_frequency (float): 基礎步頻 (Hz)。
            velocity_to_freq_gain (float): 速度轉換為額外步頻的增益。
            default_swing_duty_cycle (float): 固定的擺動相占空比。
        """
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.eps = eps
        self.default_foot_height = torch.as_tensor(default_foot_height, dtype=self.dtype, device=self.device)
        self.default_y_offset = torch.as_tensor(default_y_offset, dtype=self.dtype, device=self.device)
        self.default_x_offset = torch.as_tensor(default_x_offset, dtype=self.dtype, device=self.device)
        self.trajectory_generator_params = trajectory_generator_params

        # 相位 (初始化為 scalar tensor, 會在 generate 中根據 batch_size 自動擴展)
        self.phase = torch.tensor(phase_offset % 1.0, device=self.device, dtype=self.dtype)

        if leg_hip_position is None:
            self.leg_hip_position = torch.zeros(3, device=self.device, dtype=self.dtype)
        else:
            self.leg_hip_position = torch.as_tensor(leg_hip_position, dtype=self.dtype, device=self.device)
            assert self.leg_hip_position.shape == (3,), "leg_hip_position 必須是 shape (3,) 的向量"

        # 內部可學 / 可調參數
        self.base_frequency = torch.as_tensor(base_frequency, dtype=self.dtype, device=self.device)
        self.velocity_to_freq_gain = torch.as_tensor(velocity_to_freq_gain, dtype=self.dtype, device=self.device)
        self.default_swing_duty_cycle = torch.as_tensor(default_swing_duty_cycle, dtype=self.dtype, device=self.device)

    def _update_phase(self, frequency: torch.Tensor, dt: float | torch.Tensor):
        """根據頻率與時間步長更新此腿相位 (支援批次處理)。"""
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        # 使用 fmod 保持在 [0,1)
        self.phase = torch.fmod(self.phase + frequency * dt_t, 1.0)

    def generate(self, actions: torch.Tensor, dt: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算單腿足端目標 (x, y, z)，支援批次處理。

        Args:
            actions (torch.Tensor): 來自 policy 的調變參數張量, shape (batch_size, 4)。
            dt (float or torch.Tensor): 單步控制時間 (s)。可以是 scalar 或 shape (batch_size,)。

        Returns:
            torch.Tensor: 目標足端位置, shape (batch_size, 3) -> [[x1, y1, z1], [x2, y2, z2], ...]
        """
        batch_size = actions.shape[0]

        # 檢查並在必要時擴展 self.phase 以匹配 batch_size
        if self.phase.numel() != batch_size:
            # 使用第一個元素的值進行擴展，以保持一致的初始相位
            self.phase = self.phase.expand(batch_size).clone()

        # 1. 讀取 4 維參數並裁剪 (從 N,4 張量中分離)
        actions_on_device = actions.to(self.device, self.dtype)
        stance_vx, stance_vy, yaw_rate, step_height = torch.unbind(actions_on_device, dim=1)

        target_stance_vx = (stance_vx * self.trajectory_generator_params.stance_vx_scale).clamp(-0.8, 0.8)
        target_stance_vy = (stance_vy * self.trajectory_generator_params.stance_vy_scale).clamp(-0.5, 0.5)
        target_yaw_rate = (yaw_rate * self.trajectory_generator_params.yaw_rate_scale).clamp(-1.5, 1.5)
        target_step_height = (step_height * self.trajectory_generator_params.step_height_scale).clamp(0.02, 0.2)

        # 2. 自動推算步頻
        linear_speed = torch.sqrt(target_stance_vx**2 + target_stance_vy**2)
        target_frequency = (self.base_frequency + self.velocity_to_freq_gain * linear_speed).clamp(1.0, 4.0)

        # 3. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 4. 推導步幅
        # 避免除以零
        stance_duration = torch.where(
            target_frequency < self.eps,
            torch.zeros_like(target_frequency),
            target_stance_duty_cycle / target_frequency,
        )

        target_step_length_x = torch.clamp(target_stance_vx * stance_duration, -0.3, 0.3)
        target_step_length_y = torch.clamp(target_stance_vy * stance_duration, -0.3, 0.3)

        # 5. 更新相位並計算軌跡
        self._update_phase(target_frequency, dt)

        # --- 使用 torch.where 取代 if/else 邏輯 ---
        is_swing = self.phase < target_swing_duty_cycle

        # 為 is_swing=True 和 is_swing=False 兩種情況都計算 phase
        phase_in_swing = self.phase / target_swing_duty_cycle
        phase_in_stance = (self.phase - target_swing_duty_cycle) / target_stance_duty_cycle

        # --- Z 軸軌跡 ---
        z_swing_offset = 0.5 * target_step_height * (1 - torch.cos(2 * torch.pi * phase_in_swing))
        z_stance_offset = torch.zeros_like(z_swing_offset)
        z_offset = torch.where(is_swing, z_swing_offset, z_stance_offset)
        # 最終 Z 軸位置 = 預設高度 + 位移
        z = self.default_foot_height + z_offset

        # --- X, Y 軸軌跡 (不含 yaw) ---
        swing_multiplier = -0.5 * torch.cos(torch.pi * phase_in_swing)
        x_swing = target_step_length_x * swing_multiplier
        y_swing = target_step_length_y * swing_multiplier

        stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
        x_stance = target_step_length_x * stance_multiplier
        y_stance = target_step_length_y * stance_multiplier

        x_motion = torch.where(is_swing, x_swing, x_stance)
        y_motion = torch.where(is_swing, y_swing, y_stance)

        x = self.default_x_offset + x_motion
        y = self.default_y_offset + y_motion

        # --- Yaw 效應 (僅在支撐相且頻率不為零時加入) ---
        apply_yaw_effect = (~is_swing) & (target_frequency > self.eps)

        # 預先計算 yaw 效應 (broadcasting 會自動處理)
        # 修正：將位移計算與 stance_duration 關聯，以符合物理模型
        # scale 因子 (1 - 2 * phase_in_stance) 會將位移從 +effect 掃描到 -effect，
        # 總位移是 effect 的兩倍。因此 effect 應為總位移的一半。
        total_displacement_yaw_x = -self.leg_hip_position[1] * target_yaw_rate * stance_duration
        total_displacement_yaw_y = self.leg_hip_position[0] * target_yaw_rate * stance_duration

        yaw_effect_x = 0.5 * total_displacement_yaw_x
        yaw_effect_y = 0.5 * total_displacement_yaw_y

        scale = (1 - 2 * phase_in_stance)

        # 僅在滿足條件時增加 yaw 效應
        x = torch.where(apply_yaw_effect, x + yaw_effect_x * scale, x)
        y = torch.where(apply_yaw_effect, y + yaw_effect_y * scale, y)

        # 將 x, y, z 組合成 (batch_size, 3) 的張量
        foot_pos_rel_hip = torch.stack([x, y, z], dim=1)
        # 加上髖關節在基座標系下的位置，得到相對於基座標系的足端位置
        # 回傳足端位置以及相位，提供給觀測空間
        return (foot_pos_rel_hip + self.leg_hip_position, self.phase)

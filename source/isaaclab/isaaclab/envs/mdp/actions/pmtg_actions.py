# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence, Tuple

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
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
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._phases = torch.zeros(
            self.num_envs, 4, device=self.device
        )  # phase for each leg

        self.ik_action_cfgs = cfg.ik_action_cfgs
        self.ik_action_terms = [
            MyDifferentialInverseKinematicsAction(ik_cfg, env)
            for ik_cfg in self.ik_action_cfgs
        ]
        for term in self.ik_action_terms:
            term.set_gain(self.cfg.gain)
            term.set_residuals_limit(self.cfg.residuals_limit)

        self.trajectory_generators = [
            HybridFourDimTrajectoryGenerator(
                trajectory_generator_params=self.cfg.trajectory_generator_params,
                leg_index=i,
                device=self.device,
            )
            for i in range(4)
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

    @property
    def joint_pos_des(self) -> torch.Tensor:
        """Get the desired joint positions from all IK action terms."""
        joint_pos_list = [term.joint_pos_des for term in self.ik_action_terms]
        return torch.cat(joint_pos_list, dim=1)  # Concatenate along the joint dimension

    def process_actions(self, actions: torch.Tensor):
        """16-D action space: first 4 are for trajectory generator, last 12 are joint position residuals."""

        # 將原始動作使用tanh處理縮放平移
        self._raw_actions[:] = actions

        # 獲取指令
        command_name = self.cfg.command_name
        command_threshold = self.cfg.command_threshold
        commands = self._env.command_manager.get_command(command_name)
        command_norm = torch.norm(commands[:, :3], dim=1)
        is_standing_command = (command_norm < command_threshold).unsqueeze(1)  # Shape: (num_envs, 1)

        # Process trajectory generator arguments
        tg_actions_raw = actions[:, :4]
        # When standing, force trajectory generator actions to zero
        frequency, amp_x, amp_y, amp_z = tg_actions_raw.unbind(dim=1)
        processed_tg_args = torch.stack(
            [
                self.tanh_process(
                    frequency, self.cfg.trajectory_generator_params.frequency_limit
                ),
                self.tanh_process(
                    amp_x, self.cfg.trajectory_generator_params.step_length_x_limit
                ),
                self.tanh_process(
                    amp_y, self.cfg.trajectory_generator_params.step_length_y_limit
                ),
                self.tanh_process(
                    amp_z, self.cfg.trajectory_generator_params.step_height_limit
                ),
            ],
            dim=1,
        )
        # Process residuals (always active)
        residuals_raw = actions[:, 4:]
        processed_residuals = self.tanh_process(
            residuals_raw, self.cfg.residuals_limit
        )
        processed_tg_args = processed_tg_args * (~is_standing_command)
        self._processed_actions = torch.cat(
            [processed_tg_args, processed_residuals], dim=1
        )

        # The first 4 actions are shared trajectory generator parameters
        tg_args = self.processed_actions[:, :4]
        # FIXME: For debug only, fix the step height to a constant value, others are zero
        # tg_args[:, 0] = 2.0  # 頻率
        # tg_args[:, 1] = 0.0  # X振幅
        # tg_args[:, 2] = 0.5  # Y振幅
        # tg_args[:, 3] = 0.15  # Z振幅

        # Generate foot target positions for all legs
        # The result is a list of tensors, where each tensor is for a leg.
        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(
            self.trajectory_generators
        ):
            foot_target_position, phase = trajectory_generator.generate(
                tg_args, self._env.step_dt
            )
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
        self._phases[env_ids] = 0.0
        # Reset the phase for each trajectory generator
        for i in range(4):
            # On the first reset, the phase tensor is a scalar.
            # We need to expand it to the number of environments.
            self.trajectory_generators[i].phase = (
                self.trajectory_generators[i].phase.expand(self.num_envs).clone()
            )
            # Reset the phase for the specified environments
            self.trajectory_generators[i].phase[env_ids] = (
                self.cfg.trajectory_generator_params.phase_offsets[i] % 1.0
            )
            # Reset the IK action terms for the specified environments
            self.ik_action_terms[i].reset(env_ids)

    def tanh_process(self, data, limit):
        # 使用 tanh 將 data 從 (-inf, inf) 映射到 (-1, 1)
        tanh_data = torch.tanh(data)
        # 將 (-1, 1) 的範圍縮放到目標範圍 [min, max]
        data_min, data_max = limit
        data_range = (data_max - data_min) / 2.0
        data_bias = (data_max + data_min) / 2.0
        scaled_data = tanh_data * data_range + data_bias
        return scaled_data


class MyDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    def __init__(
        self,
        cfg: actions_cfg.DifferentialInverseKinematicsActionCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)

        self._joint_pos_des = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )

    @property
    def ik_controller(self) -> DifferentialIKController:
        return self._ik_controller

    @property
    def joint_pos_des(self) -> torch.Tensor:
        return self._joint_pos_des

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des_full_step = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
            # Add residuals to the full step IK solution
            if self._residuals is not None:
                joint_pos_des_full_step += self._residuals
            # Compute the delta and apply gain
            delta_joint_pos = joint_pos_des_full_step - joint_pos
            joint_pos_des = joint_pos + self._gain * delta_joint_pos
        else:
            joint_pos_des = joint_pos.clone()
            if self._residuals is not None:
                joint_pos_des += self._residuals

        self._joint_pos_des = joint_pos_des

    def apply_actions(self):
        # apply the desired joint positions
        self._asset.set_joint_position_target(self._joint_pos_des, self._joint_ids)

    def set_residuals(self, residuals: torch.Tensor):
        self._residuals = residuals

    def set_gain(self, gain: float):
        self._gain = gain

    def set_residuals_limit(self, limit: tuple[float, float]):
        self._residuals_limit = limit

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._joint_pos_des[env_ids] = 0.0


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器 (批次處理版本), 基於 CPG 核心參數。

    它接收一個 4 維的動作張量, shape 為 (batch_size, 4), 4個維度分別是:

    - 頻率 (frequency, f): 步態的頻率 (Hz)。
    - X 軸振幅 (amplitude_x, Ax): 控制前後步長。
    - Y 軸振幅 (amplitude_y, Ay): 控制側向步長。轉向可透過為左右腿設置不同的 Ay 實現。
    - Z 軸振幅 (amplitude_z, Az): 控制抬腿高度。

    擺動相占空比 (swing_duty_cycle) 固定。
    """

    def __init__(
        self,
        trajectory_generator_params: actions_cfg.FourLegsPMTGActionCfg.TrajectoryGeneratorCfg,
        leg_index: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        初始化單腿軌跡生成器。

        Args:
            trajectory_generator_params: 軌跡生成器參數配置
            leg_index (int): 腿的索引 (0=FL, 1=FR, 2=RL, 3=RR)
            device: 計算設備
            dtype: 數據類型
        """
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self.trajectory_generator_params = trajectory_generator_params
        self.leg_index = leg_index

        # 從對應的腿索引取得參數 (視為基礎偏移量 Offset)
        self.default_foot_height = torch.as_tensor(
            trajectory_generator_params.foot_default_heights[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        self.default_y_offset = torch.as_tensor(
            trajectory_generator_params.leg_y_offsets[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        self.default_x_offset = torch.as_tensor(
            trajectory_generator_params.leg_x_offsets[leg_index],
            dtype=self.dtype,
            device=self.device,
        )

        # 相位 (初始化為 scalar tensor, 會在 generate 中根據 batch_size 自動擴展)
        # 這是實現腿間相位差 (Δφ) 的基礎
        self.phase = torch.tensor(
            trajectory_generator_params.phase_offsets[leg_index] % 1.0,
            device=self.device,
            dtype=self.dtype,
        )

        self.leg_hip_position = torch.as_tensor(
            trajectory_generator_params.leg_hip_positions[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        assert self.leg_hip_position.shape == (
            3,
        ), "leg_hip_position 必須是 shape (3,) 的向量"

        # 從配置中取得參數
        self.default_swing_duty_cycle = torch.as_tensor(
            trajectory_generator_params.default_swing_duty_cycle,
            dtype=self.dtype,
            device=self.device,
        )

    def _update_phase(self, frequency: torch.Tensor, dt: float | torch.Tensor):
        """根據頻率與時間步長更新此腿相位 (支援批次處理)。"""
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        # 使用 fmod 保持在 [0,1)
        self.phase = torch.fmod(self.phase + frequency * dt_t, 1.0)

    def generate(
        self, actions: torch.Tensor, dt: float | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算單腿足端目標 (x, y, z)，支援批次處理。

        Args:
            actions (torch.Tensor): 來自 policy 的 CPG 調變參數張量, shape (batch_size, 4)。
                                    分別為 (frequency, amplitude_x, amplitude_y, amplitude_z)
            dt (float or torch.Tensor): 單步控制時間 (s)。可以是 scalar 或 shape (batch_size,)。

        Returns:
            torch.Tensor: 目標足端位置, shape (batch_size, 3) -> [[x1, y1, z1], [x2, y2, z2], ...]
            torch.Tensor: 當前相位, shape (batch_size,)
        """
        batch_size = actions.shape[0]

        # 檢查並在必要時擴展 self.phase 以匹配 batch_size
        if self.phase.numel() != batch_size:
            self.phase = self.phase.expand(batch_size).clone()

        # 1. 使用 tanh 將 CPG 參數從 (-inf, inf) 映射到 (-1, 1)
        actions_on_device = actions.to(self.device, self.dtype)
        frequency, amp_x, amp_y, amp_z = torch.unbind(actions_on_device, dim=1)

        # 2. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 3. 更新相位並計算軌跡
        self._update_phase(frequency, dt)

        # --- 使用 torch.where 取代 if/else 邏輯 ---
        is_swing = self.phase < target_swing_duty_cycle

        # 為 is_swing=True 和 is_swing=False 兩種情況都計算 phase
        phase_in_swing = self.phase / target_swing_duty_cycle
        phase_in_stance = (
            self.phase - target_swing_duty_cycle
        ) / target_stance_duty_cycle

        # --- Z 軸軌跡 (由振幅 Az 控制) ---
        z_swing_offset = 0.5 * amp_z * (1 - torch.cos(2 * torch.pi * phase_in_swing))
        z_stance_offset = torch.zeros_like(z_swing_offset)
        z_offset = torch.where(is_swing, z_swing_offset, z_stance_offset)
        # 最終 Z 軸位置 = 預設高度 (偏移量 O_z) + 軌跡
        z = self.default_foot_height + z_offset

        # --- X, Y 軸軌跡 (由振幅 Ax, Ay 控制) ---
        swing_multiplier = -0.5 * torch.cos(torch.pi * phase_in_swing)
        x_swing = amp_x * swing_multiplier
        y_swing = amp_y * swing_multiplier

        stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
        x_stance = amp_x * stance_multiplier
        y_stance = amp_y * stance_multiplier

        x_motion = torch.where(is_swing, x_swing, x_stance)
        y_motion = torch.where(is_swing, y_swing, y_stance)

        # 最終 X, Y 軸位置 = 預設偏移量 (O_x, O_y) + 軌跡
        x = self.default_x_offset + x_motion
        y = self.default_y_offset + y_motion

        # 注意：轉向 (Yaw) 效果應由上層控制器通過為左右腿提供不同的 `amplitude_y` 來實現，
        # 因此這裡不再單獨處理 `yaw_rate`。

        # 將 x, y, z 組合成 (batch_size, 3) 的張量
        foot_pos_rel_hip = torch.stack([x, y, z], dim=1)
        # 加上髖關節在基座標系下的位置，得到相對於基座標系的足端位置
        # 回傳足端位置以及相位，提供給觀測空間
        return (foot_pos_rel_hip + self.leg_hip_position, self.phase)

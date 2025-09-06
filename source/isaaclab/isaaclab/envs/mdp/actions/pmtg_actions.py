# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

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
        self.ik_action_cfgs = cfg.ik_action_cfgs
        self.ik_action_terms = [MyDifferentialInverseKinematicsAction(ik_cfg, env) for ik_cfg in self.ik_action_cfgs]

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    def process_actions(self, actions: torch.Tensor):
        """16-D action space前4個是軌跡生成器參數, 後12個是關節位置殘差"""
        trajectory_generators = [HybridFourDimTrajectoryGenerator(phase_offset=phase) for phase in self.cfg.phase_offsets]
        for i, trajectory_generator in enumerate(trajectory_generators):
            tg_args = actions[:4]
            foot_target_pos = trajectory_generator.generate(tg_args, self._env.physics_dt)
            self.ik_action_terms[i].process_actions(foot_target_pos)
            self.ik_action_terms[i].set_residuals(actions[4 + i * 3: 7 + i * 3])

    def apply_actions(self):
        for term in self.ik_action_terms:
            term.apply_actions()


class MyDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    @property
    def ik_controller(self) -> DifferentialIKController:
        return self._ik_controller

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        if self._residuals is not None:
            joint_pos_des += self._residuals

        # apply the desired joint positions
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def set_residuals(self, residuals: torch.Tensor):
        self._residuals = residuals


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器。

    它接收一個 4 維的完整動作向量, 4個維度分別是:

    - 前進速度 (stance_vx)
    - 側向速度 (stance_vy)
    - 轉向角速度 (yaw_rotation_rate)
    - 抬腿高度 (step_height)

    步頻 (frequency) 會根據期望速度自動調整，而擺動相占空比 (swing_duty_cycle) 則固定。
    """

    def __init__(self,
                 phase_offset: float = 0.0,
                 leg_hip_position: Sequence[float] | torch.Tensor | None = None,
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

        # 相位 (tensor 以方便未來批量 / device 一致性)
        self.phase = torch.tensor(phase_offset % 1.0, device=self.device, dtype=self.dtype)

        if leg_hip_position is None:
            self.leg_hip_position = torch.zeros(3, device=self.device, dtype=self.dtype)
        else:
            self.leg_hip_position = torch.as_tensor(leg_hip_position, dtype=self.dtype, device=self.device)
            assert self.leg_hip_position.shape == (3,), "leg_hip_position 必須是 shape (3,) 的向量"

        # 內部可學 / 可調參數 (保留為 tensor 以利 autograd)
        self.base_frequency = torch.as_tensor(base_frequency, dtype=self.dtype, device=self.device)
        self.velocity_to_freq_gain = torch.as_tensor(velocity_to_freq_gain, dtype=self.dtype, device=self.device)
        self.default_swing_duty_cycle = torch.as_tensor(default_swing_duty_cycle, dtype=self.dtype, device=self.device)

    def _update_phase(self, frequency: torch.Tensor, dt: float | torch.Tensor):
        """根據頻率與時間步長更新此腿相位 (tensor 版本)。"""
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        # 使用 fmod 保持在 [0,1)
        self.phase = torch.fmod(self.phase + frequency * dt_t, 1.0)

    def generate(self, actions: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        """
        計算單腿足端目標 (x, y, z)。

        Args:
            actions (torch.Tensor): 來自 policy 的 4 維調變參數張量。
            dt (float): 單步控制時間 (s)。

        Returns:
            torch.Tensor: shape (3,) -> [x, y, z]
        """
        # 1. 讀取 4 維參數並裁剪 (支援 dict 或 torch.Tensor 長度=4)
        assert actions.numel() == 4, "若為 Tensor 輸入，需為 shape (4,)"
        stance_vx, stance_vy, yaw_rate, step_height = actions.to(self.device, self.dtype)

        target_stance_vx = stance_vx.clamp(-0.8, 0.8)
        target_stance_vy = stance_vy.clamp(-0.5, 0.5)
        target_yaw_rate = yaw_rate.clamp(-1.5, 1.5)
        target_step_height = step_height.clamp(0.02, 0.15)

        # 2. 自動推算步頻
        linear_speed = torch.sqrt(target_stance_vx**2 + target_stance_vy**2)
        target_frequency = (self.base_frequency + self.velocity_to_freq_gain * linear_speed).clamp(1.0, 4.0)

        # 3. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 4. 推導步幅
        stance_duration = torch.where(
            target_frequency < self.eps,
            torch.zeros((), dtype=self.dtype, device=self.device),
            target_stance_duty_cycle / target_frequency,
        )

        target_step_length_x = torch.clamp(target_stance_vx * stance_duration, -0.3, 0.3)
        target_step_length_y = torch.clamp(target_stance_vy * stance_duration, -0.3, 0.3)

        # 5. 更新相位並計算軌跡
        self._update_phase(target_frequency, dt)
        phase = self.phase.item()  # scalar float for control flow

        if phase < target_swing_duty_cycle.item():
            is_swing = True
            phase_in_swing = phase / target_swing_duty_cycle.item()
        else:
            is_swing = False
            phase_in_stance = (phase - target_swing_duty_cycle.item()) / target_stance_duty_cycle.item()

        z = (
            target_step_height
            * torch.sin(
                torch.pi * torch.as_tensor(phase_in_swing, dtype=self.dtype, device=self.device)
            )
            if is_swing
            else torch.zeros((), dtype=self.dtype, device=self.device)
        )

        if is_swing:
            swing_multiplier = -0.5 * torch.cos(
                torch.pi * torch.as_tensor(phase_in_swing, dtype=self.dtype, device=self.device)
            )
            x = target_step_length_x * swing_multiplier
            y = target_step_length_y * swing_multiplier
        else:
            stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
            x = target_step_length_x * stance_multiplier
            y = target_step_length_y * stance_multiplier

        if (not is_swing) and (target_frequency > self.eps):
            yaw_effect_x = -self.leg_hip_position[1] * target_yaw_rate / target_frequency
            yaw_effect_y = self.leg_hip_position[0] * target_yaw_rate / target_frequency
            scale = (1 - 2 * phase_in_stance)
            x = x + yaw_effect_x * scale
            y = y + yaw_effect_y * scale

        return torch.stack([x, y, z])

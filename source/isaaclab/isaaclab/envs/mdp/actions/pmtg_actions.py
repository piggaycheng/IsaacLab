# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class PMTGJointPositionAction(DifferentialInverseKinematicsAction):
    pass


import numpy as np


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器。

    它接收一個 4 維的完整動作向量，4個維度分別是:

    - 前進速度 (stance_vx)
    - 側向速度 (stance_vy)
    - 轉向角速度 (yaw_rotation_rate)
    - 抬腿高度 (step_height)

    步頻 (frequency) 會根據期望速度自動調整，而擺動相占空比 (swing_duty_cycle) 則固定。
    """

    def __init__(self,
                 phase_offset: float = 0.0,
                 leg_hip_position: np.ndarray | None = None,
                 # --- 可配置的內部參數 ---
                 base_frequency: float = 1.5,
                 velocity_to_freq_gain: float = 0.8,
                 default_swing_duty_cycle: float = 0.5
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
        self.phase = float(phase_offset % 1.0)
        if leg_hip_position is None:
            self.leg_hip_position = np.zeros(3, dtype=float)
        else:
            leg_hip_position = np.asarray(leg_hip_position, dtype=float)
            assert leg_hip_position.shape == (3,), "leg_hip_position 必須是 shape (3,) 的向量"
            self.leg_hip_position = leg_hip_position

        self.base_frequency = base_frequency
        self.velocity_to_freq_gain = velocity_to_freq_gain
        self.default_swing_duty_cycle = default_swing_duty_cycle

        # **重要**: 定義用於生成軌跡的 4 個動作的鍵
        self.TRAJECTORY_ACTION_KEYS = [
            'stance_vx',
            'stance_vy',
            'yaw_rotation_rate',
            'step_height'
        ]

    def _update_phase(self, frequency: float, dt: float):
        """根據頻率與時間步長更新此腿相位。"""
        self.phase = (self.phase + frequency * dt) % 1.0

    def generate(self, actions: dict, dt: float) -> np.ndarray:
        """
        計算單腿足端目標 (x, y, z)。

        Args:
            actions (dict): 來自 policy 的 4 維調變參數字典。
            dt (float): 單步控制時間 (s)。

        Returns:
            np.ndarray: shape (3,) -> [x, y, z]
        """
        # --- 這部分的邏輯與之前的 4D 版本完全相同 ---

        # 1. 讀取 4 維參數並裁剪
        target_stance_vx = float(np.clip(actions['stance_vx'], -0.8, 0.8))
        target_stance_vy = float(np.clip(actions['stance_vy'], -0.5, 0.5))
        target_yaw_rate = float(np.clip(actions.get('yaw_rotation_rate', 0.0), -1.5, 1.5))
        target_step_height = float(np.clip(actions['step_height'], 0.02, 0.15))

        # 2. 自動推算步頻
        linear_speed = np.sqrt(target_stance_vx**2 + target_stance_vy**2)
        target_frequency = self.base_frequency + self.velocity_to_freq_gain * linear_speed
        target_frequency = float(np.clip(target_frequency, 1.0, 4.0))

        # 3. 使用固定的占空比
        target_swing_duty_cycle = self.default_swing_duty_cycle
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle

        # 4. 推導步幅
        if target_frequency < 1e-6:
            stance_duration = 0
        else:
            stance_duration = target_stance_duty_cycle / target_frequency

        target_step_length_x = target_stance_vx * stance_duration
        target_step_length_y = target_stance_vy * stance_duration
        target_step_length_x = float(np.clip(target_step_length_x, -0.3, 0.3))
        target_step_length_y = float(np.clip(target_step_length_y, -0.3, 0.3))

        # 5. 更新相位並計算軌跡
        self._update_phase(target_frequency, dt)
        phase = self.phase

        if phase < target_swing_duty_cycle:
            is_swing = True
            phase_in_swing = phase / target_swing_duty_cycle
        else:
            is_swing = False
            phase_in_stance = (phase - target_swing_duty_cycle) / target_stance_duty_cycle

        z = target_step_height * np.sin(np.pi * phase_in_swing) if is_swing else 0.0

        if is_swing:
            swing_multiplier = -0.5 * np.cos(np.pi * phase_in_swing)
            x = target_step_length_x * swing_multiplier
            y = target_step_length_y * swing_multiplier
        else:
            stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
            x = target_step_length_x * stance_multiplier
            y = target_step_length_y * stance_multiplier

        if not is_swing and target_frequency > 1e-6:
            yaw_effect_x = -self.leg_hip_position[1] * target_yaw_rate / target_frequency
            yaw_effect_y = self.leg_hip_position[0] * target_yaw_rate / target_frequency
            scale = (1 - 2 * phase_in_stance)
            x += yaw_effect_x * scale
            y += yaw_effect_y * scale

        return np.array([x, y, z], dtype=float)

    def unpack_action_array_to_dict(self, action_array: np.ndarray) -> dict:
        """
        將Policy輸出的np.ndarray轉換為帶有鍵的字典。
        """
        return {key: value for key, value in zip(self.TRAJECTORY_ACTION_KEYS, action_array)}

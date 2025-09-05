# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class PMTGJointPositionAction(DifferentialInverseKinematicsAction):
    pass


class TrotTrajectoryGenerator:
    """
    單條腿（用於四足 Trot 步態中的任一腿）之策略模組化軌跡生成器。

    與原始版本不同：此類別僅追蹤並產生「一條腿」的足端相對髖關節座標系的目標位置 (x, y, z)。
    需要的外部管理者可各自為四條腿建立四個實例，並使用不同的 `phase_offset` 與 `leg_hip_position`。
    """

    def __init__(self, phase_offset: float = 0.0, leg_hip_position: np.ndarray | None = None):
        """
        初始化單腿軌跡生成器。

        Args:
            phase_offset (float): 此腿的初始相位 (0~1)。
            leg_hip_position (np.ndarray | None): shape (3,) 髖關節在機身座標系下的位置，用於轉向時的切向速度估算。
        """
        self.phase = float(phase_offset % 1.0)
        if leg_hip_position is None:
            self.leg_hip_position = np.zeros(3, dtype=float)
        else:
            leg_hip_position = np.asarray(leg_hip_position, dtype=float)
            assert leg_hip_position.shape == (3,), "leg_hip_position 必須是 shape (3,) 的向量"
            self.leg_hip_position = leg_hip_position

        # **非常重要**: 定義Policy輸出陣列中每個索引的含義
        # 這個順序必須與你定義RL Action Space時的順序完全一致！
        self.ACTION_KEYS = [
            'frequency',
            'step_height',
            'swing_duty_cycle',
            'stance_vx',
            'stance_vy',
            'yaw_rotation_rate'
        ]

    def _update_phase(self, frequency: float, dt: float):
        """根據頻率與時間步長更新此腿相位。"""
        self.phase = (self.phase + frequency * dt) % 1.0

    def generate(self, actions: dict, dt: float) -> np.ndarray:
        """
        計算單腿足端目標 (x, y, z)。

        Args:
            actions (dict): 來自 policy 的調變參數：
                frequency, step_height, swing_duty_cycle, stance_vx, stance_vy, yaw_rotation_rate(optional)
                說明：改為以支撐相期望腳相對身體的速度 (stance_vx, stance_vy) 來間接決定步幅，
                減少策略直接輸出步幅所需的耦合與尺度推理負擔。
            dt (float): 單步控制時間 (s)。

        Returns:
            np.ndarray: shape (3,) -> [x, y, z]
        """
        # 1. 參數裁剪與讀取
        target_frequency = float(np.clip(actions['frequency'], 1.0, 4.0))  # Hz
        target_step_height = float(np.clip(actions['step_height'], 0.02, 0.15))  # m
        target_swing_duty_cycle = float(np.clip(actions.get('swing_duty_cycle', 0.5), 0.2, 0.8))
        target_stance_duty_cycle = 1.0 - target_swing_duty_cycle
        # Policy 直接輸出的「支撐相中腳相對身體座標系的期望速度」(m/s)
        target_stance_vx = float(np.clip(actions['stance_vx'], -0.8, 0.8))
        target_stance_vy = float(np.clip(actions['stance_vy'], -0.5, 0.5))
        target_yaw_rate = float(actions.get('yaw_rotation_rate', 0.0))  # rad/s (相對簡化)

        # 2. 由支撐相速度 -> 推導步幅 (總位移 L)。
        # 物理近似：步幅 L = v_stance * T_stance，其中 T_stance = stance_duty_cycle / frequency。
        # 原先軌跡公式使用的 step_length 表示總掃掠距離 L，位置線性從 +L/2 -> -L/2。
        # 在該線性段中：x(phase) = L * (0.5 - p)，p∈[0,1]，因此腳相對身體速度 (忽略相位到時間縮放) 為常數。
        # 真正速度：dx/dt = (-L) * (frequency / stance_duty_cycle)。期望其 ≈ target_stance_vx。
        # 反推 L = target_stance_vx * (stance_duty_cycle / frequency)。與 v_stance * T_stance 一致。
        stance_duration = target_stance_duty_cycle / target_frequency  # seconds
        target_step_length_x = target_stance_vx * stance_duration
        target_step_length_y = target_stance_vy * stance_duration
        # 出於穩定性與與舊界面幅值尺度一致，仍然裁剪 (若需要可調整範圍)。
        target_step_length_x = float(np.clip(target_step_length_x, -0.3, 0.3))
        target_step_length_y = float(np.clip(target_step_length_y, -0.3, 0.3))

        # 更新相位
        self._update_phase(target_frequency, dt)
        phase = self.phase

        # 擺動 or 支撐
        if phase < target_swing_duty_cycle:
            is_swing = True
            phase_in_swing = phase / target_swing_duty_cycle
        else:
            is_swing = False
            phase_in_stance = (phase - target_swing_duty_cycle) / target_stance_duty_cycle

        # z 軌跡（抬腿）
        if is_swing:
            z = target_step_height * np.sin(np.pi * phase_in_swing)
        else:
            z = 0.0

        # x, y 軌跡
        if is_swing:
            swing_multiplier = -0.5 * np.cos(np.pi * phase_in_swing)
            x = target_step_length_x * swing_multiplier
            y = target_step_length_y * swing_multiplier
        else:
            stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
            x = target_step_length_x * stance_multiplier
            y = target_step_length_y * stance_multiplier

        # 轉向：僅在支撐相施加切向漂移（簡化）
        if not is_swing and target_frequency > 0.0:
            yaw_effect_x = -self.leg_hip_position[1] * target_yaw_rate / target_frequency
            yaw_effect_y = self.leg_hip_position[0] * target_yaw_rate / target_frequency
            # 使用 (1 - 2*phase_in_stance) 保持與原設計一致的線性掃掠
            scale = (1 - 2 * phase_in_stance)
            x += yaw_effect_x * scale
            y += yaw_effect_y * scale

        return np.array([x, y, z], dtype=float)

    def _unpack_action_array_to_dict(self, action_array: np.ndarray) -> dict:
        """
        將Policy輸出的np.ndarray轉換為帶有鍵的字典。
        """
        return {key: value for key, value in zip(self.ACTION_KEYS, action_array)}

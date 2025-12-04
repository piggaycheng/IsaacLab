# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.physics import IMUSensor

from dataclasses import MISSING, dataclass
from typing import Tuple
import torch
from .ik.pinocchio_differential_ik import InverseKinematicsSolver

ik_joint_names = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


class Go2FlatTerrainPolicy(PolicyController):
    """The Go2 quadruped"""

    def __init__(
        self,
        prim_path: str,
        training_folder: str,
        urdf_path: str,
        urdf_package_dir: str,
        root_path: Optional[str] = None,
        name: str = "go2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            usd_path = assets_root_path + "/Isaac/Robots/Unitree/Go2/go2.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            training_folder + "/exported/policy.pt",
            training_folder + "/params/env.yaml",
        )
        self._policy_counter = 0

        self._obs = None

        self._imu_sensor = IMUSensor(
            prim_path="/World/" + name + "/imu/imu_sensor",
            name="imu_sensor",
            frequency=60,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
        )

        # ------------ pmtg local inference variables ------------
        self._action_cfg = go2_action_config()
        self._command = None
        self._fade_speed = 0.05
        self._trajectory_generators = [
            HybridFourDimTrajectoryGenerator(
                self._action_cfg.trajectory_generator_params, i
            )
            for i in range(4)
        ]
        self._phases = torch.zeros(1, 4)
        self._last_tanh_output = None
        self._joint_pos_ik = None
        self._processed_actions = np.zeros(20)
        self._current_fade = np.zeros(1)
        self._fade_speed = 0.05

        self._ik_solvers = []
        self._foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for foot_name in self._foot_names:
            self._ik_solvers.append(
                InverseKinematicsSolver(
                    urdf_path=urdf_path,
                    ee_name=foot_name,
                )
            )

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(76)
        # Base lin vel
        # obs[:3] = lin_vel_b
        # Base ang vel
        obs[:3] = ang_vel_b
        # Gravity
        obs[3:6] = gravity_b
        # Command
        obs[6:9] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[9:21] = current_joint_pos - self.default_pos
        obs[21:33] = current_joint_vel
        obs[33:53] = self.last_tanh_output
        obs[53:61] = self.phase_sin_cos
        obs[61:73] = self.joint_pos_ik_error
        obs[73:76] = self.lin_acc

        self._obs = obs.copy()

        return obs

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        raw_action = super()._compute_action(obs)
        return self.compute_joint_targets(raw_action)

    def forward(
        self, dt, command, action: Optional[np.ndarray], is_local: Optional[bool] = None
    ):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """

        self._command = command

        if action is not None:
            self.action = action.copy()
            articulation_action = ArticulationAction(joint_positions=self.action)
            self.robot.apply_action(articulation_action)

        if is_local and action is None:
            if self._policy_counter % self._decimation == 0:
                obs = self._compute_observation(command)
                action = self._compute_action(obs)
                self.action = reorder_joints(
                    from_order=ik_joint_names,
                    to_order=self.joint_names,
                    data=action,
                )

            articulation_action = ArticulationAction(joint_positions=self.action)
            self.robot.apply_action(articulation_action)

        self._policy_counter += 1

    @property
    def observation(self):
        return self._obs

    @property
    def decimation(self) -> int:
        return self._decimation

    @property
    def physics_dt(self) -> float:
        return self._dt

    @property
    def joint_names(self) -> list:
        return self.robot.dof_names

    @property
    def current_relative_joint_positions(self) -> np.ndarray:
        return self.robot.get_joint_positions() - self.default_pos

    @property
    def current_joint_velocities(self) -> np.ndarray:
        return self.robot.get_joint_velocities()

    @property
    def current_absolute_joint_positions(self) -> np.ndarray:
        """Returns the absolute joint positions."""
        return self.robot.get_joint_positions()

    @property
    def ang_vel_b(self) -> np.ndarray:
        """Returns the base angular velocity in body frame."""
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)

        return ang_vel_b

    @property
    def gravity_b(self) -> np.ndarray:
        """Returns the gravity vector in body frame."""
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        return gravity_b

    @property
    def imu_values(self) -> dict:
        """Returns the IMU sensor readings."""
        value = self._imu_sensor.get_current_frame()
        return {
            "linear_acceleration": value["lin_acc"],
            "angular_velocity": value["ang_vel"],
            "orientation": value["orientation"],
        }

    @property
    def last_tanh_output(self) -> np.ndarray:
        """Returns the last tanh output from the policy action processing."""
        if self._last_tanh_output is None:
            return np.zeros(20)
        return self._last_tanh_output

    @property
    def phase_sin_cos(self):
        if self._phases is None:
            return np.zeros(8)
        sin_phases = torch.sin(2 * np.pi * self._phases)
        cos_phases = torch.cos(2 * np.pi * self._phases)
        # Stack, flatten, and convert to numpy array of shape (8,)
        return torch.stack([sin_phases, cos_phases], dim=2).view(-1).numpy()

    @property
    def joint_pos_ik_error(self):
        if self._joint_pos_ik is None:
            return np.zeros(12)

        global ik_joint_names

        reorder_names = self.joint_names
        reordered_ik = reorder_joints(
            from_order=ik_joint_names, to_order=reorder_names, data=self._joint_pos_ik
        )

        return reordered_ik - self.current_absolute_joint_positions

    @property
    def lin_acc(self):
        linear_acceleration = self.imu_values["linear_acceleration"]
        return linear_acceleration

    def compute_joint_targets(self, policy_output: np.ndarray) -> np.ndarray:
        """
        Computes the target joint positions based on the action and current joint states.

        Args:
            action (np.ndarray): The action from the policy.
        Returns:
            np.ndarray: The target joint positions.
        """

        # Apply tanh
        actions = np.tanh(policy_output)
        self._last_tanh_output = actions.copy()

        last_cpg_args = self._processed_actions[:8].copy()
        last_residuals = self._processed_actions[8:].copy()

        cpg_actions_raw = actions[:8]
        frequency = cpg_actions_raw[0]
        amp_x = cpg_actions_raw[1]
        amp_y = cpg_actions_raw[2]
        amp_z = cpg_actions_raw[3]
        offset_x = cpg_actions_raw[4]
        offset_y = cpg_actions_raw[5]
        offset_z = cpg_actions_raw[6]
        yaw_param = cpg_actions_raw[7]

        tg_params = self._action_cfg.trajectory_generator_params

        # Mapping
        processed_cpg_args = np.array(
            [
                tanh_post_process(frequency, tg_params.frequency_limit),
                tanh_post_process(amp_x, tg_params.step_length_x_limit),
                tanh_post_process(amp_y, tg_params.step_length_y_limit),
                tanh_post_process(amp_z, tg_params.step_height_limit),
                tanh_post_process(offset_x, tg_params.offset_x_limit),
                tanh_post_process(offset_y, tg_params.offset_y_limit),
                tanh_post_process(offset_z, tg_params.offset_z_limit),
                tanh_post_process(yaw_param, tg_params.yaw_limit),
            ]
        )

        # LPF (Filter)
        processed_cpg_args = (
            self._action_cfg.cpg_lpf_alpha * processed_cpg_args
            + (1 - self._action_cfg.cpg_lpf_alpha) * last_cpg_args
        )

        # Fade Factor
        if self._command is None:
            speed_norm = 0.0
        else:
            cmd_vel = np.array([self._command[0], self._command[1]])
            speed_norm = np.linalg.norm(cmd_vel)

        target_fade = 1.0 if speed_norm > self._action_cfg.command_threshold else 0.0

        diff = target_fade - self._current_fade
        step = np.clip(diff, -self._fade_speed, self._fade_speed)
        self._current_fade += step

        # Apply fade factor to Amps(1-3), Offsets(4-6), Yaw(7)
        params_to_fade = processed_cpg_args[1:8]
        processed_cpg_args[1:8] = params_to_fade * self._current_fade

        # Process residuals
        residuals_raw = actions[8:]
        processed_residuals = tanh_post_process(
            residuals_raw, self._action_cfg.residuals_limit
        )

        # Apply LPF to residuals
        processed_residuals = (
            self._action_cfg.residuals_lpf_alpha * processed_residuals
            + (1 - self._action_cfg.residuals_lpf_alpha) * last_residuals
        )

        self._processed_actions = np.concatenate(
            [processed_cpg_args, processed_residuals]
        )

        tg_args = torch.from_numpy(processed_cpg_args).double().unsqueeze(0)
        # For testing purpose, use fixed args
        tg_args = torch.tensor([[2.0, 0.2, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0]], dtype=torch.double)

        foot_target_positions = []
        for trajectory_generator_idx, trajectory_generator in enumerate(
            self._trajectory_generators
        ):
            foot_target_position, phase = trajectory_generator.generate(
                tg_args, self.physics_dt * self.decimation
            )
            foot_target_positions.append(
                foot_target_position.detach().cpu().numpy().squeeze()
            )
            self._phases[:, trajectory_generator_idx] = phase

        joint_targets = np.zeros(12)
        ik_joint_targets = np.zeros(12)

        # Reorder joints for IK (Isaac Sim order -> URDF order)
        curr_q_urdf = reorder_joints(
            from_order=self.joint_names,
            to_order=ik_joint_names,
            data=self.current_absolute_joint_positions,
        )

        for idx, foot in enumerate(self._foot_names):
            try:
                q_next = self._ik_solvers[idx].compute(
                    q_current=curr_q_urdf,
                    target_pos=foot_target_positions[idx],
                )
                ik_joint_targets[idx * 3 : (idx + 1) * 3] = q_next[
                    idx * 3 : (idx + 1) * 3
                ]
            except Exception as e:
                print(f"IK solver error for {foot}: {e}")
        self._joint_pos_ik = ik_joint_targets.copy()

        for idx, foot in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            processed_residual = processed_residuals[idx * 3 : (idx + 1) * 3]
            joint_targets[idx * 3 : (idx + 1) * 3] = (
                ik_joint_targets[idx * 3 : (idx + 1) * 3] + processed_residual
            )

        # 目前順序是[L1_hip, L1_thigh, L1_calf, L2_hip, ...]
        return ik_joint_targets


@dataclass
class ActionCfg:
    @dataclass
    class TrajectoryGeneratorCfg:
        """Configuration for the trajectory generator used in PMTG."""

        leg_hip_positions: tuple[list[float], list[float], list[float], list[float]] = (
            MISSING  # LF, RF, RL, RR
        )
        """四條腿的髖關節相對於機身的位置, 用於計算轉向效果"""

        default_swing_duty_cycle: float = 0.5
        """Fixed swing duty cycle ratio. Defaults to 0.5."""

        # Frequency limits
        frequency_limit: tuple[float, float] = (1.0, 4.0)
        """Frequency limits (Hz). Defaults to (1.0, 4.0)."""

        # Step length limits
        step_length_x_limit: tuple[float, float] = (-0.4, 0.4)
        """X step length limits (m). Defaults to (-0.4, 0.4)."""

        step_length_y_limit: tuple[float, float] = (-0.2, 0.2)
        """Y step length limits (m). Defaults to (-0.2, 0.2)."""

        step_height_limit: tuple[float, float] = (0.0, 0.2)
        """Step height limits (m). Defaults to (0.0, 0.2)."""

        # Offset limits
        offset_x_limit: tuple[float, float] = (-0.1, 0.1)
        """X offset limits (m). Defaults to (-0.1, 0.1)."""

        offset_y_limit: tuple[float, float] = (-0.1, 0.1)
        """Y offset limits (m). Defaults to (-0.1, 0.1)."""

        offset_z_limit: tuple[float, float] = (-0.1, 0.1)
        """Z offset limits (m). Defaults to (-0.1, 0.1)."""

        # Yaw limit
        yaw_limit: tuple[float, float] = (-1.0, 1.0)
        """Yaw command limits. Defaults to (-1.0, 1.0)."""

        foot_default_heights: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )  # FL, FR, RL, RR
        """預設的腳部高度, 用於計算Z軸位置"""

        default_leg_y_offsets: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )  # FL, FR, RL, RR
        """四條腿的Y軸預設偏移量, 用於計算Y軸位置"""

        default_leg_x_offsets: tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )  # FL, FR, RL, RR
        """四條腿的X軸預設偏移量, 用於計算X軸位置"""

        phase_offsets: tuple[float, float, float, float] = (
            0.0,
            0.5,
            0.5,
            0.0,
        )  # LF, RF, RL, RR
        """四條腿的相位偏移量, 以實現對角步態"""

    trajectory_generator_params: TrajectoryGeneratorCfg = MISSING  # type: ignore

    gain: float = 1.0
    """增益因子, 用於apply_action效果的強度"""

    residuals_limit: tuple[float, float] = (-0.1, 0.1)
    """關節位置殘差的限制範圍, 防止過大的調整"""

    command_name: str = "base_velocity"
    """The name of the command to use for the trajectory generator. Defaults to "base_velocity"."""
    command_threshold: float = 0.1
    """Threshold to consider command as zero command. Defaults to 0.1."""

    cpg_lpf_alpha: float = 0.15
    """The weight for the low-pass filter (LPF). Defaults to 0.15."""

    residuals_lpf_alpha: float = 0.85
    """The weight for the low-pass filter (LPF) applied to joint residuals. Defaults to 0.85."""


class HybridFourDimTrajectoryGenerator:
    """
    單條腿之混合控制軌跡生成器 (批次處理版本), 基於 CPG 核心參數。

    它接收一個 8 維的動作張量, shape 為 (batch_size, 8), 8個維度分別是:

    - 頻率 (frequency, f): 步態的頻率 (Hz)。
    - X 軸振幅 (amplitude_x, Ax): 控制前後步長。
    - Y 軸振幅 (amplitude_y, Ay): 控制側向步長。
    - Z 軸振幅 (amplitude_z, Az): 控制抬腿高度。
    - X 軸偏移 (offset_x, Ox): 控制X軸偏移。
    - Y 軸偏移 (offset_y, Oy): 控制Y軸偏移。
    - Z 軸偏移 (offset_z, Oz): 控制Z軸偏移。
    - 轉向參數 (yaw_param): 控制轉向。

    擺動相占空比 (swing_duty_cycle) 固定。
    """

    def __init__(
        self,
        trajectory_generator_params: ActionCfg.TrajectoryGeneratorCfg,
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
            trajectory_generator_params.default_leg_y_offsets[leg_index],
            dtype=self.dtype,
            device=self.device,
        )
        self.default_x_offset = torch.as_tensor(
            trajectory_generator_params.default_leg_x_offsets[leg_index],
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
            actions (torch.Tensor): 來自 policy 的 CPG 調變參數張量, shape (batch_size, 8)。
                                    分別為 (frequency, amplitude_x, amplitude_y, amplitude_z, offset_x, offset_y, offset_z, yaw_param)
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
        (
            frequency,
            amp_x,
            amp_y,
            amp_z,
            offset_x,
            offset_y,
            offset_z,
            yaw_param,
        ) = torch.unbind(actions_on_device, dim=1)

        # Yaw logic
        turn_gain = 0.15

        # X-axis (Differential Steering)
        diff_x = yaw_param * turn_gain
        # 0: FL, 1: FR, 2: RL, 3: RR
        is_left = (self.leg_index == 0) or (self.leg_index == 2)

        if is_left:
            amp_x_leg = amp_x - diff_x
        else:
            amp_x_leg = amp_x + diff_x

        # Y-axis (Lateral Cornering)
        diff_y = yaw_param * turn_gain
        is_front = (self.leg_index == 0) or (self.leg_index == 1)

        if is_front:
            amp_y_leg = amp_y + diff_y
        else:
            amp_y_leg = amp_y - diff_y

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
        z_motion = torch.where(is_swing, z_swing_offset, z_stance_offset)
        # 最終 Z 軸位置 = 預設高度 (偏移量 O_z) + 軌跡 + offset_z
        z = self.default_foot_height + z_motion + offset_z

        # --- X, Y 軸軌跡 (由振幅 Ax, Ay 控制) ---
        swing_multiplier = -0.5 * torch.cos(torch.pi * phase_in_swing)
        x_swing = amp_x_leg * swing_multiplier
        y_swing = amp_y_leg * swing_multiplier

        stance_multiplier = 0.5 * (1 - 2 * phase_in_stance)
        x_stance = amp_x_leg * stance_multiplier
        y_stance = amp_y_leg * stance_multiplier

        x_motion = torch.where(is_swing, x_swing, x_stance)
        y_motion = torch.where(is_swing, y_swing, y_stance)

        # 最終 X, Y 軸位置 = 預設偏移量 (O_x, O_y) + 軌跡 + offset_x, offset_y
        x = self.default_x_offset + x_motion + offset_x
        y = self.default_y_offset + y_motion + offset_y

        # 注意：轉向 (Yaw) 效果應由上層控制器通過為左右腿提供不同的 `amplitude_y` 來實現，
        # 因此這裡不再單獨處理 `yaw_rate`。

        # 將 x, y, z 組合成 (batch_size, 3) 的張量
        foot_pos_rel_hip = torch.stack([x, y, z], dim=1)
        # 加上髖關節在基座標系下的位置，得到相對於基座標系的足端位置
        # 回傳足端位置以及相位，提供給觀測空間
        return (foot_pos_rel_hip + self.leg_hip_position, self.phase)


def go2_action_config():
    return ActionCfg(
        gain=1.0,
        trajectory_generator_params=ActionCfg.TrajectoryGeneratorCfg(
            leg_hip_positions=(
                [0.1934, 0.0465, 0.0],
                [0.1934, -0.0465, 0.0],
                [-0.1934, 0.0465, 0.0],
                [-0.1934, -0.0465, 0.0],
            ),  # FL, FR, RL, RR
            foot_default_heights=(-0.3, -0.3, -0.32, -0.32),
            default_leg_y_offsets=(0.12, -0.12, 0.12, -0.12),
            default_leg_x_offsets=(0.02, 0.02, -0.05, -0.05),
            step_length_x_limit=(-0.2, 0.2),
            step_length_y_limit=(-0.15, 0.15),
            step_height_limit=(0.0, 0.15),
            offset_x_limit=(-0.03, 0.03),
            offset_y_limit=(-0.02, 0.02),
            offset_z_limit=(-0.02, 0.02),
        ),
    )


def tanh_post_process(data: np.ndarray | float, limit: tuple[float, float]):
    # 將 (-1, 1) 的範圍縮放到目標範圍 [min, max]
    data_min, data_max = limit
    data_range = (data_max - data_min) / 2.0
    data_bias = (data_max + data_min) / 2.0
    scaled_data = data * data_range + data_bias
    return scaled_data


def reorder_joints(
    from_order: list[str], to_order: list[str], data: np.ndarray
) -> np.ndarray:
    map = dict(zip(from_order, data))
    reordered_data = np.array([map[joint_name] for joint_name in to_order])
    return reordered_data

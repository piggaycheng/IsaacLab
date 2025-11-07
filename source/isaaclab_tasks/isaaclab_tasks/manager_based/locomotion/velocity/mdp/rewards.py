# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.envs.mdp.actions.pmtg_actions import FourLegsPMTGAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (
        torch.norm(command[:, :2], dim=1) < command_threshold
    )


def pmtg_joint_residuals_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalizes the L2 norm of the joint position residuals."""
    # PMTG action 的後 12 個維度是 residuals
    action_term = env.action_manager.get_term(action_name)
    pmtg_action_term = cast(FourLegsPMTGAction, action_term)
    residuals = pmtg_action_term.processed_actions[:, 4:]
    return torch.sum(torch.square(residuals), dim=1)


def pmtg_amplitude_residual_ratio_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    command_threshold: float = 0.06,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Rewards a higher ratio of amplitude usage compared to residual usage, using a squared (L2) reward.

    This encourages the policy to rely on the trajectory generator for primary movements,
    rather than simply maximizing amplitudes. The reward is proportional to the square of:
    amp_energy / (amp_energy + residual_energy).
    """
    # 1. Check if the robot is commanded to move
    commands = env.command_manager.get_command(command_name)
    command_norm = torch.norm(commands[:, :3], dim=1)
    is_moving_command = command_norm > command_threshold

    # If not moving, no reward
    if not torch.any(is_moving_command):
        return torch.zeros_like(command_norm)

    # 2. Get amplitudes and residuals from the action term
    action_term = env.action_manager.get_term(action_name)
    pmtg_action_term = cast(FourLegsPMTGAction, action_term)

    # Amplitudes are the 2nd, 3rd, and 4th params of the trajectory generator
    amplitudes = pmtg_action_term.processed_actions[:, 1:4]
    residuals = pmtg_action_term.processed_actions[:, 4:]

    # 3. Calculate the energy (sum of squares) for both components
    amp_energy = torch.sum(torch.square(amplitudes), dim=1)
    residual_energy = torch.sum(torch.square(residuals), dim=1)

    # 4. Calculate the ratio reward
    # This ratio is between 0 and 1
    ratio_reward = amp_energy / (amp_energy + residual_energy + epsilon)

    # 5. Apply the squared reward only when commanded to move
    return torch.square(ratio_reward) * is_moving_command


def conditional_joint_residuals_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "base_velocity",
    yaw_command_threshold: float = 0.25,
    linear_penalty_scale: float = 1.0,
    turn_penalty_scale: float = 0.1,
) -> torch.Tensor:
    """
    Penalizes the L2 norm of joint residuals, applying a smaller penalty during turns.
    """
    commands = env.command_manager.get_command(command_name)
    yaw_command = commands[:, 2]

    action_term = env.action_manager.get_term(action_name)
    pmtg_action_term = cast(FourLegsPMTGAction, action_term)
    residuals = pmtg_action_term.processed_actions[:, 4:]

    residual_penalty = torch.sum(torch.square(residuals), dim=1)

    # Check if the main command is for turning
    is_turning = torch.abs(yaw_command) > yaw_command_threshold

    # Apply a higher penalty for linear motion and a lower one for turning
    penalty_scale = torch.where(is_turning, turn_penalty_scale, linear_penalty_scale)

    return residual_penalty * penalty_scale


def stand_still_amp_deviation_exp(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    command_name: str = "base_velocity",
    command_threshold: float = 0.06,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward small deviations from zero amplitudes when the command is very small."""
    commands = env.command_manager.get_command(command_name)
    is_zero_command = torch.norm(commands[:, :2], dim=1) < command_threshold

    action_term = env.action_manager.get_term(action_name)
    pmtg_action_term = cast(FourLegsPMTGAction, action_term)
    amplitudes = pmtg_action_term.processed_actions[:, 1:4]

    amp_deviation = torch.sum(torch.square(amplitudes), dim=1)
    reward = torch.exp(-amp_deviation / std**2)

    return reward * is_zero_command


def stand_still_residuals_exp(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward small residuals when the amplitudes are small."""
    action_term = env.action_manager.get_term(action_name)
    pmtg_action_term = cast(FourLegsPMTGAction, action_term)
    filtered_amplitudes = pmtg_action_term.filtered_amplitudes

    is_standing_still = (
        (
            filtered_amplitudes[:, 0]
            < pmtg_action_term.cfg.trajectory_generator_params.dead_zone
        )
        & (
            filtered_amplitudes[:, 1]
            < pmtg_action_term.cfg.trajectory_generator_params.dead_zone
        )
        & (
            filtered_amplitudes[:, 2]
            < pmtg_action_term.cfg.trajectory_generator_params.dead_zone
        )
    )
    residuals = pmtg_action_term.processed_actions[:, 4:]

    residual_deviation = torch.sum(torch.square(residuals), dim=1)
    reward = torch.exp(-residual_deviation / std**2)

    return reward * is_standing_still

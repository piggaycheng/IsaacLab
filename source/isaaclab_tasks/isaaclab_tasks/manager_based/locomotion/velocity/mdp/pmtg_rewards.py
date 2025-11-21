from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs.mdp.actions.pmtg_actions import FourLegsPMTGAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pmtg_cpg_args_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the smoothness of the CPG arguments (after tanh).

    This computes the L2 norm of the difference between the current and previous
    tanh-processed CPG arguments.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_cpg = action_term.raw_actions[:, :8]
    prev_tanh_cpg = action_term.last_raw_actions[:, :8]

    return torch.mean(torch.square(curr_tanh_cpg - prev_tanh_cpg), dim=1)


def pmtg_residuals_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the smoothness of the residuals (after tanh).

    This computes the L2 norm of the difference between the current and previous
    tanh-processed residuals.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_res = action_term.raw_actions[:, 8:]
    prev_tanh_res = action_term.last_raw_actions[:, 8:]

    return torch.mean(torch.square(curr_tanh_res - prev_tanh_res), dim=1)


def pmtg_tanh_residuals_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the L2 norm of the residuals (after tanh).

    This encourages the residuals to be zero, applied on the tanh-processed values.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_res = action_term.raw_actions[:, 8:]

    return torch.mean(torch.sum(torch.square(curr_tanh_res), dim=1))


def pmtg_cpg_when_stationary_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize non-zero cpg when the robot is commanded to be stationary.

    This computes the L2 norm of the amplitudes (amp_x, amp_y),
    offsets (offset_x, offset_y, offset_z), and yaw_param when the command magnitude is below a threshold.
    """
    # Get the command
    cmd = env.command_manager.get_command(command_name)
    # Check if stationary
    is_stationary = (torch.norm(cmd[:, :2], dim=1) < threshold) & (
        torch.abs(cmd[:, 2]) < threshold
    )

    # Get the action term
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    # Get tanh-processed CPG args need to be penalized
    # Exclude amp_z (index 3)
    tanh_cpg = torch.cat(
        (action_term.raw_actions[:, 1:3], action_term.raw_actions[:, 4:8]), dim=1
    )

    # Compute penalty: sum of squares of amplitudes
    penalty = torch.sum(torch.square(tanh_cpg), dim=1)

    # Apply mask
    return penalty * is_stationary.float()


def pmtg_amplitudes_z_when_stationary_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize non-zero amplitude z when the robot is commanded to be stationary.

    This computes the square of the amplitude z when the command magnitude is below a threshold.
    """
    # Get the command
    cmd = env.command_manager.get_command(command_name)
    # Check if stationary
    is_stationary = (torch.norm(cmd[:, :2], dim=1) < threshold) & (
        torch.abs(cmd[:, 2]) < threshold
    )

    # Get the action term
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    # Get tanh-processed amplitude z
    amp_z = action_term.processed_amplitudes[:, 2]

    # Compute penalty: square of amplitude z
    penalty = torch.square(amp_z)
    # Apply mask
    return penalty * is_stationary.float()


def pmtg_joint_pos_ik_error_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the L2 norm of the difference between IK-computed joint positions and current joint positions.

    The difference is computed as: joint_pos_ik - current_joint_pos.
    The current joint positions are reordered to match the IK output format (grouped by joint type).
    """
    action_term = env.action_manager.get_term(action_name)
    # Cast the action term to the specific type to access its properties
    pmtg_action = cast(FourLegsPMTGAction, action_term)

    # Get IK computed joint positions (already in [J1L1, J1L2, ..., J2L1, ...] format)
    ik_pos = pmtg_action.joint_pos_ik

    # Get current joint positions
    asset: Articulation = env.scene[asset_cfg.name]
    curr_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Compute L2 norm of the difference
    error = ik_pos - curr_pos
    return torch.sum(torch.square(error), dim=1)

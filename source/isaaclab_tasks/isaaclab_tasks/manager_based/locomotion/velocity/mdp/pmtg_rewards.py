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

    return torch.sum(torch.square(curr_tanh_cpg - prev_tanh_cpg), dim=1)


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

    return torch.sum(torch.square(curr_tanh_res - prev_tanh_res), dim=1)


def pmtg_tanh_residuals_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the L2 norm of the residuals (after tanh).

    This encourages the residuals to be zero, applied on the tanh-processed values.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_res = action_term.raw_actions[:, 8:]

    return torch.sum(torch.square(curr_tanh_res), dim=1)


def pmtg_cpg_when_stationary(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize non-zero cpg when the robot is commanded to be stationary.

    This computes the L2 norm of the amplitudes (amp_x, amp_y, amp_z),
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
    tanh_cpg = action_term.raw_actions[:, 1:8]

    # Compute penalty: sum of squares of amplitudes
    penalty = torch.sum(torch.square(tanh_cpg), dim=1)

    # Apply mask
    return penalty * is_stationary.float()

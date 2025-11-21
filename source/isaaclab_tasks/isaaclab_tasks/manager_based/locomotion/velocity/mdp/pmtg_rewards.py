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


def _compute_tanh_cpg_args(
    action_term: FourLegsPMTGAction, raw_actions: torch.Tensor
) -> torch.Tensor:
    """Helper to compute tanh-processed CPG arguments from raw actions."""
    tg_actions_raw = raw_actions[:, :4]
    frequency, amp_x, amp_y, amp_z = tg_actions_raw.unbind(dim=1)

    return torch.stack(
        [
            action_term.tanh_process(
                frequency, action_term.cfg.trajectory_generator_params.frequency_limit
            ),
            action_term.tanh_process(
                amp_x, action_term.cfg.trajectory_generator_params.step_length_x_limit
            ),
            action_term.tanh_process(
                amp_y, action_term.cfg.trajectory_generator_params.step_length_y_limit
            ),
            action_term.tanh_process(
                amp_z, action_term.cfg.trajectory_generator_params.step_height_limit
            ),
        ],
        dim=1,
    )


def _compute_tanh_residuals(
    action_term: FourLegsPMTGAction, raw_actions: torch.Tensor
) -> torch.Tensor:
    """Helper to compute tanh-processed residuals from raw actions."""
    residuals_raw = raw_actions[:, 4:]
    return action_term.tanh_process(residuals_raw, action_term.cfg.residuals_limit)


def pmtg_cpg_args_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the smoothness of the CPG arguments (after tanh).

    This computes the L2 norm of the difference between the current and previous
    tanh-processed CPG arguments.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_cpg = _compute_tanh_cpg_args(action_term, env.action_manager.action)
    prev_tanh_cpg = _compute_tanh_cpg_args(action_term, env.action_manager.prev_action)

    return torch.sum(torch.square(curr_tanh_cpg - prev_tanh_cpg), dim=1)


def pmtg_residuals_smoothness_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the smoothness of the residuals (after tanh).

    This computes the L2 norm of the difference between the current and previous
    tanh-processed residuals.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_res = _compute_tanh_residuals(action_term, env.action_manager.action)
    prev_tanh_res = _compute_tanh_residuals(action_term, env.action_manager.prev_action)

    return torch.sum(torch.square(curr_tanh_res - prev_tanh_res), dim=1)


def pmtg_tanh_residuals_l2(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Penalize the L2 norm of the residuals (after tanh).

    This encourages the residuals to be zero, applied on the tanh-processed values.
    """
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    curr_tanh_res = _compute_tanh_residuals(action_term, env.action_manager.action)

    return torch.sum(torch.square(curr_tanh_res), dim=1)


def pmtg_amplitudes_when_stationary(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "joint_pos",
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize non-zero amplitudes when the robot is commanded to be stationary.

    This computes the L2 norm of the amplitudes (amp_x, amp_y, amp_z) when the
    command magnitude is below a threshold.
    """
    # Get the command
    cmd = env.command_manager.get_command(command_name)
    # Check if stationary
    is_stationary = (torch.norm(cmd[:, :2], dim=1) < threshold) & (torch.abs(cmd[:, 2]) < threshold)

    # Get the action term
    action_term = cast(FourLegsPMTGAction, env.action_manager.get_term(action_name))

    # Get tanh-processed CPG args
    # _compute_tanh_cpg_args returns [freq, amp_x, amp_y, amp_z]
    tanh_cpg = _compute_tanh_cpg_args(action_term, env.action_manager.action)

    # Extract amplitudes (indices 1, 2, 3)
    amplitudes = tanh_cpg[:, 1:4]

    # Compute penalty: sum of squares of amplitudes
    penalty = torch.sum(torch.square(amplitudes), dim=1)

    # Apply mask
    return penalty * is_stationary.float()

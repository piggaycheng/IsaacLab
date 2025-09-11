from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast
from isaaclab.envs.mdp.actions.pmtg_actions import FourLegsPMTGAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def trajectory_generator_phase(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Observation of the trajectory generator's phase for each leg."""
    action_term = env.action_manager.get_term(action_name)
    # Cast the action term to the specific type to access its properties
    pmtg_action = cast(FourLegsPMTGAction, action_term)
    return pmtg_action.phases

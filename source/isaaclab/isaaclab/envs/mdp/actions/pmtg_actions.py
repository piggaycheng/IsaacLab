from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import PMTGJointPositionActionCfg


class PMTGJointPositionAction(ActionTerm):
    """PMTG-driven joint position action.

    Policy outputs a latent vector (per env). This term maps latent + time -> joint position targets
    every simulation step using an internal trajectory generator.
    """

    cfg: PMTGJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: PMTGJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for PMTG action: {self._joint_names} [{self._joint_ids}]"
        )
        if self._num_joints == self._asset.num_joints and not cfg.preserve_order:
            self._joint_ids = slice(None)

        # latent dimension
        self._latent_dim = int(cfg.latent_dim)
        self._latent = torch.zeros(self.num_envs, self._latent_dim, device=self.device)

        # output scaling/offset
        self._out_scale = self._parse_per_joint_value(cfg.output_scale, default=1.0)
        if cfg.use_default_offset:
            self._out_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._out_offset = self._parse_per_joint_value(cfg.output_offset, default=0.0)

        # internal PMTG weights (fixed random maps latent->per-joint params)
        g = torch.Generator(device=self.device).manual_seed(1234)
        self._W_amp = torch.randn(self._latent_dim, self._num_joints, generator=g, device=self.device) * 0.1
        self._W_bias = torch.randn(self._latent_dim, self._num_joints, generator=g, device=self.device) * 0.1
        self._W_phase = torch.randn(self._latent_dim, self._num_joints, generator=g, device=self.device) * 0.1

        # PMTG params
        self._freq = float(cfg.freq_hz)
        self._amp_scale = float(cfg.amp_scale)
        self._bias_scale = float(cfg.bias_scale)
        self._phase_scale = float(cfg.phase_scale)

        # time per env (seconds)
        self._t = torch.zeros(self.num_envs, device=self.device)

        # optional per-joint clip from cfg.clip if provided at ActionTermCfg level
        self._clip = None
        if cfg.clip is not None:
            clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._num_joints, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.clip, self._joint_names)
            clip[:, index_list] = torch.tensor(value_list, device=self.device)
            self._clip = clip

    @property
    def action_dim(self) -> int:
        # The manager will allocate latent_dim actions for this term
        return self._latent_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._latent

    @property
    def processed_actions(self) -> torch.Tensor:
        # PMTG uses the latent directly; joint targets are produced in apply_actions
        return self._latent

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self._latent[env_ids] = 0.0
        self._t[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # store current latent
        if actions.shape[1] != self._latent_dim:
            raise ValueError(
                f"PMTG latent dim mismatch: expected {self._latent_dim}, got {actions.shape[1]}"
            )
        self._latent[:] = actions

    def apply_actions(self):
        # compute per-joint parameters from latent
        amp = torch.tanh(self._latent @ self._W_amp) * self._amp_scale
        bias = torch.tanh(self._latent @ self._W_bias) * self._bias_scale
        phase = (self._latent @ self._W_phase) * self._phase_scale

        # time advance (env.physics_dt at sim-rate)
        omega_t = (2 * math.pi * self._freq) * self._t.unsqueeze(-1)
        joint_cmd = amp * torch.sin(omega_t + phase) + bias

        # scale/offset to joint space
        joint_targets = joint_cmd * self._out_scale + self._out_offset

        if self._clip is not None:
            joint_targets = torch.clamp(joint_targets, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        # send commands
        self._asset.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)

        # increment time
        self._t += self._env.physics_dt

    def _parse_per_joint_value(self, value, default: float) -> torch.Tensor:
        """Parses a float or regex-dict into a (num_envs, num_joints) tensor on device."""
        if isinstance(value, (float, int)):
            return torch.full((self.num_envs, self._num_joints), float(value), device=self.device)
        elif isinstance(value, dict):
            tensor = torch.full((self.num_envs, self._num_joints), float(default), device=self.device)
            idxs, _, vals = string_utils.resolve_matching_names_values(value, self._joint_names)
            tensor[:, idxs] = torch.tensor(vals, device=self.device)
            return tensor
        else:
            return torch.full((self.num_envs, self._num_joints), float(default), device=self.device)

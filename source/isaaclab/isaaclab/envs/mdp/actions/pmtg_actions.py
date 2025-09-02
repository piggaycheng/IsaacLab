# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        # whether to include per-joint residuals in the action space
        # default False to preserve backward compatibility unless explicitly enabled in cfg
        self._include_residual = bool(getattr(cfg, "include_residual", False))
        # residual actions (per joint)
        self._residual = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # allow scaling and optional clipping of residual term
        self._residual_scale = self._parse_per_joint_value(getattr(cfg, "residual_scale", 1.0), default=1.0)
        self._residual_clip = None
        residual_clip_cfg = getattr(cfg, "residual_clip", None)
        if residual_clip_cfg is None:
            # also support legacy scalar limit via `residual_limit`
            residual_limit = getattr(cfg, "residual_limit", None)
            if residual_limit is not None:
                residual_clip_cfg = {".*": [-abs(float(residual_limit)), abs(float(residual_limit))]}
        if residual_clip_cfg is not None:
            rclip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._num_joints, 1
            )
            r_index, _, r_values = string_utils.resolve_matching_names_values(residual_clip_cfg, self._joint_names)
            rclip[:, r_index] = torch.tensor(r_values, device=self.device)
            self._residual_clip = rclip

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

        # fixed per-joint phase offsets (e.g., to enforce gait leg phase relations)
        phase_offset_value = getattr(cfg, "phase_offset", 0.0)
        self._phase_offset = self._parse_per_joint_value(phase_offset_value, default=0.0)

        # time per env (seconds)
        self._t = torch.zeros(self.num_envs, device=self.device)

        # optional per-joint clip from cfg.clip if provided at ActionTermCfg level
        self._clip = None
        clip_cfg = getattr(cfg, "clip", None)
        if clip_cfg is not None:
            clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._num_joints, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(clip_cfg, self._joint_names)
            clip[:, index_list] = torch.tensor(value_list, device=self.device)
            self._clip = clip

        # keep a copy of last raw action vector from the policy (for logging/inspection)
        self._raw_action = None

    @property
    def action_dim(self) -> int:
        # Manager allocates latent + optional per-joint residuals
        return self._latent_dim + (self._num_joints if self._include_residual else 0)

    @property
    def raw_actions(self) -> torch.Tensor:
        # Return the last raw action vector provided by the policy
        if self._raw_action is None:
            # before first action, synthesize from internal buffers
            return self.processed_actions
        return self._raw_action

    @property
    def processed_actions(self) -> torch.Tensor:
        # Concatenate latent and residual (if enabled) after basic validation/sanitization
        if self._include_residual:
            return torch.cat([self._latent, self._residual], dim=1)
        else:
            return self._latent

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self._latent[env_ids] = 0.0
        self._residual[env_ids] = 0.0
        self._t[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # Accept either [latent] or [latent | residual] for backward compatibility
        self._raw_action = actions

        if actions.dim() != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Actions must be (num_envs, D). Got {tuple(actions.shape)} while num_envs={self.num_envs}."
            )

        if self._include_residual:
            expected_with_res = self._latent_dim + self._num_joints
            if actions.shape[1] == expected_with_res:
                self._latent[:] = actions[:, : self._latent_dim]
                self._residual[:] = actions[:, self._latent_dim :]
            elif actions.shape[1] == self._latent_dim:
                # allow latent-only input: zero residuals
                self._latent[:] = actions
                self._residual[:] = 0.0
            else:
                raise ValueError(
                    f"PMTG action dim mismatch: expected {expected_with_res} (latent+residual) "
                    f"or {self._latent_dim} (latent-only), got {actions.shape[1]}"
                )
        else:
            if actions.shape[1] != self._latent_dim:
                raise ValueError(
                    f"PMTG latent dim mismatch: expected {self._latent_dim}, got {actions.shape[1]}"
                )
            self._latent[:] = actions

    def apply_actions(self):
        # compute per-joint parameters from latent
        amp = torch.tanh(self._latent @ self._W_amp) * self._amp_scale
        bias = torch.tanh(self._latent @ self._W_bias) * self._bias_scale
        phase = (self._latent @ self._W_phase) * self._phase_scale + self._phase_offset

        # time advance (env.physics_dt at sim-rate)
        omega_t = (2 * math.pi * self._freq) * self._t.unsqueeze(-1)
        joint_cmd = amp * torch.sin(omega_t + phase) + bias

        # optional per-joint residual term from the policy
        if self._include_residual:
            residual = self._residual * self._residual_scale
            if self._residual_clip is not None:
                residual = torch.clamp(residual, min=self._residual_clip[:, :, 0], max=self._residual_clip[:, :, 1])
            joint_cmd = joint_cmd + residual

        # scale/offset to joint space
        joint_targets = joint_cmd * self._out_scale + self._out_offset

        if self._clip is not None:
            joint_targets = torch.clamp(joint_targets, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        # send commands
        self._asset.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)

        # increment time
        self._t += self._env.physics_dt

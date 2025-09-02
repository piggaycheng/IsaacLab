# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2FlatEnvCfg_PMTG(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.joint_pos = mdp.PMTGJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            latent_dim=12,  # latent size
            freq_hz=1.0,  # gait frequency (Hz)
            amp_scale=1.0,
            bias_scale=0.0,
            phase_scale=3.14159265,
            output_scale=0.5,  # overall joint target scale
            use_default_offset=True,  # center around default joint pos
            # enable residual: action = PMTG(latent,t) + residual
            include_residual=True,
            residual_scale=1.0,
            residual_limit=0.35,  # optional clip of residuals per joint ([-limit, +limit])
            phase_offset={
                "FL.*": 0.0,
                "FR.*": 3.14159265,
                "RL.*": 3.14159265,
                "RR.*": 0.0,
            },
        )

        self.rewards.track_lin_vel_xy_exp.weight = 2.5


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_PLAY(UnitreeGo2FlatEnvCfg_PMTG):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

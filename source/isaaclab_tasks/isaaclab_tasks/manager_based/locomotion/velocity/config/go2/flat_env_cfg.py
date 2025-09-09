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

        self.actions.joint_pos = mdp.FourLegsPMTGActionCfg(
            asset_name="robot",
            ik_action_cfgs=[
                mdp.DifferentialInverseKinematicsActionCfg(
                    asset_name="robot",
                    joint_names=["FL_.*"],
                    body_name="FL_foot",
                    controller=mdp.DifferentialIKControllerCfg(
                        command_type="position",
                        ik_method="dls",
                        use_relative_mode=False,
                    ),
                ),
                mdp.DifferentialInverseKinematicsActionCfg(
                    asset_name="robot",
                    joint_names=["FR_.*"],
                    body_name="FR_foot",
                    controller=mdp.DifferentialIKControllerCfg(
                        command_type="position",
                        ik_method="dls",
                        use_relative_mode=False,
                    ),
                ),
                mdp.DifferentialInverseKinematicsActionCfg(
                    asset_name="robot",
                    joint_names=["RL_.*"],
                    body_name="RL_foot",
                    controller=mdp.DifferentialIKControllerCfg(
                        command_type="position",
                        ik_method="dls",
                        use_relative_mode=False,
                    ),
                ),
                mdp.DifferentialInverseKinematicsActionCfg(
                    asset_name="robot",
                    joint_names=["RR_.*"],
                    body_name="RR_foot",
                    controller=mdp.DifferentialIKControllerCfg(
                        command_type="position",
                        ik_method="dls",
                        use_relative_mode=False,
                    ),
                ),
            ],
            leg_hip_positions=([0.1934, 0.0465, 0.0], [0.1934, -0.0465, 0.0], [-0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0]),  # FL, FR, RL, RR
            # gain=0.5,
            foot_default_heights=(-0.3, -0.3, -0.35, -0.35),
            leg_y_offsets=(0.1, -0.1, 0.1, -0.1),
        )


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

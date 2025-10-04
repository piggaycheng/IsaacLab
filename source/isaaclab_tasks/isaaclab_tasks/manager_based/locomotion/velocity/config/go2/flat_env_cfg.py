# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

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

        # self.scene.robot.actuators['base_legs'].stiffness = 50.0
        # self.scene.robot.actuators['base_legs'].damping = 1.0

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
            gain=1.0,
            residuals_scale=0.02,
            action_smoothing_alpha=1.0,
            trajectory_generator_params=mdp.FourLegsPMTGActionCfg.TrajectoryGeneratorCfg(
                leg_hip_positions=([0.1934, 0.0465, 0.0], [0.1934, -0.0465, 0.0], [-0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0]),  # FL, FR, RL, RR
                foot_default_heights=(-0.25, -0.25, -0.28, -0.28),
                leg_y_offsets=(0.1, -0.1, 0.1, -0.1),
                leg_x_offsets=(0.05, 0.05, -0.05, -0.05),
                stance_vx_scale=0.5,
                stance_vy_scale=0.5,
                yaw_rate_scale=0.5,
                step_height_scale=0.5,
            )
        )

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.standing_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )
        self.rewards.action_rate_l2 = None

        # self.observations.policy.pmtg_phase = ObsTerm(
        #     func=mdp.trajectory_generator_phase,
        #     params={
        #         "action_name": "joint_pos",
        #     },
        # )


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

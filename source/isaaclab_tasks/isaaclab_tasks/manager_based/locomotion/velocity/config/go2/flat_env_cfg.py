# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ImuCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
)

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
class PMTGObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        pmtg_phase = ObsTerm(
            func=mdp.trajectory_generator_phase,
            params={
                "action_name": "joint_pos",
            },
        )
        pmtg_joint_pos_des = ObsTerm(
            func=mdp.trajectory_generator_joint_pos_des,
            params={
                "action_name": "joint_pos",
            },
        )
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = None
        height_scan = None

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
            trajectory_generator_params=mdp.FourLegsPMTGActionCfg.TrajectoryGeneratorCfg(
                leg_hip_positions=(
                    [0.1934, 0.0465, 0.0],
                    [0.1934, -0.0465, 0.0],
                    [-0.1934, 0.0465, 0.0],
                    [-0.1934, -0.0465, 0.0],
                ),  # FL, FR, RL, RR
                foot_default_heights=(-0.3, -0.3, -0.32, -0.32),
                leg_y_offsets=(0.12, -0.12, 0.12, -0.12),
                leg_x_offsets=(0.02, 0.02, -0.05, -0.05),
            ),
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


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v1(UnitreeGo2FlatEnvCfg_PMTG):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.pmtg_phase = ObsTerm(
            func=mdp.trajectory_generator_phase,
            params={
                "action_name": "joint_pos",
            },
        )

        self.observations.policy.pmtg_joint_pos_des = ObsTerm(
            func=mdp.trajectory_generator_joint_pos_des,
            params={
                "action_name": "joint_pos",
            },
            history_length=2,
        )

        self.observations.policy.joint_pos.history_length = 3
        self.observations.policy.joint_vel.history_length = 2


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v2(UnitreeGo2FlatEnvCfg_PMTG_v1):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.1

        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.alive = RewTerm(
            func=mdp.is_alive,
            weight=1.0,
        )
        # self.rewards.joint_residuals_penalty = RewTerm(
        #     func=mdp.pmtg_joint_residuals_l2,
        #     weight=-0.1,
        #     params={
        #         "action_name": "joint_pos",
        #     },
        # )
        # self.rewards.amplitude_residuals_ratio = RewTerm(
        #     func=mdp.pmtg_amplitude_residual_ratio_l2,
        #     weight=1.0,
        #     params={
        #         "command_name": "base_velocity",
        #         "action_name": "joint_pos",
        #     },
        # )
        # self.rewards.conditional_joint_residuals_penalty = RewTerm(
        #     func=mdp.conditional_joint_residuals_l2,
        #     weight=-0.1,
        #     params={
        #         "action_name": "joint_pos",
        #         "command_name": "base_velocity",
        #     },
        # )
        self.rewards.feet_slide_penalty = RewTerm(
            func=mdp.feet_slide,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        self.rewards.standing_still = None


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v2_1(UnitreeGo2FlatEnvCfg_PMTG_v2):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.pmtg_joint_pos_des.history_length = 0
        self.observations.policy.joint_pos.history_length = 0
        self.observations.policy.joint_vel.history_length = 0


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v2_PLAY(UnitreeGo2FlatEnvCfg_PMTG_v2):
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


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v3(UnitreeGo2FlatEnvCfg_PMTG_v2):

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.0,
        )

        self.observations = PMTGObservationsCfg()

        self.rewards.alive = None


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v3_PLAY(UnitreeGo2FlatEnvCfg_PMTG_v3):
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


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v4(UnitreeGo2FlatEnvCfg_PMTG_v3):

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.2

        self.scene.robot.actuators["base_legs"].stiffness = 100.0
        self.scene.robot.actuators["base_legs"].damping = 1.0

        self.actions.joint_pos.trajectory_generator_params.dead_zone = 0.03
        self.actions.joint_pos.lpf_alpha = 0.2
        self.actions.joint_pos.residuals_limit = (-0.1, 0.1)
        self.actions.joint_pos.residuals_dead_zone = 0.02

        self.rewards.stand_still_amp_deviation = RewTerm(
            func=mdp.stand_still_amp_deviation_exp,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "action_name": "joint_pos",
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=spot_mdp.foot_clearance_reward,
            weight=0.5,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        self.rewards.action_rate_l2 = None
        self.rewards.standing_still_residuals = RewTerm(
            func=mdp.stand_still_residuals_exp,
            weight=1.0,
            params={
                "action_name": "joint_pos",
                "std": 0.5,
            },
        )


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v4_PLAY(UnitreeGo2FlatEnvCfg_PMTG_v4):

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


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v5(UnitreeGo2FlatEnvCfg_PMTG_v4):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.pmtg_joint_pos_des.history_length = 2
        self.observations.policy.joint_pos.history_length = 3
        self.observations.policy.joint_vel.history_length = 2

        self.observations.critic.pmtg_joint_pos_des.history_length = 2
        self.observations.critic.joint_pos.history_length = 3
        self.observations.critic.joint_vel.history_length = 2


@configclass
class UnitreeGo2FlatEnvCfg_PMTG_v5_PLAY(UnitreeGo2FlatEnvCfg_PMTG_v5):

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

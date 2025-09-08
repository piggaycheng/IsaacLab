# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import UNITREE_GO2_CFG
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=55.0,
                damping=0.5,
                friction=0.0,
            )
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
            rot=(0.0, 1.0, 0.0, 0.0)
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    fl_diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")
    fr_diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")
    rl_diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")
    rr_diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")
    fl_diff_ik_controller = DifferentialIKController(fl_diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    fr_diff_ik_controller = DifferentialIKController(fr_diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    rl_diff_ik_controller = DifferentialIKController(rl_diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    rr_diff_ik_controller = DifferentialIKController(rr_diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.0, 0.05, 0.0],
        [0.0, -0.05, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, fr_diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    fl_robot_entity_cfg = SceneEntityCfg("robot", joint_names=["FL_.*"], body_names=["FL_foot"])
    fr_robot_entity_cfg = SceneEntityCfg("robot", joint_names=["FR_.*"], body_names=["FR_foot"])
    rl_robot_entity_cfg = SceneEntityCfg("robot", joint_names=["RL_.*"], body_names=["RL_foot"])
    rr_robot_entity_cfg = SceneEntityCfg("robot", joint_names=["RR_.*"], body_names=["RR_foot"])

    # Resolving the scene entities
    fl_robot_entity_cfg.resolve(scene)
    fr_robot_entity_cfg.resolve(scene)
    rl_robot_entity_cfg.resolve(scene)
    rr_robot_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        fl_ee_jacobi_idx = fl_robot_entity_cfg.body_ids[0] - 1
        fr_ee_jacobi_idx = fr_robot_entity_cfg.body_ids[0] - 1
        rl_ee_jacobi_idx = rl_robot_entity_cfg.body_ids[0] - 1
        rr_ee_jacobi_idx = rr_robot_entity_cfg.body_ids[0] - 1
    else:
        fl_ee_jacobi_idx = fl_robot_entity_cfg.body_ids[0]
        fr_ee_jacobi_idx = fr_robot_entity_cfg.body_ids[0]
        rl_ee_jacobi_idx = rl_robot_entity_cfg.body_ids[0]
        rr_ee_jacobi_idx = rr_robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # Simulation loop
    while simulation_app.is_running():
        # 前左腿 (FL)
        fl_ee_pose_w = robot.data.body_pose_w[:, fl_robot_entity_cfg.body_ids[0]]
        fl_root_pose_w = robot.data.root_pose_w
        fl_ee_pos_b, fl_ee_quat_b = subtract_frame_transforms(
            fl_root_pose_w[:, 0:3], fl_root_pose_w[:, 3:7], fl_ee_pose_w[:, 0:3], fl_ee_pose_w[:, 3:7]
        )
        # 前右腿 (FR)
        fr_ee_pose_w = robot.data.body_pose_w[:, fr_robot_entity_cfg.body_ids[0]]
        fr_root_pose_w = robot.data.root_pose_w
        fr_ee_pos_b, fr_ee_quat_b = subtract_frame_transforms(
            fr_root_pose_w[:, 0:3], fr_root_pose_w[:, 3:7], fr_ee_pose_w[:, 0:3], fr_ee_pose_w[:, 3:7]
        )
        # 後左腿 (RL)
        rl_ee_pose_w = robot.data.body_pose_w[:, rl_robot_entity_cfg.body_ids[0]]
        rl_root_pose_w = robot.data.root_pose_w
        rl_ee_pos_b, rl_ee_quat_b = subtract_frame_transforms(
            rl_root_pose_w[:, 0:3], rl_root_pose_w[:, 3:7], rl_ee_pose_w[:, 0:3], rl_ee_pose_w[:, 3:7]
        )
        # 後右腿 (RR)
        rr_ee_pose_w = robot.data.body_pose_w[:, rr_robot_entity_cfg.body_ids[0]]
        rr_root_pose_w = robot.data.root_pose_w
        rr_ee_pos_b, rr_ee_quat_b = subtract_frame_transforms(
            rr_root_pose_w[:, 0:3], rr_root_pose_w[:, 3:7], rr_ee_pose_w[:, 0:3], rr_ee_pose_w[:, 3:7]
        )

        if count % 150 == 0:
            ik_commands[:] = ee_goals[current_goal_idx]
            fl_diff_ik_controller.reset()
            fr_diff_ik_controller.reset()
            rl_diff_ik_controller.reset()
            rr_diff_ik_controller.reset()

            count = 0
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

        full_jacobian = robot.root_physx_view.get_jacobians()
        fl_jacobian = full_jacobian[:, fl_ee_jacobi_idx, :, fl_robot_entity_cfg.joint_ids]
        fr_jacobian = full_jacobian[:, fr_ee_jacobi_idx, :, fr_robot_entity_cfg.joint_ids]
        rl_jacobian = full_jacobian[:, rl_ee_jacobi_idx, :, rl_robot_entity_cfg.joint_ids]
        rr_jacobian = full_jacobian[:, rr_ee_jacobi_idx, :, rr_robot_entity_cfg.joint_ids]

        fl_joint_pos = robot.data.joint_pos[:, fl_robot_entity_cfg.joint_ids]
        fr_joint_pos = robot.data.joint_pos[:, fr_robot_entity_cfg.joint_ids]
        rl_joint_pos = robot.data.joint_pos[:, rl_robot_entity_cfg.joint_ids]
        rr_joint_pos = robot.data.joint_pos[:, rr_robot_entity_cfg.joint_ids]

        fl_diff_ik_controller.set_command(ik_commands, fl_ee_pos_b, fl_ee_quat_b)
        fr_diff_ik_controller.set_command(ik_commands, fr_ee_pos_b, fr_ee_quat_b)
        rl_diff_ik_controller.set_command(ik_commands, rl_ee_pos_b, rl_ee_quat_b)
        rr_diff_ik_controller.set_command(ik_commands, rr_ee_pos_b, rr_ee_quat_b)

        fl_joint_pos_des = fl_diff_ik_controller.compute(fl_ee_pos_b, fl_ee_quat_b, fl_jacobian, fl_joint_pos)
        fr_joint_pos_des = fr_diff_ik_controller.compute(fr_ee_pos_b, fr_ee_quat_b, fr_jacobian, fr_joint_pos)
        rl_joint_pos_des = rl_diff_ik_controller.compute(rl_ee_pos_b, rl_ee_quat_b, rl_jacobian, rl_joint_pos)
        rr_joint_pos_des = rr_diff_ik_controller.compute(rr_ee_pos_b, rr_ee_quat_b, rr_jacobian, rr_joint_pos)

        # apply actions
        robot.set_joint_position_target(fl_joint_pos_des, joint_ids=fl_robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(fr_joint_pos_des, joint_ids=fr_robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(rl_joint_pos_des, joint_ids=rl_robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(rr_joint_pos_des, joint_ids=rr_robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        # ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # Visualize goal marker at target position with identity quaternion orientation
        # goal_marker.visualize(
        #     ik_commands[:, :3] + scene.env_origins,
        #     torch.tensor([1.0, 0.0, 0.0, 0.0], device=ik_commands.device).expand(ik_commands.shape[0], 4)
        # )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

"""Inverse kinematics demo (refactored with RobotController class and physics callback).

This refactors the original script so that IK logic is encapsulated inside a controller
class and executed through the world's physics callback system. A stop-event subscription
resets internal state when the simulation timeline is halted (stop / play cycle).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from isaacsim import SimulationApp  # type: ignore  # (Runtime provided by Isaac Sim environment)

# Launch the simulation (GUI)
simulation_app = SimulationApp({"headless": False})

import carb
import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)

import omni
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics


@dataclass
class KinematicsConfig:
    robot_description_path: str
    urdf_path: str
    end_effector_name: str = "FL_foot"


class RobotController:
    """Encapsulates robot + IK solver state and reacts to physics & stop events."""

    def __init__(self, robot_prim_path: str, kinematics_config: KinematicsConfig):
        self.robot_prim_path = robot_prim_path
        self.cfg = kinematics_config
        self.robot = SingleArticulation(prim_path=robot_prim_path, position=(0, 0, 0.0))
        self.target = SingleXFormPrim("/World/Target", position=(0.2, 0.15, -0.25))

        kinematics_solver = LulaKinematicsSolver(
            robot_description_path=self.cfg.robot_description_path,
            urdf_path=self.cfg.urdf_path,
        )
        self.ik_solver = ArticulationKinematicsSolver(self.robot, kinematics_solver, self.cfg.end_effector_name)

        self.is_initialized = False
        self.stop_subscription = None  # Will hold the timeline stop subscription

    # Callback signature: fn(step_size: float) -> None
    def on_physics_step(self, step_size: float):  # noqa: D401 - Isaac Sim callback
        if not self.is_initialized:
            # Initialize (also covers re-initialization after stop->play)
            self.robot.initialize()
            self.is_initialized = True

        # Perform IK each physics step
        target_pos, target_orn = self.target.get_world_pose()
        action, success = self.ik_solver.compute_inverse_kinematics(target_pos, np.array([1, 0, 0, 0]))
        if success:
            # Apply returned joint position command vector
            self.robot.apply_action(action)
        else:
            carb.log_warn("IK did not converge this step; skipping action.")

    def on_stop_event(self, event):  # noqa: D401 - timeline stop event
        carb.log_info("Simulation stopped. Resetting controller initialization state.")
        self.is_initialized = False


def _create_world():  # Returns an instance compatible with World API (typing relaxed)
    """Create and return a configured World with lighting & ground plane."""
    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    physics_dt = 1.0 / 200.0
    render_dt = 1.0 / 50.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    GroundPlane(prim_path="/World/GroundPlane", z_position=-0.5)

    # stage = omni.usd.get_context().get_stage()
    # scene = UsdPhysics.Scene.Define(stage, "/World/physics")
    # scene.CreateGravityMagnitudeAttr().Set(0.0)

    return world  # type: ignore


def _load_robot_usd(robot_prim_path: str):
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Could not find assets root path. Check Nucleus connection or asset configuration."
        )
    usd_path = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
    add_reference_to_stage(usd_path, robot_prim_path)


def main():
    parser = argparse.ArgumentParser(description="Inverse kinematics demo (Go2)")
    parser.parse_args()  # (No CLI args currently; placeholder for future extensions)

    world = _create_world()

    # Robot prim path & USD loading
    robot_prim_path = "/World/Go2"
    _load_robot_usd(robot_prim_path)

    # Kinematics configuration (paths kept as provided by user environment)
    kinematics_config = KinematicsConfig(
        robot_description_path="D:/Users/yucheng/Documents/MyIsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/robot_description/go2_robot_description.yaml",
        urdf_path="D:/Users/yucheng/Documents/MyIsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/robot_description/go2_description.urdf",
        end_effector_name="FL_foot",
    )

    # Create controller & register callbacks
    controller = RobotController(robot_prim_path, kinematics_config)
    world.add_physics_callback("robot_ik_callback", controller.on_physics_step)

    timeline = get_timeline_interface()
    # Timeline stop event stream (typing may not expose get_stop_event_stream; ignore if unresolved)
    controller.stop_subscription = timeline.get_timeline_event_stream().create_subscription_to_pop(  # type: ignore[attr-defined]
        controller.on_stop_event
    )

    # Reset & start playing
    world.reset()
    world.play()

    # Main render loop
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    main()

    simulation_app.close()

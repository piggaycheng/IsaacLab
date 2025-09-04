import argparse

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.api import World
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()


def main():
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    physics_dt = 1 / 200.0
    render_dt = 1 / 50.0
    my_world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

    GroundPlane(prim_path="/World/GroundPlane", z_position=0)
    target = SingleXFormPrim("/World/Target", position=(0.5, 0, 0.5))
    # target = prim_utils.create_prim("/World/Cube", prim_type="Cube", position=(0, 1, 0.5), scale=(0.05, 0.05, 0.05))

    robot_prim_path = "/World/Go2"
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find assets root path. Please check your Nucleus connection or asset configuration.")
    path_to_robot_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
    add_reference_to_stage(path_to_robot_usd, robot_prim_path)
    robot = SingleArticulation(prim_path=robot_prim_path, position=(0, 0, 0.5))

    # 執行一次 my_world.reset() 來確保所有物件都被正確初始化
    my_world.reset()

    robot.initialize()

    kinematics_solver = LulaKinematicsSolver(
        robot_description_path="D:/Users/yucheng/Documents/MyIsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/robot_description/go2_robot_description.yaml",
        urdf_path="D:/Users/yucheng/Documents/MyIsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/robot_description/go2_description.urdf"
    )
    end_effector_name = "RR_foot"

    articulation_kinematics_solver = ArticulationKinematicsSolver(robot, kinematics_solver, end_effector_name)

    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_playing():
            target_position, target_orientation = target.get_world_pose()
            action, success = articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)

            if success:
                robot.apply_action(action)
            else:
                carb.log_warn("IK did not converge to a solution.  No action is being taken")


if __name__ == "__main__":
    main()

   # 確保模擬在結束時正確關閉
    simulation_app.close()

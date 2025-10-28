# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.storage.native import get_assets_root_path

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.policy.go2_policy import Go2FlatTerrainPolicy
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist


class Go2LocomotionNode(Node):
    def __init__(self):
        super().__init__('go2_locomotion_ros')

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.obs_publisher = self.create_publisher(
            Float64MultiArray,
            '/observation',
            sensor_qos
        )

        self.joint_state_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            sensor_qos
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            '/imu',
            sensor_qos
        )

        self.command_publisher = self.create_publisher(
            Twist,
            '/command',
            reliable_qos
        )

        self.action_subscriber = self.create_subscription(
            Float64MultiArray,
            '/action',
            self.action_callback,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

        self._action = None  # 用於存儲接收到的 action

    def publish_observation(self, observation: np.ndarray):
        msg = Float64MultiArray()
        msg.data = observation.tolist()
        self.obs_publisher.publish(msg)

    def publish_joint_states(self, joint_names: list, joint_positions: np.ndarray, joint_velocities: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = joint_positions.tolist()
        msg.velocity = joint_velocities.tolist()
        self.joint_state_publisher.publish(msg)

    def publish_imu(self, imu_values: dict):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.linear_acceleration.x = float(imu_values["linear_acceleration"][0])
        msg.linear_acceleration.y = float(imu_values["linear_acceleration"][1])
        msg.linear_acceleration.z = float(imu_values["linear_acceleration"][2])
        msg.angular_velocity.x = float(imu_values["angular_velocity"][0])
        msg.angular_velocity.y = float(imu_values["angular_velocity"][1])
        msg.angular_velocity.z = float(imu_values["angular_velocity"][2])
        msg.orientation.w = float(imu_values["orientation"][0])
        msg.orientation.x = float(imu_values["orientation"][1])
        msg.orientation.y = float(imu_values["orientation"][2])
        msg.orientation.z = float(imu_values["orientation"][3])
        self.imu_publisher.publish(msg)

    def publish_command(self, command: np.ndarray):
        msg = Twist()
        msg.linear.x = command[0]
        msg.linear.y = command[1]
        msg.angular.z = command[2]
        self.command_publisher.publish(msg)

    def action_callback(self, msg):
        self._action = np.array(msg.data)

    @property
    def action(self):
        if self._action is not None:
            return np.array([self._action[index:index + 3] for index in range(0, len(self._action), 3)]).T.flatten()

        return None


class Go2_runner(object):
    node: Go2LocomotionNode

    def __init__(self, render_dt, training_folder, node) -> None:
        """
        creates the simulation world with preset physics_dt and render_dt and creates an Go2 robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.

        """

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # spawn warehouse scene
        prim = define_prim("/World/Ground", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        prim.GetReferences().AddReference(asset_path)

        self._go2 = Go2FlatTerrainPolicy(
            prim_path="/World/go2",
            training_folder=training_folder,
            name="go2",
            usd_path=assets_root_path + "/Isaac/Robots/Unitree/Go2/go2.usd",
            position=np.array([0, 0, 0.5]),
        )

        self._world = World(stage_units_in_meters=1.0, physics_dt=self._go2.physics_dt, rendering_dt=render_dt)

        self._base_command = np.zeros(3)

        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        self.needs_reset = False
        self.first_step = True

        self.node = node

    def setup(self) -> None:
        """
        Set up keyboard listener and add physics callback

        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("go2_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        """
        Physics call back, initialize robot (first frame) and call controller forward function to compute and apply joint torque

        """
        if self.first_step:
            self._go2.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            rclpy.spin_once(self.node, timeout_sec=0)  # Non-blocking spin to process incoming messages
            self._go2.forward(step_size, self._base_command, action=self.node.action)
            # self.node.publish_observation(self._go2.observation)
            self.node.publish_joint_states(
                self._go2.joint_names,
                self._go2.current_absolute_joint_positions,
                self._go2.current_joint_velocities,
            )
            self.node.publish_imu(self._go2.imu_values)
            self.node.publish_command(self._base_command)

    def run(self) -> None:
        """
        Step simulation based on rendering downtime

        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        Keyboard subscriber callback to when kit is updated.

        """

        # when a key is pressed for released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


def main():
    """
    Parse arguments and instantiate the Go2 runner

    """

    parser = argparse.ArgumentParser(description="Go2 Locomotion ROS Demo")
    parser.add_argument("--training_folder", type=str, default="", help="Path to the training folder containing the policy and env yaml")
    args = parser.parse_args()

    rclpy.init()
    node = Go2LocomotionNode()

    training_folder = args.training_folder
    if training_folder == "":
        carb.log_error("Please provide a valid training folder path containing the policy and env yaml")
        return

    render_dt = 1 / 60.0

    runner = Go2_runner(render_dt=render_dt, training_folder=training_folder, node=node)
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    runner.run()
    simulation_app.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

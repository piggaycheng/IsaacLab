import numpy as np

import pinocchio as pin
import pink
from pink import solve_ik as pink_solve_ik, FrameTask
import qpsolvers
from loop_rate_limiters import RateLimiter


class InverseKinematicsSolver:
    def __init__(
        self,
        robot_wrapper: pin.RobotWrapper,
        ee_name_list: list[str],
        q_ref: np.ndarray | None = None,
        rate=50.0,
        solver="proxqp",
    ):
        """
        Initialize the inverse kinematics solver.
        Args:
            urdf_filename (str): Path to the URDF file of the robot.
            package_dirs (list[str], optional): List of package directories for resolving URDF dependencies.
            q_ref (np.ndarray, optional): Reference joint configuration for the robot.
        """

        self._robot = robot_wrapper

        q_mid_range = (
            self._robot.model.lowerPositionLimit + self._robot.model.upperPositionLimit
        ) / 2.0
        q = q_ref if q_ref is not None else q_mid_range
        self._configuration = pink.Configuration(self._robot.model, self._robot.data, q)

        self.rate_limiter = RateLimiter(rate)

        self.solver = qpsolvers.available_solvers[0]
        if solver in qpsolvers.available_solvers:
            self.solver = solver
        else:
            print(f"Warning: {solver} is not available. Using {self.solver} instead.")

        self.task_dict = {}

        for ee_name in ee_name_list:
            task = FrameTask(
                ee_name,
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=0.0,  # [cost] / [rad]
                lm_damping=1.0e-4,
            )
            self.task_dict[ee_name] = task

        for task in self.task_dict.values():
            task.set_target_from_configuration(self._configuration)

    def solve_ik(self, ee_name, ee_target_pos, curr_q) -> np.ndarray:
        """
        Solve the inverse kinematics problem to find the next joint configuration.

        Returns:
            np.ndarray: The next joint configuration.
        """
        dt = self.rate_limiter.period

        target_rot = np.identity(3)
        target_pos = np.array(ee_target_pos)
        target_pose = pin.SE3(target_rot, target_pos)  # type: ignore
        task = self.task_dict[ee_name]
        task.set_target(target_pose)

        # 更新目前的關節角度
        clipped_q = np.clip(
            curr_q,
            self._robot.model.lowerPositionLimit,
            self._robot.model.upperPositionLimit,
        )
        self._configuration.update(clipped_q)

        velocity = pink_solve_ik(
            self._configuration,
            [task],
            dt,
            solver=self.solver,
        )
        return self._configuration.integrate(velocity, dt)

    @property
    def configuration(self):
        return self._configuration

    @property
    def robot(self):
        return self._robot


def get_pin_robot_wrapper(
    urdf_filename: str, package_dirs: list[str] | str | None = None, root_joint=None
) -> pin.RobotWrapper:
    """
    Load a robot from a URDF file and return a Pinocchio RobotWrapper.
    Args:
        urdf_filename (str): Path to the URDF file of the robot.
        package_dirs (list[str] or str, optional): List of package directories for resolving URDF dependencies.
        root_joint (pin.JointModel, optional): The root joint model for the robot.
    Returns:
        RobotWrapper: The loaded robot wrapped in a Pinocchio RobotWrapper.
    """
    if package_dirs is None:
        package_dirs = []
    elif isinstance(package_dirs, str):
        package_dirs = [package_dirs]

    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_filename,
        package_dirs=package_dirs,
        root_joint=root_joint,
    )

    return robot

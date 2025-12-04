import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


class InverseKinematicsSolver:
    def __init__(
        self,
        urdf_path: str,
        ee_name: str,
        damping: float = 0.01,
    ):
        """
        Initialize the Inverse Kinematics Solver using Pinocchio (Standalone).

        Args:
            urdf_path (str): Path to the URDF file.
            ee_name (str): Name of the end-effector frame.
            hip_name (str): Name of the hip frame (origin for target_pos).
            damping (float): Damping factor for DLS (lambda).
        """
        # 1. Load Model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 2. Get Frame ID
        if not self.model.existFrame(ee_name):
            raise ValueError(f"Frame '{ee_name}' does not exist in the URDF.")
        self.ee_id = self.model.getFrameId(ee_name)

        # 3. Parameters
        self.damping_sq = damping**2

        # 4. Cache Joint Limits
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

        self.neutral_q = np.array(
            [
                0.0,
                0.95995,
                -1.78023,
                0.0,
                0.95995,
                -1.78023,
                0.0,
                0.95995,
                -1.78023,
                0.0,
                0.95995,
                -1.78023,
            ],
            dtype=np.float64,
        )

    def compute(self, q_current: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Compute the next joint configuration using Differential IK (DLS).

        Args:
            q_current (np.ndarray): Current joint configuration (Full robot state).
            target_pos (np.ndarray): Target position [x, y, z] (relative to Hip frame).
            target_quat (np.ndarray): Target orientation quaternion [x, y, z, w] (in Base frame).

        Returns:
            np.ndarray: The next joint configuration for all joints.
        """
        # 正向運動學 & Jacobian
        pin.framesForwardKinematics(self.model, self.data, q_current)
        pin.computeJointJacobians(self.model, self.data, q_current)

        # 獲取 6xN 的 Jacobian
        J_full = pin.getFrameJacobian(
            self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED
        )

        # 只取 Linear 部分的 Rows (6 -> 3)
        J_pos = J_full[:3, :]

        # --- 誤差計算 ---
        # 獲取當前 End-Effector 在 Base Frame 的位置
        curr_pos = self.data.oMf[self.ee_id].translation

        # 計算誤差 (在 Base Frame 下)
        error = target_pos - curr_pos

        # DLS 求解
        J_JT = J_pos @ J_pos.T
        damped_matrix = J_JT + self.damping_sq * np.eye(3)

        dq = J_pos.T @ np.linalg.solve(damped_matrix, error)

        # 積分
        q_next = pin.integrate(self.model, q_current, dq)

        # 關節限制
        q_next = np.clip(q_next, self.q_min, self.q_max)

        return q_next


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

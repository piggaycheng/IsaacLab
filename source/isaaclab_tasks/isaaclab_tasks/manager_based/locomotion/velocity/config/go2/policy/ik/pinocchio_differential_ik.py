import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


class InverseKinematicsSolver:
    def __init__(self, urdf_path: str, ee_name: str, dt: float = 0.01, damping: float = 0.05):
        """
        Initialize the Inverse Kinematics Solver using Pinocchio (Standalone).

        Args:
            urdf_path (str): Path to the URDF file.
            ee_name (str): Name of the end-effector frame.
            dt (float): Control time step (seconds).
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
        self.dt = dt
        self.damping_sq = damping**2

        # 4. Cache Joint Limits
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

    def compute(self, q_current: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """
        Compute the next joint configuration using Differential IK (DLS).

        Args:
            q_current (np.ndarray): Current joint configuration.
            target_pos (np.ndarray): Target position [x, y, z] (in Base frame).
            target_quat (np.ndarray): Target orientation quaternion [x, y, z, w] (in Base frame).

        Returns:
            np.ndarray: The next joint configuration.
        """
        # --- A. Forward Kinematics ---
        pin.framesForwardKinematics(self.model, self.data, q_current)
        pin.computeJointJacobians(self.model, self.data, q_current)

        # --- B. Get Jacobian ---
        # Use LOCAL_WORLD_ALIGNED to match the error calculation in World frame
        J = pin.getJointJacobian(
            self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # --- C. Compute Error (6D) ---
        # 1. Get current EE pose
        curr_transform = self.data.oMf[self.ee_id]
        curr_pos = curr_transform.translation
        curr_rot = curr_transform.rotation

        # 2. Position Error
        pos_err = target_pos - curr_pos

        # 3. Orientation Error
        # Convert target quaternion to rotation matrix
        # Note: scipy Rotation expects [x, y, z, w]
        target_rot = R.from_quat(target_quat).as_matrix()

        # Calculate rotation error in Local frame: R_diff = R_current.T * R_target
        # log3 converts rotation matrix difference to axis-angle vector
        rot_err_local = pin.log3(curr_rot.T @ target_rot)

        # Convert error back to World/Base frame
        rot_err = curr_rot @ rot_err_local

        # 4. Combined Error Vector
        error = np.concatenate([pos_err, rot_err])

        # --- D. Damped Least Squares (DLS) ---
        # dq = J^T * (J * J^T + lambda^2 * I)^-1 * error
        m = J.shape[0]  # Typically 6
        J_JT = J @ J.T
        damped_matrix = J_JT + self.damping_sq * np.eye(m)

        try:
            lambda_term = np.linalg.solve(damped_matrix, error)
            dq = J.T @ lambda_term
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed, stopping motion.")
            dq = np.zeros_like(q_current)

        # --- E. Integration and Clipping ---
        q_next = pin.integrate(self.model, q_current, dq * self.dt)
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

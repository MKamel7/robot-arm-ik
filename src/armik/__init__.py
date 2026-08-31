"""armik: forward/inverse kinematics and trajectory planning for a serial arm."""

from .robot import SerialArm, dh_transform, UR5_DH, UR5_JOINT_LIMITS
from .rotations import matrix_from_rotvec, rotvec_from_matrix, slerp, skew
from .ik import solve_ik, pose_error, manipulability, IKResult
from .analytical import analytical_ik
from .select import Branch, follow, nearest_equivalent, rank_branches, score_branch, select_branch
from .trajectory import (
    joint_trajectory,
    multi_waypoint_trajectory,
    path_trajectory,
    cartesian_line,
)
from .rrt import rrt_connect

__all__ = [
    "SerialArm", "dh_transform", "UR5_DH", "UR5_JOINT_LIMITS",
    "matrix_from_rotvec", "rotvec_from_matrix", "slerp", "skew",
    "solve_ik", "pose_error", "manipulability", "IKResult", "analytical_ik",
    "Branch", "follow", "nearest_equivalent", "rank_branches", "score_branch",
    "select_branch",
    "joint_trajectory", "multi_waypoint_trajectory", "path_trajectory",
    "cartesian_line",
    "rrt_connect",
]

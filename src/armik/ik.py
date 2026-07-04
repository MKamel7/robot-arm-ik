"""Inverse kinematics by damped least squares (Levenberg-Marquardt on the Jacobian).

Numerical IK: iteratively shrink the 6D pose error (position and orientation)
with the update

    dq = J^T (J J^T + lambda^2 I)^-1 e

The damping term lambda keeps the step bounded near singular configurations,
where a plain Jacobian pseudo-inverse would demand enormous joint velocities and
diverge. The UR5's home pose is exactly such a singularity, which makes this a
real test of the solver rather than a synthetic one.
"""

from dataclasses import dataclass

import numpy as np

from .robot import SerialArm
from .rotations import rotvec_from_matrix


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-vector error [position(3); orientation(3)] from current pose to target.

    The orientation part is the rotation vector of R_target R_current^T,
    expressed in the base frame so it pairs with the base-frame geometric
    Jacobian's angular rows.
    """
    e_p = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    e_o = rotvec_from_matrix(R_err)
    return np.concatenate([e_p, e_o])


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


def solve_ik(arm: SerialArm, T_target: np.ndarray, q_init: np.ndarray, *,
             damping: float = 0.05, max_iters: int = 200,
             pos_tol: float = 1e-4, rot_tol: float = 1e-3,
             step_clamp: float = 0.5) -> IKResult:
    """Solve IK for T_target starting from q_init.

    Damped least squares with per-iteration step clamping (the step magnitude is
    capped so the linearisation stays valid). Returns an IKResult; success is
    True once both position and orientation errors fall within tolerance.
    """
    q = np.array(q_init, dtype=float)
    lam2 = damping ** 2
    I6 = np.eye(6)
    for it in range(1, max_iters + 1):
        T = arm.fk(q)
        e = pose_error(T, T_target)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        if pos_err < pos_tol and rot_err < rot_tol:
            # Return the exact q that satisfied the tolerance. Limits are enforced
            # per-iteration below (not as a post-hoc clamp of a converged solution,
            # which would silently move the pose off the target it just reached).
            return IKResult(q, True, it, pos_err, rot_err)
        J = arm.jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * I6, e)
        norm = np.linalg.norm(dq)
        if norm > step_clamp:
            dq *= step_clamp / norm
        # Clamp each iterate into the joint limits. A target only reachable via an
        # out-of-limit angle then simply never converges (success stays False),
        # which is the honest outcome rather than a limit-violating "solution".
        q = arm.clamp(q + dq)
    T = arm.fk(q)
    e = pose_error(T, T_target)
    return IKResult(q, False, max_iters,
                    float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))


def manipulability(arm: SerialArm, q: np.ndarray) -> float:
    """Yoshikawa manipulability sqrt(det(J J^T)); goes to 0 at a singularity."""
    J = arm.jacobian(q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

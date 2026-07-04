"""Rotation utilities: rotation-vector (axis-angle) <-> matrix, and SLERP.

Kept dependency-light (numpy only) and numerically robust at the awkward
angles (near 0 and near pi) that a naive implementation gets wrong.
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that skew(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def matrix_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (axis * angle) via Rodrigues."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) from a rotation matrix.

    Handles the two degenerate cases explicitly: angle ~ 0 (undefined axis,
    returns zero) and angle ~ pi (the antisymmetric part vanishes, so the
    axis is recovered from the symmetric part instead).
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if angle < 1e-8:
        return np.zeros(3)

    if np.pi - angle < 1e-6:
        # Near pi: R = 2 a aᵀ - I, so (R + I)/2 = a aᵀ. Recover |a_i| from the
        # diagonal and fix relative signs from the row of the dominant axis.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-8:
            axis = A[k, :] / axis[k]
        axis = axis / np.linalg.norm(axis)
        return angle * axis

    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return angle * axis


def slerp(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Interpolate between two rotation matrices along the shortest geodesic."""
    R_rel = R0.T @ R1
    v = rotvec_from_matrix(R_rel)
    return R0 @ matrix_from_rotvec(s * v)

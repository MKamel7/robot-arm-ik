"""Serial-manipulator model: DH parameters, forward kinematics, Jacobian.

The default arm is the Universal Robots UR5, whose standard Denavit-Hartenberg
parameters are published by the manufacturer, so using a real robot keeps every
number verifiable rather than invented.
"""

from dataclasses import dataclass, field

import numpy as np

# UR5 standard DH parameters: (a, alpha, d) per joint; theta is the joint variable.
UR5_DH = [
    (0.0,      np.pi / 2, 0.089159),
    (-0.425,   0.0,       0.0),
    (-0.39225, 0.0,       0.0),
    (0.0,      np.pi / 2, 0.10915),
    (0.0,     -np.pi / 2, 0.09465),
    (0.0,      0.0,       0.0823),
]

# Joint limits (rad): UR5 allows +/- 2pi; keep the classic +/- 2pi range.
UR5_JOINT_LIMITS = np.array([[-2 * np.pi, 2 * np.pi]] * 6)


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard Denavit-Hartenberg homogeneous transform for one link."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


@dataclass
class SerialArm:
    """A revolute serial manipulator defined by DH parameters."""

    dh: list = field(default_factory=lambda: list(UR5_DH))
    joint_limits: np.ndarray = field(default_factory=lambda: UR5_JOINT_LIMITS.copy())

    @property
    def n(self) -> int:
        return len(self.dh)

    def frames(self, q: np.ndarray) -> list:
        """Cumulative base-to-frame transforms [T0_0, T0_1, ..., T0_n].

        Length n+1: T0_0 is the base (identity), T0_n is the end-effector.
        """
        T = np.eye(4)
        out = [T]
        for (a, alpha, d), theta in zip(self.dh, q):
            T = T @ dh_transform(a, alpha, d, theta)
            out.append(T)
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        """End-effector pose T0_n (4x4) for joint configuration q."""
        return self.frames(q)[-1]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian (6 x n) in the base frame.

        Column i for a revolute joint: [ z_{i-1} x (p_e - p_{i-1}) ; z_{i-1} ],
        where z and p come from the cumulative frame preceding that joint.
        """
        frames = self.frames(q)
        p_e = frames[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            z = frames[i][:3, 2]
            p = frames[i][:3, 3]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        return J

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp a configuration to the joint limits."""
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def random_config(self, rng: np.random.Generator, margin: float = 0.1) -> np.ndarray:
        """A random configuration strictly inside the joint limits."""
        lo = self.joint_limits[:, 0] + margin
        hi = self.joint_limits[:, 1] - margin
        return rng.uniform(lo, hi)

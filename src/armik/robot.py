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

# UR5e standard DH parameters. Same 6R topology as the UR5, but a taller base
# (d1) and slightly different wrist offsets (d4, d5, d6). Cross-validated against
# the MuJoCo Menagerie `universal_robots_ur5e` model: identity joint mapping,
# tool position agrees to 1.5 mm worst case and 0.98 mm mean across random
# configurations, orientation to 4e-6 degrees. Asserted in
# tests/test_ur5e_mujoco.py, which loads the model and does the comparison.
# The residual is the Menagerie XML rounding link lengths to the millimetre
# against the datasheet values below, not an error in either model.
UR5E_DH = [
    (0.0,      np.pi / 2, 0.1625),
    (-0.425,   0.0,       0.0),
    (-0.3922,  0.0,       0.0),
    (0.0,      np.pi / 2, 0.1333),
    (0.0,     -np.pi / 2, 0.0997),
    (0.0,      0.0,       0.0996),
]

# Joint limits (rad): UR arms allow +/- 2pi; keep the classic +/- 2pi range.
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


def dh_transform_modified(a: float, alpha: float, d: float,
                          theta: float) -> np.ndarray:
    """Modified (Craig) Denavit-Hartenberg transform for one link.

    NOT interchangeable with `dh_transform` above, and mixing them silently
    produces a plausible-looking arm with the wrong geometry. The difference is
    where the link twist is applied: standard DH rotates about x AFTER
    translating along z, Craig rotates about x FIRST. A parameter table is only
    meaningful together with the convention it was written for.

    This exists because Franka Emika publishes the Panda's parameters in Craig
    form, and retyping them into a standard-DH transform would be the exact
    mistake this docstring is warning about.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,      -st,       0.0,       a],
        [st * ca,  ct * ca, -sa, -d * sa],
        [st * sa,  ct * sa,  ca,  d * ca],
        [0.0,      0.0,      0.0,     1.0],
    ])


# Franka Emika Panda, 7R, in MODIFIED (Craig) DH as the manufacturer publishes
# it: (a, alpha, d) per joint. The flange offset is folded into the last link's
# d, so `fk` returns the flange rather than joint 7's frame.
#
# NOT CROSS-VALIDATED against an external model, unlike the UR5e. There is no
# Franka asset vendored here, so this geometry rests on the published table
# rather than on agreement with a simulator, and that is a weaker footing. The
# UR5e's 1.5 mm agreement with MuJoCo is a measurement; this is a transcription.
# Said here rather than left for a reader to assume otherwise.
PANDA_DH = [
    (0.0,     0.0,        0.333),
    (0.0,    -np.pi / 2,  0.0),
    (0.0,     np.pi / 2,  0.316),
    (0.0825,  np.pi / 2,  0.0),
    (-0.0825, -np.pi / 2, 0.384),
    (0.0,     np.pi / 2,  0.0),
    (0.088,   np.pi / 2,  0.107),
]

# The Panda's real limits, and they matter for redundancy rather than being
# decoration: joints 4 and 6 are ASYMMETRIC about zero, so a null-space
# controller that pushes every joint toward the midpoint of its range moves
# them somewhere a symmetric arm would not need to go. A test asserts the
# asymmetry survives, because replacing these with a tidy plus-or-minus band
# would quietly delete the interesting half of the problem.
PANDA_JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])


@dataclass
class SerialArm:
    """A revolute serial manipulator defined by DH parameters."""

    dh: list = field(default_factory=lambda: list(UR5_DH))
    joint_limits: np.ndarray = field(default_factory=lambda: UR5_JOINT_LIMITS.copy())
    #: Which DH convention `dh` is written in. Carried on the arm rather than
    #: passed at call time so a parameter table cannot be separated from the
    #: transform that makes sense of it.
    modified_dh: bool = False

    @classmethod
    def ur5(cls) -> "SerialArm":
        """The Universal Robots UR5 (the library default), stated explicitly."""
        return cls(dh=list(UR5_DH), joint_limits=UR5_JOINT_LIMITS.copy())

    @classmethod
    def ur5e(cls) -> "SerialArm":
        """The Universal Robots UR5e. Same solver and Jacobian, UR5e geometry.

        Its FK is cross-validated against the MuJoCo Menagerie UR5e model to
        1.5 mm worst case (tests/test_ur5e_mujoco.py), so the joint angles this
        arm's IK produces drive that model directly.
        """
        return cls(dh=[list(link) for link in UR5E_DH],
                   joint_limits=UR5_JOINT_LIMITS.copy())

    @classmethod
    def panda(cls) -> "SerialArm":
        """The Franka Emika Panda: 7R, and therefore REDUNDANT.

        Seven joints against a six-dimensional task leaves a one-dimensional
        null space at every non-singular configuration, which is the whole
        reason this arm is here. `armik.redundancy` uses it to do something
        useful with that freedom; on the 6R arms there is nothing to use.

        Its geometry is transcribed from the published Craig-form table and is
        NOT cross-validated against a simulator, unlike `ur5e()`. See PANDA_DH.
        """
        return cls(dh=[list(link) for link in PANDA_DH],
                   joint_limits=PANDA_JOINT_LIMITS.copy(),
                   modified_dh=True)

    @property
    def n(self) -> int:
        return len(self.dh)

    def frames(self, q: np.ndarray) -> list:
        """Cumulative base-to-frame transforms [T0_0, T0_1, ..., T0_n].

        Length n+1: T0_0 is the base (identity), T0_n is the end-effector.
        """
        transform = dh_transform_modified if self.modified_dh else dh_transform
        T = np.eye(4)
        out = [T]
        for (a, alpha, d), theta in zip(self.dh, q):
            T = T @ transform(a, alpha, d, theta)
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

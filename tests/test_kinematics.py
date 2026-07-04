"""Forward kinematics and Jacobian correctness."""

import numpy as np

from armik import SerialArm
from armik.rotations import rotvec_from_matrix


def test_fk_returns_valid_se3():
    arm = SerialArm()
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = arm.random_config(rng)
        T = arm.fk(q)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)   # orthonormal
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)  # proper rotation
        assert np.allclose(T[3], [0, 0, 0, 1])


def test_fk_within_reach():
    arm = SerialArm()
    rng = np.random.default_rng(2)
    # The tool position can never exceed the sum of the link lengths, which is
    # the true kinematic upper bound (~1.19 m for the UR5). The oft-quoted
    # "0.85 m reach" is the nominal working radius, not this hard maximum.
    max_reach = sum(abs(a) + abs(d) for a, alpha, d in arm.dh)
    for _ in range(200):
        q = arm.random_config(rng)
        p = arm.fk(q)[:3, 3]
        assert np.linalg.norm(p) <= max_reach + 1e-9


def test_jacobian_matches_finite_difference():
    """The analytic geometric Jacobian must match a finite-difference of FK,
    checked across several random configurations (not just one)."""
    arm = SerialArm()
    rng = np.random.default_rng(3)
    eps = 1e-6
    for _ in range(10):
        q = arm.random_config(rng)
        J = arm.jacobian(q)
        T0 = arm.fk(q)
        for i in range(arm.n):
            dq = np.zeros(arm.n)
            dq[i] = eps
            T1 = arm.fk(q + dq)
            # linear rows: d(position)/dq_i
            dv = (T1[:3, 3] - T0[:3, 3]) / eps
            assert np.allclose(dv, J[:3, i], atol=1e-4)
            # angular rows: rotation vector of the incremental rotation, /eps
            dw = rotvec_from_matrix(T1[:3, :3] @ T0[:3, :3].T) / eps
            assert np.allclose(dw, J[3:, i], atol=1e-4)


def test_fk_matches_known_ur5_pose():
    """Golden value: pin FK to an external reference so a DH transcription error
    is caught. The UR5 tool position at the all-zero configuration is a published,
    reproducible number; the self-consistent tests (FD-Jacobian, reach bound) can
    all pass with a wrong DH table, this one cannot."""
    arm = SerialArm()
    p = arm.fk(np.zeros(6))[:3, 3]
    assert np.allclose(p, [-0.81725, -0.19145, -0.00549], atol=1e-5)


def test_home_is_singular():
    """The UR5 home configuration is a known singularity (rank drops below 6)."""
    arm = SerialArm()
    J = arm.jacobian(np.zeros(6))
    assert np.linalg.matrix_rank(J, tol=1e-6) < 6

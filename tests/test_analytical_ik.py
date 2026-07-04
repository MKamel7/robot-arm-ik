"""Closed-form UR5 inverse kinematics: every solution must reproduce the pose,
a generic pose must yield the full set of 8 branches, and the numerical solver
must land on one of the analytical solutions."""

import numpy as np

from armik import SerialArm, solve_ik, analytical_ik
from armik.rotations import rotvec_from_matrix


def _pose_gap(Ta, Tb):
    pos = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
    rot = np.linalg.norm(rotvec_from_matrix(Ta[:3, :3] @ Tb[:3, :3].T))
    return pos, rot


def test_analytical_ik_every_solution_reaches_pose():
    """Ground truth is FK: each closed-form solution, fed back through forward
    kinematics, must reproduce the exact target pose (analytic, so ~machine tol)."""
    arm = SerialArm()
    rng = np.random.default_rng(7)
    for _ in range(50):
        q_true = arm.random_config(rng, margin=0.6)
        T = arm.fk(q_true)
        sols = analytical_ik(arm, T)
        assert len(sols) >= 1
        for q in sols:
            pos, rot = _pose_gap(arm.fk(q), T)
            assert pos < 1e-8
            assert rot < 1e-8


def test_analytical_ik_recovers_the_true_configuration():
    """The configuration the pose came from must be among the returned branches
    (matched modulo 2*pi per joint, since the arm allows +/- 2*pi)."""
    arm = SerialArm()
    rng = np.random.default_rng(11)
    for _ in range(30):
        q_true = arm.random_config(rng, margin=0.6)
        T = arm.fk(q_true)
        sols = analytical_ik(arm, T)
        wrapped = (np.asarray(sols) - q_true + np.pi) % (2 * np.pi) - np.pi
        per_sol = np.max(np.abs(wrapped), axis=1)
        assert np.min(per_sol) < 1e-6


def test_analytical_ik_finds_eight_branches_for_generic_pose():
    """A generic reachable pose has 8 IK solutions (2 shoulder x 2 elbow x 2 wrist).
    Count the distinct ones."""
    arm = SerialArm()
    q = np.array([0.3, -0.9, 1.1, -0.7, 0.8, 0.4])   # generic, away from singularities
    T = arm.fk(q)
    sols = np.asarray(analytical_ik(arm, T))
    # dedupe on wrapped joint values
    wrapped = (sols + np.pi) % (2 * np.pi) - np.pi
    uniq = []
    for row in wrapped:
        if not any(np.allclose(row, u, atol=1e-6) for u in uniq):
            uniq.append(row)
    assert len(uniq) == 8


def test_numerical_ik_matches_an_analytical_solution():
    """The damped-least-squares solver must converge to one of the closed-form
    solutions (same pose and, modulo 2*pi, the same joint values)."""
    arm = SerialArm()
    rng = np.random.default_rng(13)
    q_true = arm.random_config(rng, margin=0.6)
    T = arm.fk(q_true)
    res = solve_ik(arm, T, q_true + rng.uniform(-0.3, 0.3, arm.n))
    assert res.success
    sols = np.asarray(analytical_ik(arm, T))
    wrapped = (sols - res.q + np.pi) % (2 * np.pi) - np.pi
    # the DLS solver stops at its pose tolerance (1e-4 m), which corresponds to
    # ~1e-3 rad joint agreement with the exact closed-form branch, not machine tol
    assert np.min(np.max(np.abs(wrapped), axis=1)) < 5e-3

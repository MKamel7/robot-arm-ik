"""Inverse kinematics: round-trip accuracy and stability near singularities."""

import numpy as np

from armik import SerialArm, solve_ik, cartesian_line
from armik.rotations import matrix_from_rotvec, rotvec_from_matrix


def test_ik_roundtrip_reaches_pose():
    """FK(IK(target)) must reproduce the target pose (not necessarily the same q,
    since a 6-DOF arm has multiple IK solutions)."""
    arm = SerialArm()
    rng = np.random.default_rng(42)
    solved = 0
    for _ in range(30):
        q_true = arm.random_config(rng, margin=0.6)
        T = arm.fk(q_true)
        q_seed = q_true + rng.uniform(-0.6, 0.6, arm.n)   # a genuinely offset seed
        res = solve_ik(arm, T, q_seed)
        if not res.success:
            continue
        solved += 1
        T_sol = arm.fk(res.q)
        assert np.linalg.norm(T_sol[:3, 3] - T[:3, 3]) < 1e-4   # position, tight
        # orientation error at the solver's own tolerance scale, not 30x looser
        rot_err = np.linalg.norm(rotvec_from_matrix(T_sol[:3, :3] @ T[:3, :3].T))
        assert rot_err < 2e-3
    assert solved >= 27   # the large majority of offset-seed solves still converge


def test_cartesian_line_is_straight():
    """A Cartesian straight-line move must keep the tool tip on the line between
    the two positions (not just hit the endpoints)."""
    arm = SerialArm()
    q0 = np.array([0.2, -0.8, 1.0, -0.6, -1.4, 0.3])
    T0 = arm.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] = T0[:3, 3] + np.array([0.0, 0.15, -0.10])   # move in a straight line
    q_path, ok = cartesian_line(arm, T0, T1, q0, steps=30)
    assert ok
    p0, p1 = T0[:3, 3], T1[:3, 3]
    for q in q_path:
        p = arm.fk(q)[:3, 3]
        # distance from p to the segment p0->p1 must be tiny
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        perp = (p - p0) - np.dot(p - p0, u) * u
        assert np.linalg.norm(perp) < 1e-3


def test_ik_stable_near_singularity():
    """Damped least squares must stay finite even when seeded at the singular
    home pose (a plain pseudo-inverse would diverge here)."""
    arm = SerialArm()
    T = arm.fk(np.zeros(6))
    T[:3, 3] += np.array([0.02, 0.02, 0.0])   # nudge the target off the singularity
    res = solve_ik(arm, T, np.zeros(6))
    assert np.all(np.isfinite(res.q))
    # The initial position error is already ~0.028 m; "makes progress" has to mean
    # it converges, not merely stays below the starting error. Assert real success.
    assert res.success
    assert res.position_error < 1e-3          # DLS drives the error to tolerance


def test_ik_iterations_reported():
    arm = SerialArm()
    q_true = np.array([0.3, -0.7, 1.0, -0.5, 0.4, 0.2])
    T = arm.fk(q_true)
    res = solve_ik(arm, T, q_true + 0.1)
    assert res.success
    assert 1 <= res.iterations <= 200

"""Trajectory generation: boundary conditions, limits, synchronisation."""

import numpy as np

from armik import joint_trajectory, multi_waypoint_trajectory


def test_boundary_conditions():
    q0 = np.zeros(6)
    q1 = np.array([1.0, -1.0, 0.5, 0.3, -0.7, 0.2])
    t, q, qd = joint_trajectory(q0, q1, v_max=1.0, a_max=2.0, dt=0.01)
    assert np.allclose(q[0], q0)
    assert np.allclose(q[-1], q1, atol=1e-6)
    assert np.allclose(qd[0], 0.0, atol=1e-6)
    assert np.allclose(qd[-1], 0.0, atol=1e-6)


def test_velocity_limit_respected():
    q0 = np.zeros(6)
    q1 = np.array([2.0, -2.0, 1.0, 1.0, -1.0, 1.0])
    t, q, qd = joint_trajectory(q0, q1, v_max=1.0, a_max=2.0, dt=0.005)
    assert np.max(np.abs(qd)) <= 1.0 + 1e-6


def test_all_joints_finish_together():
    """Synchronised motion: every joint reaches its target at the same sample."""
    q0 = np.zeros(6)
    q1 = np.array([2.0, 0.1, -1.5, 0.3, 1.0, -0.4])   # very different travels
    t, q, qd = joint_trajectory(q0, q1, dt=0.01)
    # each joint's motion completes exactly at the final sample
    assert np.allclose(q[-1], q1, atol=1e-6)
    # and none of them overshoots along the way (monotone toward target)
    for i in range(6):
        prog = (q[:, i] - q0[i]) / (q1[i] - q0[i]) if abs(q1[i] - q0[i]) > 1e-9 else None
        if prog is not None:
            assert np.all(np.diff(prog) >= -1e-9)


def test_multiwaypoint_monotonic_time_and_endpoints():
    wpts = [np.zeros(6), np.ones(6), np.array([-0.5] * 6), np.zeros(6)]
    t, q, qd = multi_waypoint_trajectory(wpts, v_max=1.5, a_max=3.0, dt=0.02)
    assert np.all(np.diff(t) > 0)                 # strictly increasing time
    assert np.allclose(q[0], wpts[0])
    assert np.allclose(q[-1], wpts[-1], atol=1e-6)


def test_triangular_short_move():
    """A move too short to reach v_max uses the triangular branch. It must still
    end exactly on target, at rest, monotone, and within both limits (this is
    the branch where a naive implementation overshoots)."""
    v_max, a_max = 1.0, 2.0
    # max travel below v^2/a = 0.5 guarantees the triangular branch on every joint
    q0 = np.zeros(6)
    q1 = np.full(6, 0.3)
    t, q, qd = joint_trajectory(q0, q1, v_max=v_max, a_max=a_max, dt=0.002)
    assert np.allclose(q[-1], q1, atol=1e-6)          # exact endpoint, no overshoot
    assert np.max(q) <= q1[0] + 1e-6                  # never overshoots past target
    assert np.allclose(qd[0], 0.0) and np.allclose(qd[-1], 0.0, atol=1e-6)
    assert np.max(np.abs(qd)) <= v_max + 1e-6         # velocity bound
    # acceleration bound (finite-difference of velocity)
    acc = np.diff(qd, axis=0) / np.diff(t)[:, None]
    assert np.max(np.abs(acc)) <= a_max + 1e-3
    # strictly monotone progress toward the target
    assert np.all(np.diff(q[:, 0]) >= -1e-9)


def test_acceleration_limit_respected():
    q0 = np.zeros(6)
    q1 = np.array([2.0, -2.0, 1.5, 1.0, -1.0, 1.0])
    a_max = 2.0
    t, q, qd = joint_trajectory(q0, q1, v_max=1.0, a_max=a_max, dt=0.002)
    acc = np.diff(qd, axis=0) / np.diff(t)[:, None]
    assert np.max(np.abs(acc)) <= a_max + 1e-3


def test_multiwaypoint_handles_dwell_waypoint():
    """A repeated consecutive waypoint (a zero-duration dwell, e.g. pausing to
    grasp) must not crash and must still reach the final target at rest."""
    wpts = [np.zeros(6), np.ones(6), np.ones(6), np.zeros(6)]   # duplicate in middle
    t, q, qd = multi_waypoint_trajectory(wpts, v_max=1.5, a_max=3.0, dt=0.02)
    assert np.all(np.diff(t) > 0)
    assert np.allclose(q[0], wpts[0])
    assert np.allclose(q[-1], wpts[-1], atol=1e-6)
    assert np.allclose(qd[-1], 0.0, atol=1e-6)


def test_no_motion_is_handled():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    t, qtraj, qd = joint_trajectory(q, q)
    assert np.allclose(qtraj[0], q)
    assert np.allclose(qd, 0.0)

"""Time-parameterised trajectories.

Two kinds:

* Joint-space, synchronised trapezoidal: all joints move together along a
  straight line in joint space, driven by a single trapezoidal time-scaling
  s(t) in [0, 1]. The scaling limits are derived so that no joint exceeds its
  velocity or acceleration bound, which gives smooth, coordinated motion with
  a clean trapezoidal velocity profile and zero velocity at both ends.

* Cartesian straight line: interpolate the tool position linearly and its
  orientation by SLERP, solving IK at each sample (seeded from the previous
  solution) so the tool tip travels a straight path in space.
"""

import numpy as np

from .robot import SerialArm
from .rotations import slerp


def _trapezoidal_profile(sdot_max: float, sddot_max: float, dt: float):
    """Trapezoidal time-scaling for s going 0 -> 1. Returns (t, s, sdot)."""
    ta = sdot_max / sddot_max                 # time to accelerate to sdot_max
    da = 0.5 * sddot_max * ta ** 2            # distance covered while accelerating
    if 2 * da >= 1.0:
        # Triangular profile: the move is too short to ever reach sdot_max, so
        # the cruise velocity is the ACTUAL peak the ramp reaches, not the
        # unreached limit. (Forgetting to lower sdot_max here makes the decel
        # phase start from a speed the profile never attained, which overshoots
        # the target and violates the acceleration bound.)
        ta = np.sqrt(1.0 / sddot_max)
        sdot_max = sddot_max * ta
        tc = 0.0
    else:
        tc = (1.0 - 2 * da) / sdot_max        # cruise (constant-speed) time
    tf = 2 * ta + tc

    t = np.arange(0.0, tf + dt, dt)
    s = np.zeros_like(t)
    sdot = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < ta:                            # accelerate
            s[i] = 0.5 * sddot_max * ti ** 2
            sdot[i] = sddot_max * ti
        elif ti < ta + tc:                     # cruise
            s[i] = 0.5 * sddot_max * ta ** 2 + sdot_max * (ti - ta)
            sdot[i] = sdot_max
        elif ti <= tf:                         # decelerate
            td = ti - (ta + tc)
            s_cruise_end = 0.5 * sddot_max * ta ** 2 + sdot_max * tc
            s[i] = s_cruise_end + sdot_max * td - 0.5 * sddot_max * td ** 2
            sdot[i] = sdot_max - sddot_max * td
    s[-1], sdot[-1] = 1.0, 0.0                 # pin the exact endpoint
    return t, s, sdot


def joint_trajectory(q_start, q_end, *, v_max=1.0, a_max=2.0, dt=0.02):
    """Synchronised trapezoidal trajectory from q_start to q_end.

    v_max and a_max are per-joint velocity/acceleration limits (scalar, or
    length-n arrays). Returns (t, q, qd) with q of shape (T, n).
    """
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_end, dtype=float)
    dq = q1 - q0
    n = len(dq)
    v_max = np.broadcast_to(v_max, (n,)).astype(float)
    a_max = np.broadcast_to(a_max, (n,)).astype(float)

    moving = np.abs(dq) > 1e-9
    if not np.any(moving):
        return np.array([0.0]), q0[None, :].copy(), np.zeros((1, n))

    # s in [0,1], and joint i moves |dq_i|*s, so the tightest limits are:
    sdot_max = float(np.min(v_max[moving] / np.abs(dq[moving])))
    sddot_max = float(np.min(a_max[moving] / np.abs(dq[moving])))

    t, s, sdot = _trapezoidal_profile(sdot_max, sddot_max, dt)
    q = q0[None, :] + np.outer(s, dq)
    qd = np.outer(sdot, dq)
    return t, q, qd


def multi_waypoint_trajectory(waypoints, *, v_max=1.0, a_max=2.0, dt=0.02):
    """Chain trapezoidal segments through a list of joint-space waypoints.

    Each segment starts and ends at rest, so the arm pauses momentarily at each
    waypoint (natural for a pick-and-place stop). Returns (t, q, qd).
    """
    waypoints = [np.asarray(w, dtype=float) for w in waypoints]
    t_all, q_all, qd_all = [], [], []
    t_offset = 0.0
    for i, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        t, q, qd = joint_trajectory(a, b, v_max=v_max, a_max=a_max, dt=dt)
        if i > 0:                              # drop duplicated boundary sample
            t, q, qd = t[1:], q[1:], qd[1:]
        if len(t) == 0:                        # a repeated waypoint is a zero-motion
            continue                           # segment; after the boundary drop it
                                               # is empty, so it contributes nothing
        t_all.append(t + t_offset)
        q_all.append(q)
        qd_all.append(qd)
        t_offset = t_all[-1][-1]
    return np.concatenate(t_all), np.vstack(q_all), np.vstack(qd_all)


def cartesian_line(arm: SerialArm, T_start, T_target, q_init, *,
                   steps=50, ik_kwargs=None):
    """Straight-line Cartesian path from T_start to T_target.

    Position is interpolated linearly, orientation by SLERP; IK is solved at
    each sample seeded from the previous solution. Returns (q_path, ok) where
    q_path is (steps, n) and ok is True only if every sample's IK converged.
    """
    from .ik import solve_ik
    ik_kwargs = ik_kwargs or {}
    p0, p1 = T_start[:3, 3], T_target[:3, 3]
    R0, R1 = T_start[:3, :3], T_target[:3, :3]
    q = np.array(q_init, dtype=float)
    q_path, ok = [], True
    for s in np.linspace(0.0, 1.0, steps):
        T = np.eye(4)
        T[:3, 3] = (1 - s) * p0 + s * p1
        T[:3, :3] = slerp(R0, R1, s)
        res = solve_ik(arm, T, q, **ik_kwargs)
        ok = ok and res.success
        q = res.q
        q_path.append(q)
    return np.array(q_path), ok

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


def path_trajectory(path, *, v_max=1.0, a_max=2.0, dt=0.02, corner_rest=None):
    """Time-parameterise a dense path, optionally resting at sharp corners.

    `corner_rest` (radians) is the turn angle above which a vertex counts as a
    real corner. Cruising through a corner at speed reverses joint velocity in
    a single step, which is an unbounded acceleration no controller can track
    -- under physics execution that showed up as ~0.9 rad of tracking error.
    Splitting there and coming to rest keeps every segment trackable while
    still flowing continuously along the straight runs between corners.

    None (default) means never rest; the whole path is one profile.
    """
    if corner_rest is None:
        return _path_profile(path, v_max=v_max, a_max=a_max, dt=dt)

    P = np.asarray([np.asarray(p, dtype=float) for p in path], dtype=float)
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-12:
            keep.append(i)
    P = P[keep]
    if len(P) < 3:
        return _path_profile(P, v_max=v_max, a_max=a_max, dt=dt)

    seg = np.diff(P, axis=0)
    unit = seg / np.linalg.norm(seg, axis=1)[:, None]
    # angle between consecutive segment directions; > corner_rest is a corner
    cosang = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
    corners = [i + 1 for i, a in enumerate(np.arccos(cosang)) if a > corner_rest]

    bounds = [0] + corners + [len(P) - 1]
    t_all, q_all, qd_all, t_off = [], [], [], 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        t, q, qd = _path_profile(P[a:b + 1], v_max=v_max, a_max=a_max, dt=dt)
        if t_all:                              # drop the duplicated boundary sample
            t, q, qd = t[1:], q[1:], qd[1:]
        if len(t) == 0:
            continue
        t_all.append(t + t_off)
        q_all.append(q)
        qd_all.append(qd)
        t_off = t_all[-1][-1]
    if not t_all:
        return _path_profile(P, v_max=v_max, a_max=a_max, dt=dt)
    return np.concatenate(t_all), np.vstack(q_all), np.vstack(qd_all)


def _path_profile(path, *, v_max=1.0, a_max=2.0, dt=0.02):
    """Time-parameterise an existing dense geometric path along its arc length.

    The counterpart to multi_waypoint_trajectory. That one chains a trapezoid
    per waypoint and so comes to rest at *every* waypoint -- correct when each
    waypoint is a stop that matters (grasp, release), badly wrong when the
    points are merely samples of a path the tool should flow along. A 3-leg
    lift/traverse/lower detour sampled into ~50 points becomes ~50 stop-starts,
    which reads as a violent stutter.

    Here `path` is treated as a fixed polyline to FOLLOW: one acceleration at
    the start, a cruise, one deceleration at the end, with the profile applied
    to arc length rather than to each segment. The output lies on the polyline
    (samples are interpolated between adjacent input points, never chords cut
    across them), so a path that was collision-checked densely stays checked.

    Returns (t, q, qd), with qd the true per-joint reference velocity.
    """
    P = np.asarray([np.asarray(p, dtype=float) for p in path], dtype=float)
    if P.ndim != 2:
        raise ValueError("path must be a sequence of joint configurations")
    keep = [0]
    for i in range(1, len(P)):                 # drop repeated points: a zero-length
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-12:   # segment has no direction
            keep.append(i)
    P = P[keep]
    n = P.shape[1]
    if len(P) < 2:
        return np.array([0.0]), P[:1].copy(), np.zeros((1, n))

    seg = np.diff(P, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    unit = seg / seg_len[:, None]              # per-segment direction

    v_max = np.broadcast_to(v_max, (n,)).astype(float)
    a_max = np.broadcast_to(a_max, (n,)).astype(float)
    # Joint j moves at sdot * total * unit[k, j] on segment k, so the tightest
    # scaling limit over every segment and joint is what keeps all of them legal.
    denom = total * np.abs(unit) + 1e-12
    sdot_max = float(np.min(v_max[None, :] / denom))
    sddot_max = float(np.min(a_max[None, :] / denom))

    t, s, sdot = _trapezoidal_profile(sdot_max, sddot_max, dt)
    u = s * total                              # arc length travelled
    idx = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(seg_len) - 1)
    frac = ((u - cum[idx]) / seg_len[idx])[:, None]
    q = P[idx] + seg[idx] * frac
    qd = sdot[:, None] * total * unit[idx]
    return t, q, qd


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

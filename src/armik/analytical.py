"""Closed-form inverse kinematics for the UR5.

The UR5 is a 6R arm with a spherical-ish wrist geometry whose inverse kinematics
has a known analytic solution: for a generic reachable pose there are exactly
eight joint configurations that reach it (two shoulder, two elbow, two wrist
branches). Unlike the numerical damped-least-squares solver in `ik.py`, this
returns *all* of them in closed form, with no seed and no iteration.

This is also the strongest possible check on the numerical solver and the DH
model: every analytic solution, pushed back through forward kinematics, must
reproduce the target pose to machine precision.

Derivation follows R. S. Andersen, "Kinematics of a UR5" (Aalborg University,
2018), which uses exactly the standard DH parameters in `robot.py`.
"""

import numpy as np

from .robot import SerialArm, dh_transform


_UNIT_EPS = 1e-9


def _acos(x: float) -> float:
    """acos that absorbs sub-epsilon float overshoot but returns NaN for a
    genuinely out-of-range argument, so an unreachable branch is dropped rather
    than clamped into a wrong 'solution'."""
    if x > 1.0 + _UNIT_EPS or x < -1.0 - _UNIT_EPS:
        return np.nan
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def _asin(x: float) -> float:
    """asin with the same reachability semantics as `_acos`."""
    if x > 1.0 + _UNIT_EPS or x < -1.0 - _UNIT_EPS:
        return np.nan
    return float(np.arcsin(np.clip(x, -1.0, 1.0)))


def analytical_ik(arm: SerialArm, T_target: np.ndarray) -> np.ndarray:
    """All closed-form UR5 IK solutions for the tool pose T_target (4x4).

    Returns an (k, 6) array of joint configurations, k <= 8 (fewer only if a
    branch is unreachable). Valid for the standard 6-DOF UR5 DH model; the DH
    constants are read from `arm` so it stays consistent with forward kinematics.
    Angles are principal values and may be matched to another configuration
    modulo 2*pi per joint.
    """
    if arm.n != 6:
        raise ValueError("analytical_ik is specific to the 6-DOF UR5")

    # DH constants, pulled from the arm so FK and IK cannot drift apart. (d1 and
    # d5 do not appear: they cancel in this frame-relative formulation.)
    a2 = arm.dh[1][0]
    a3 = arm.dh[2][0]
    d4 = arm.dh[3][2]
    d6 = arm.dh[5][2]

    def Ti(i: int, theta: float) -> np.ndarray:
        a, alpha, d = arm.dh[i]
        return dh_transform(a, alpha, d, theta)

    T = np.asarray(T_target, dtype=float)
    th = np.zeros((6, 8))

    # --- theta1: two shoulder solutions -------------------------------------
    p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
    psi = np.arctan2(p05[1], p05[0])
    r05 = np.hypot(p05[0], p05[1])
    phi = _acos(d4 / r05)
    th[0, 0:4] = np.pi / 2 + psi + phi
    th[0, 4:8] = np.pi / 2 + psi - phi

    # --- theta5: two wrist-flip solutions per theta1 ------------------------
    for c in (0, 4):
        if np.isnan(th[0, c]):
            continue
        T16 = np.linalg.inv(Ti(0, th[0, c])) @ T
        a5 = _acos((T16[2, 3] - d4) / d6)
        th[4, c:c + 2] = a5
        th[4, c + 2:c + 4] = -a5

    # --- theta6 (undefined when sin(theta5)=0; pinned to 0 there) -----------
    for c in (0, 2, 4, 6):
        if np.isnan(th[4, c]):
            continue
        T61 = np.linalg.inv(np.linalg.inv(Ti(0, th[0, c])) @ T)
        s5 = np.sin(th[4, c])
        if abs(s5) < 1e-8:
            th[5, c:c + 2] = 0.0
        else:
            th[5, c:c + 2] = np.arctan2(-T61[1, 2] / s5, T61[0, 2] / s5)

    # --- theta3: two elbow solutions per (theta1, theta5) -------------------
    for c in (0, 2, 4, 6):
        if np.isnan(th[4, c]) or np.isnan(th[5, c]):
            continue
        T10 = np.linalg.inv(Ti(0, th[0, c]))
        T46 = Ti(4, th[4, c]) @ Ti(5, th[5, c])
        T14 = T10 @ T @ np.linalg.inv(T46)
        p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
        L = np.linalg.norm(p13)
        # Out of 2-link elbow reach -> that (shoulder, wrist) branch has no real
        # elbow solution; `_acos` returns NaN and the branch is dropped below.
        t3 = _acos((L ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3))
        th[2, c] = t3
        th[2, c + 1] = -t3

    # --- theta2, theta4: fixed once the elbow branch is chosen --------------
    for c in range(8):
        if np.isnan(th[2, c]):
            th[:, c] = np.nan
            continue
        T10 = np.linalg.inv(Ti(0, th[0, c]))
        T46 = Ti(4, th[4, c]) @ Ti(5, th[5, c])
        T14 = T10 @ T @ np.linalg.inv(T46)
        p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
        L = np.linalg.norm(p13)
        th[1, c] = -np.arctan2(p13[1], -p13[0]) + _asin(a3 * np.sin(th[2, c]) / L)
        T34 = np.linalg.inv(Ti(2, th[2, c])) @ np.linalg.inv(Ti(1, th[1, c])) @ T14
        th[3, c] = np.arctan2(T34[1, 0], T34[0, 0])

    sols = th.T
    return sols[~np.isnan(sols).any(axis=1)]

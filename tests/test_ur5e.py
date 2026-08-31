"""UR5e model support.

The library was written for the UR5; the UR5e is the same 6R topology with a
slightly taller base and different wrist offsets. These tests pin the UR5e
forward kinematics against frozen values and confirm the existing IK solver
works unchanged on the UR5e.

The cross-validation against the MuJoCo Menagerie `universal_robots_ur5e`
model lives in tests/test_ur5e_mujoco.py, which loads the model and does the
comparison. Nothing in this file does, and this docstring used to imply it.
"""

import numpy as np

from armik import SerialArm, solve_ik


def test_ur5e_constructor():
    arm = SerialArm.ur5e()
    assert arm.n == 6
    # UR5e is distinct from the default UR5 (taller base d1).
    assert not np.isclose(arm.dh[0][2], SerialArm().dh[0][2])


def test_ur5e_fk_golden():
    """fk(0) at the zero pose, pinned against regression.

    This is a frozen constant at 1e-9, so it catches any change to the DH
    table but says nothing on its own about whether the table is right. What
    ties it to reality is test_ur5e_mujoco.py, which checks this same value
    against where the MuJoCo model actually puts the flange.
    """
    arm = SerialArm.ur5e()
    T = arm.fk(np.zeros(6))
    assert np.allclose(T[:3, 3], [-0.8172, -0.2329, 0.0628], atol=1e-9)
    assert np.allclose(T[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-9)


def test_ur5e_ik_roundtrip():
    """The damped-least-squares IK reaches poses generated from UR5e FK.

    Seeded near the solution (an offset seed), matching the UR5 IK test; a few
    near-singular configurations may not converge, so the large majority must.
    """
    arm = SerialArm.ur5e()
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(30):
        q_true = arm.random_config(rng, margin=0.6)
        T = arm.fk(q_true)
        q_seed = q_true + rng.uniform(-0.6, 0.6, arm.n)
        res = solve_ik(arm, T, q_seed)
        if not res.success:
            continue
        solved += 1
        assert np.linalg.norm(arm.fk(res.q)[:3, 3] - T[:3, 3]) < 1e-4
    assert solved >= 27

"""Redundancy resolution on a 7R arm, and the Panda geometry it runs on.

The tests that matter here:

  * `test_nullspace_motion_does_not_move_the_tool` is the property the whole
    method rests on. If null-space motion disturbs the task then the secondary
    objective is not free and everything downstream is wrong.
  * `test_a_six_joint_arm_has_no_freedom_to_spend` is the control. A redundancy
    module that appeared to do something useful on a non-redundant arm would be
    computing nonsense somewhere, and this is what catches that.
  * `test_too_much_null_space_gain_is_worse_than_none` records a real finding
    rather than a tuned-away inconvenience. The controller integrates in
    discrete steps, so a large null-space step overshoots the hill it is
    climbing. The objective is not wrong; the gain has an optimum and past it
    the result is worse than doing nothing.
  * `test_the_panda_flange_matches_the_published_ready_pose` and
    `test_the_panda_reach_matches_the_published_figure` are the only external
    checks on a DH table typed in from a datasheet. Without them the geometry
    rests on nobody having made a typo.
"""

from __future__ import annotations

import numpy as np
import pytest

from armik.redundancy import (
    combined,
    compare,
    damped_pseudo_inverse,
    follow,
    joint_limit_gradient,
    joint_limit_margin,
    manipulability,
    manipulability_gradient,
    nullspace_projector,
    redundancy_dimension,
    resolve,
)
from armik.robot import SerialArm

READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2,
                  np.pi / 4])


def _drag(steps: int = 60) -> list[np.ndarray]:
    """A straight-line Cartesian drag that walks the arm toward its limits."""
    return [np.array([0.012, 0.0, -0.004, 0.0, 0.0, 0.0])
            for _ in range(steps)]


# --- the Panda geometry, checked against something outside this repository ---
def test_the_panda_flange_matches_the_published_ready_pose() -> None:
    """A DH table typed from a datasheet is a transcription until checked.

    The Panda's flange at the standard ready configuration is widely published
    as approximately [0.307, 0, 0.590]. Agreement to a millimetre says the
    table, the convention and the flange offset are all right together, which
    no internal consistency check could establish.
    """
    flange = SerialArm.panda().fk(READY)[:3, 3]
    assert flange == pytest.approx([0.307, 0.0, 0.590], abs=1e-3)


def test_the_panda_reach_matches_the_published_figure() -> None:
    """Franka publishes 855 mm, measured from the shoulder rather than the base.

    A second, independent check on the same table: the ready pose fixes one
    configuration, this one exercises the whole joint range.
    """
    arm = SerialArm.panda()
    shoulder = arm.frames(np.zeros(arm.n))[1][:3, 3]
    rng = np.random.default_rng(0)
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    reach = max(float(np.linalg.norm(arm.fk(rng.uniform(low, high))[:3, 3]
                                     - shoulder)) for _ in range(4000))
    assert reach == pytest.approx(0.855, abs=0.01)


def test_the_panda_joint_limits_are_asymmetric() -> None:
    """Joints 4 and 6 do not straddle zero, and that is the interesting half.

    Replacing them with a tidy plus-or-minus band would make limit avoidance a
    trivial pull toward zero and delete the problem.
    """
    limits = SerialArm.panda().joint_limits
    assert limits[3][1] < 0.0, "joint 4 is entirely negative"
    assert limits[5][0] > -0.1, "joint 6 barely reaches zero"


def test_the_two_dh_conventions_are_not_interchangeable() -> None:
    """Mixing them yields a plausible arm with the wrong geometry.

    Reading the Panda's Craig-form table with the standard transform is exactly
    the mistake `dh_transform_modified` exists to prevent, so it is worth
    seeing it produce a different arm rather than assuming it would.
    """
    from armik.robot import PANDA_DH, PANDA_JOINT_LIMITS

    wrong = SerialArm(dh=[list(link) for link in PANDA_DH],
                      joint_limits=PANDA_JOINT_LIMITS.copy(),
                      modified_dh=False)
    assert not np.allclose(wrong.fk(READY)[:3, 3],
                           SerialArm.panda().fk(READY)[:3, 3], atol=1e-3)


# --- the property everything else rests on -----------------------------------
def test_nullspace_motion_does_not_move_the_tool() -> None:
    """If it did, the secondary objective would not be free."""
    arm = SerialArm.panda()
    J = arm.jacobian(READY)
    projector = nullspace_projector(J)
    rng = np.random.default_rng(1)
    for _ in range(20):
        motion = projector @ rng.normal(size=arm.n)
        assert np.linalg.norm(J @ motion) < 1e-3


def test_a_seven_joint_arm_has_exactly_one_spare_dimension() -> None:
    assert redundancy_dimension(SerialArm.panda(), READY) == 1


def test_a_six_joint_arm_has_no_freedom_to_spend() -> None:
    """The control. There is nothing to project onto, and that is the truth."""
    arm = SerialArm.ur5e()
    q = np.array([0.1, -1.0, 1.2, -0.5, 1.4, 0.3])
    assert redundancy_dimension(arm, q) == 0
    assert np.allclose(nullspace_projector(arm.jacobian(q)), 0.0, atol=1e-3)


def test_a_secondary_objective_does_nothing_on_a_non_redundant_arm() -> None:
    """The same statement one level up, at the interface a caller uses."""
    arm = SerialArm.ur5e()
    q = np.array([0.1, -1.0, 1.2, -0.5, 1.4, 0.3])
    twist = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    plain = resolve(arm, q, twist)
    with_objective = resolve(arm, q, twist, objective=joint_limit_gradient,
                             gain=1000.0)
    assert np.allclose(plain, with_objective, atol=1e-3)


# --- the measures ------------------------------------------------------------
def test_the_margin_is_a_half_at_the_middle_of_every_range() -> None:
    arm = SerialArm.panda()
    mid = 0.5 * (arm.joint_limits[:, 0] + arm.joint_limits[:, 1])
    assert joint_limit_margin(arm, mid) == pytest.approx(0.5)


def test_the_margin_is_zero_on_a_hard_stop() -> None:
    arm = SerialArm.panda()
    q = 0.5 * (arm.joint_limits[:, 0] + arm.joint_limits[:, 1])
    q[3] = arm.joint_limits[3, 0]
    assert joint_limit_margin(arm, q) == pytest.approx(0.0)


def test_the_margin_is_normalised_by_each_joints_own_range() -> None:
    """Otherwise the wider joint is always reported as the safer one."""
    arm = SerialArm.panda()
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    q = low + 0.25 * (high - low)
    assert joint_limit_margin(arm, q) == pytest.approx(0.25)


def test_the_limit_gradient_points_toward_the_middle() -> None:
    arm = SerialArm.panda()
    mid = 0.5 * (arm.joint_limits[:, 0] + arm.joint_limits[:, 1])
    q = mid.copy()
    q[0] += 1.0
    assert joint_limit_gradient(arm, q)[0] < 0.0, "must pull back down"
    q[0] = mid[0] - 1.0
    assert joint_limit_gradient(arm, q)[0] > 0.0


def test_the_limit_gradient_vanishes_at_the_middle() -> None:
    arm = SerialArm.panda()
    mid = 0.5 * (arm.joint_limits[:, 0] + arm.joint_limits[:, 1])
    assert np.allclose(joint_limit_gradient(arm, mid), 0.0)


def test_the_manipulability_gradient_actually_points_uphill() -> None:
    """A gradient with the wrong sign steers the arm into the singularity."""
    arm = SerialArm.panda()
    grad = manipulability_gradient(arm, READY)
    direction = grad / np.linalg.norm(grad)
    before = manipulability(arm, READY)
    for step in (1e-3, 1e-2, 1e-1):
        assert manipulability(arm, READY + step * direction) > before


def test_manipulability_falls_toward_a_singularity() -> None:
    arm = SerialArm.panda()
    stretched = np.zeros(arm.n)
    assert manipulability(arm, stretched) < manipulability(arm, READY)


def test_the_damped_inverse_is_a_right_inverse_of_a_wide_jacobian() -> None:
    arm = SerialArm.panda()
    J = arm.jacobian(READY)
    assert (J @ damped_pseudo_inverse(J)) == pytest.approx(np.eye(6), abs=1e-2)


# --- what the controller actually buys ---------------------------------------
def test_limit_avoidance_improves_the_worst_margin() -> None:
    """The claim, at the gain the benchmark identifies as the good one."""
    arm = SerialArm.panda()
    baseline, controlled = compare(arm, READY, _drag(),
                                   objective=joint_limit_gradient, gain=5.0)
    assert controlled.worst_margin > baseline.worst_margin * 1.5


def test_the_task_is_still_done_while_the_objective_runs() -> None:
    """A secondary objective that degrades the primary one is not free."""
    arm = SerialArm.panda()
    _, controlled = compare(arm, READY, _drag(),
                            objective=joint_limit_gradient, gain=5.0)
    assert controlled.worst_task_error < 1e-3


def test_manipulability_maximisation_improves_manipulability() -> None:
    arm = SerialArm.panda()
    baseline, controlled = compare(arm, READY, _drag(),
                                   objective=manipulability_gradient, gain=0.2)
    assert controlled.mean_manipulability > baseline.mean_manipulability


def test_too_much_null_space_gain_is_worse_than_none() -> None:
    """A finding, recorded rather than tuned away.

    The controller integrates in discrete steps, so a large null-space step
    overshoots the hill it is climbing and lands somewhere worse than where it
    started. The objective is not wrong and the gradient is not wrong; the gain
    has an optimum. Anyone reaching for a bigger number should meet this first.
    """
    arm = SerialArm.panda()
    baseline = follow(arm, READY, _drag())
    excessive = follow(arm, READY, _drag(),
                       objective=manipulability_gradient, gain=8.0)
    assert excessive.mean_manipulability < baseline.mean_manipulability


def test_an_enormous_gain_drives_joints_onto_their_stops() -> None:
    """The same lesson at the other objective, and worse.

    At gain 300 the limit-avoidance controller, whose entire purpose is to stay
    away from the stops, puts a joint on one. That is what makes the gain a
    parameter worth measuring rather than guessing.
    """
    arm = SerialArm.panda()
    reckless = follow(arm, READY, _drag(),
                      objective=joint_limit_gradient, gain=300.0)
    assert reckless.worst_margin == pytest.approx(0.0, abs=1e-6)


def test_the_two_objectives_can_be_combined_with_stated_weights() -> None:
    arm = SerialArm.panda()
    mixed = combined((joint_limit_gradient, 5.0),
                     (manipulability_gradient, 0.2))
    trace = follow(arm, READY, _drag(), objective=mixed, gain=1.0)
    assert trace.configurations.shape == (60, arm.n)


def test_a_zero_weight_removes_an_objective_entirely() -> None:
    arm = SerialArm.panda()
    only_limits = combined((joint_limit_gradient, 1.0),
                           (manipulability_gradient, 0.0))
    assert np.allclose(only_limits(arm, READY),
                       joint_limit_gradient(arm, READY))


def test_the_trace_keeps_the_worst_as_well_as_the_mean() -> None:
    """A controller improving the average while pinning one joint on a stop has
    not helped, and only the worst value shows it."""
    arm = SerialArm.panda()
    trace = follow(arm, READY, _drag())
    assert trace.worst_margin <= float(np.mean(trace.margins))
    assert trace.margins.shape == (60,)
    assert trace.manipulabilities.shape == (60,)


def test_following_clips_to_the_joint_limits() -> None:
    """A limit-avoidance result measured on a path that walked through a limit
    and came back would be meaningless."""
    arm = SerialArm.panda()
    trace = follow(arm, READY, _drag(), objective=joint_limit_gradient,
                   gain=300.0)
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    assert np.all(trace.configurations >= low - 1e-9)
    assert np.all(trace.configurations <= high + 1e-9)

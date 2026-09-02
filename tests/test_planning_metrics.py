"""Path metrics must describe the motion, not the planner's output format.

The comparison these back puts an interpolation that returns two waypoints
beside a sampling planner that returns forty and an optimiser that returns
sixty. If a metric moves when only the waypoint count changes, the resulting
table ranks output formats.
"""

import numpy as np
import pytest

from armik.planning_metrics import (
    joint_path_length,
    resample,
    smoothness,
    task_path_length,
)


def _line(n, dim=7):
    return np.linspace(np.zeros(dim), np.ones(dim), n)


def test_a_straight_line_is_perfectly_smooth_whatever_its_length():
    """The headline property: this measures shape, not distance."""
    assert smoothness(_line(2)) < 1e-12
    assert smoothness(np.linspace(np.zeros(7), 5 * np.ones(7), 2)) < 1e-12


def test_waypoint_count_does_not_change_the_score():
    """The same geometric path sampled two ways scores the same.

    Not bitwise: summing 59 short steps and one long one differ in the last
    ulp, and both smoothness values land around 1e-32 rather than at zero. The
    claim is that the difference is numerical noise and not a ranking.
    """
    coarse = _line(2)
    fine = _line(60)

    assert smoothness(coarse) == pytest.approx(smoothness(fine), abs=1e-20)
    assert joint_path_length(coarse) == pytest.approx(joint_path_length(fine), rel=1e-12)


def test_a_zigzag_is_less_smooth_than_a_line():
    zig = np.array([[0.0] * 7, [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]], float)

    assert smoothness(zig) > smoothness(_line(3))


def test_resampling_is_equal_arc_length_not_equal_index():
    """A path with one long and one short segment resamples uniformly in space."""
    path = np.array([[0.0, 0.0], [0.1, 0.0], [1.1, 0.0]])
    out = resample(path, 11)
    spacing = np.linalg.norm(np.diff(out, axis=0), axis=1)

    assert np.allclose(spacing, spacing[0], atol=1e-9)


def test_a_path_that_goes_nowhere_is_not_an_error():
    """Start equal to goal is a legitimate request, and has no arc length."""
    still = np.zeros((4, 7))

    assert joint_path_length(still) == 0.0
    assert smoothness(still) == 0.0
    assert resample(still, 20).shape == (20, 7)


def test_task_length_is_metres_and_joint_length_is_radians():
    """They are different questions and can rank two paths differently."""
    positions = np.array([[0.0, 0, 0], [0.3, 0, 0], [0.3, 0.4, 0]])

    assert task_path_length(positions) == 0.7


def test_a_path_needs_two_waypoints():
    for bad in (np.zeros((1, 7)), np.zeros(7)):
        try:
            joint_path_length(bad)
        except ValueError:
            continue
        raise AssertionError("a single waypoint is not a path")


def test_distance_to_a_box_is_zero_inside_and_positive_outside():
    from armik.planning_metrics import distance_to_box

    centre, half = (0.0, 0.0, 0.0), (0.1, 0.1, 0.1)

    assert distance_to_box([[0.0, 0.0, 0.0]], centre, half) == 0.0
    assert distance_to_box([[0.3, 0.0, 0.0]], centre, half) == pytest.approx(0.2)
    assert distance_to_box([[0.1, 0.1, 0.1]], centre, half) == pytest.approx(0.0)


def test_distance_to_a_box_uses_the_nearest_point():
    from armik.planning_metrics import distance_to_box

    points = [[5.0, 0, 0], [0.4, 0, 0], [9.0, 0, 0]]

    assert distance_to_box(points, (0, 0, 0), (0.1, 0.1, 0.1)) == pytest.approx(0.3)


def test_link_points_lie_along_the_arm_and_include_the_base():
    from armik.planning_metrics import link_points
    from armik.robot import SerialArm

    arm = SerialArm.panda()
    q = np.zeros(7)
    points = link_points(arm, q, per_link=4)

    # Eight frames give seven segments; each contributes per_link - 1 points
    # after the shared endpoint, plus the base itself.
    assert len(points) == 1 + 7 * 3
    assert np.allclose(points[0], arm.frames(q)[0][:3, 3])
    assert np.allclose(points[-1], arm.fk(q)[:3, 3])


def test_clearance_falls_when_a_path_passes_closer_to_a_box():
    """The property the benchmark needs: it must rank two paths differently."""
    from armik.planning_metrics import clearance
    from armik.robot import SerialArm

    arm = SerialArm.panda()
    near = np.linspace(np.zeros(7), np.array([0.0, 0.4, 0.0, -0.4, 0.0, 0.8, 0.0]), 10)
    box_close = [((0.4, 0.0, 0.5), (0.05, 0.05, 0.05))]
    box_far = [((1.5, 0.0, 0.5), (0.05, 0.05, 0.05))]

    assert clearance(arm, near, box_close) < clearance(arm, near, box_far)

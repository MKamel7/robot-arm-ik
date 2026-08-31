"""Branch selection: the chosen configuration must be reachable and sane.

Two properties matter and they are different. The chosen configuration has to
be a real solution to the pose, which is a correctness claim about the shift
applied to it. And it has to be the one a controller should be given, which is
a claim about the cost. Both are tested; the second is the one a naive
selector gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from armik import SerialArm, analytical_ik, manipulability
from armik.select import (
    SINGULAR_FLOOR,
    TAU,
    Branch,
    follow,
    nearest_equivalent,
    rank_branches,
    score_branch,
    select_branch,
)


@pytest.fixture
def arm():
    return SerialArm.ur5()


def reachable_pose(arm, q):
    return arm.fk(np.asarray(q, dtype=float))


# ---- the 2-pi shift ---------------------------------------------------------

def test_a_solution_is_moved_to_the_equivalent_nearest_the_current_pose(arm):
    """The defect this module exists to avoid.

    A joint at +3.0 rad and a solution reported at -3.0 rad describe the same
    arm. Taking the principal value literally is a 6 rad unwind to reach a pose
    the arm was 0.28 rad away from.
    """
    principal = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    shifted = nearest_equivalent(principal, current, arm.joint_limits)

    assert shifted[0] == pytest.approx(-3.0 + TAU)
    assert abs(shifted[0] - current[0]) < 0.3
    assert abs(principal[0] - current[0]) > 5.9


def test_the_shift_is_refused_when_it_would_leave_the_joint_travel(arm):
    """A wrist that would have to pass a limit keeps its principal value."""
    tight = SerialArm.ur5()
    tight.joint_limits = np.array([[-np.pi, np.pi]] * 6)

    principal = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    shifted = nearest_equivalent(principal, current, tight.joint_limits)

    assert shifted[0] == pytest.approx(-3.0), "shifted outside the joint travel"


def test_the_shift_preserves_the_pose(arm):
    """A 2-pi shift is the same arm, so FK must be unchanged."""
    q = arm.random_config(np.random.default_rng(3), margin=0.5)
    reference = q + TAU * np.array([1, 0, -1, 0, 1, 0])

    shifted = nearest_equivalent(q, reference, arm.joint_limits)

    assert np.allclose(arm.fk(shifted), arm.fk(q), atol=1e-9)


# ---- the choice -------------------------------------------------------------

def test_the_selected_branch_actually_reaches_the_pose(arm):
    """Everything else is worthless if the answer is not a solution.

    Some random configurations are genuinely near-singular: about 2.5% of them
    fall below SINGULAR_FLOOR, and for those the honest answer is that there is
    no branch worth handing a controller. So a refusal is allowed, and it is
    asserted to happen only for that reason rather than being waved through.
    """
    rng = np.random.default_rng(0)
    reached = refused = 0

    for _ in range(60):
        q_true = arm.random_config(rng, margin=0.7)
        target = reachable_pose(arm, q_true)

        chosen = select_branch(arm, analytical_ik(arm, target), q_true)

        if chosen is None:
            assert manipulability(arm, q_true) < SINGULAR_FLOOR, (
                "refused a pose that was not near-singular")
            refused += 1
            continue
        assert np.allclose(arm.fk(chosen.q), target, atol=1e-6)
        reached += 1

    assert reached > 50, f"only {reached} of 60 poses got a branch"
    assert refused < 6, f"refused {refused} of 60, more than the singular rate"


def test_it_prefers_the_configuration_the_arm_is_already_near(arm):
    """Travel dominates when the branches are otherwise comparable."""
    rng = np.random.default_rng(1)
    q_true = arm.random_config(rng, margin=0.8)
    target = reachable_pose(arm, q_true)
    nearby = q_true + rng.uniform(-0.05, 0.05, 6)

    chosen = select_branch(arm, analytical_ik(arm, target), nearby)

    assert chosen is not None
    assert chosen.travel < 1.0, f"walked {chosen.travel:.2f} rad to a pose it was at"


def test_a_branch_outside_the_joint_limits_is_rejected_not_ranked_last(arm):
    """Infeasible is a different answer from expensive.

    A cost term cannot express "not at any price": raise the weight enough and
    an out-of-limits configuration still wins when every alternative is worse.
    """
    tight = SerialArm.ur5()
    tight.joint_limits = np.array([[-0.2, 0.2]] * 6)
    q = np.array([1.5, -1.0, 1.0, -0.5, 0.5, 0.0])

    branch = score_branch(tight, q, np.zeros(6))

    assert not branch.feasible
    assert branch.rejected == "outside joint limits"
    assert branch.cost == float("inf")


def test_a_singular_configuration_is_rejected_outright(arm):
    """The UR5 home pose is genuinely singular, which is why it is used here."""
    branch = score_branch(arm, np.zeros(6), np.zeros(6))

    assert not branch.feasible
    assert branch.rejected == "singular"


def test_selection_returns_none_rather_than_the_least_bad_infeasible_answer(arm):
    tight = SerialArm.ur5()
    tight.joint_limits = np.array([[-0.05, 0.05]] * 6)
    q_true = SerialArm.ur5().random_config(np.random.default_rng(4), margin=0.8)
    target = reachable_pose(SerialArm.ur5(), q_true)

    assert select_branch(tight, analytical_ik(tight, target), np.zeros(6)) is None


def test_no_solutions_at_all_is_none_not_an_exception(arm):
    assert select_branch(arm, np.empty((0, 6)), np.zeros(6)) is None


def test_ranking_puts_every_feasible_branch_before_every_rejected_one(arm):
    rng = np.random.default_rng(5)
    q_true = arm.random_config(rng, margin=0.7)
    target = reachable_pose(arm, q_true)

    ranked = rank_branches(arm, analytical_ik(arm, target), q_true)

    feasible = [i for i, b in enumerate(ranked) if b.feasible]
    rejected = [i for i, b in enumerate(ranked) if not b.feasible]
    assert not feasible or not rejected or max(feasible) < min(rejected)
    assert all(isinstance(b, Branch) for b in ranked)


def test_weighting_travel_harder_changes_which_branch_wins(arm):
    """Otherwise the weights are decoration and the cost is really one term."""
    rng = np.random.default_rng(6)
    q_true = arm.random_config(rng, margin=0.9)
    target = reachable_pose(arm, q_true)
    far = q_true + rng.uniform(-2.5, 2.5, 6)

    travel_first = select_branch(arm, analytical_ik(arm, target), far,
                                 travel_weight=50.0, singularity_weight=0.0,
                                 limit_weight=0.0)
    safety_first = select_branch(arm, analytical_ik(arm, target), far,
                                 travel_weight=0.0, singularity_weight=50.0,
                                 limit_weight=0.0)

    assert travel_first is not None and safety_first is not None
    assert travel_first.travel <= safety_first.travel
    assert safety_first.manipulability >= travel_first.manipulability


# ---- continuity across a path -----------------------------------------------

def worst_step(path):
    return max((float(np.max(np.abs(b - a))) for a, b in zip(path, path[1:])),
               default=0.0)


def smooth_path(arm, seed, waypoints=50):
    rng = np.random.default_rng(seed)
    start = arm.random_config(rng, margin=0.9)
    q = start.copy()
    poses = []
    for _ in range(waypoints):
        q = q + rng.uniform(-0.05, 0.05, 6)
        poses.append(arm.fk(q))
    return start, poses


def test_following_a_path_is_smooth_once_the_singular_floor_is_off(arm):
    """The property the whole module is for, over many paths rather than one.

    An earlier version of this test used a single seed and a short path, and
    passed while `apps/benchmark_ik.py` was measuring a 3.2 rad jump on a
    longer one. A seed is not a sample.

    With every singularity guard off, chained selection is pure continuity and
    there is nothing left to excuse a jump.
    """
    for seed in range(12):
        start, poses = smooth_path(arm, seed)

        path, failed = follow(arm, poses, start,
                              singularity_weight=0.0, limit_weight=0.0,
                              singular_floor=0.0)

        assert failed == [], f"seed {seed}: no branch at waypoints {failed}"
        assert worst_step(path) < 0.5, (
            f"seed {seed}: branch jump of {worst_step(path):.2f} rad")


def test_chaining_beats_taking_whichever_branch_comes_first(arm):
    """The comparison that justifies the module existing at all."""
    from armik.analytical import analytical_ik as solve

    chained_worse = 0
    for seed in range(12):
        start, poses = smooth_path(arm, seed)

        naive = [np.asarray(solve(arm, pose)[0], dtype=float)
                 for pose in poses if len(solve(arm, pose))]
        chained, _ = follow(arm, poses, start,
                            singularity_weight=0.0, limit_weight=0.0,
                            singular_floor=0.0)

        if worst_step(chained) > worst_step(naive):
            chained_worse += 1

    assert chained_worse == 0, (
        f"chained selection was less continuous than naive on {chained_worse} "
        f"of 12 paths")


def test_the_remaining_jumps_are_the_singular_floor_and_nothing_else(arm):
    """Why the default weights still produce occasional large steps.

    Not drift between branches: a refusal to track through a region where the
    arm loses a degree of freedom. Turning the floor off removes them, which is
    what makes that explanation a measurement rather than a story.
    """
    jumps_with_floor = jumps_without = 0

    for seed in range(12):
        start, poses = smooth_path(arm, seed)

        guarded, _ = follow(arm, poses, start)
        unguarded, _ = follow(arm, poses, start, singularity_weight=0.0,
                              limit_weight=0.0, singular_floor=0.0)

        jumps_with_floor += worst_step(guarded) > 1.0
        jumps_without += worst_step(unguarded) > 1.0

    assert jumps_without == 0, "a jump survived with every guard disabled"
    assert jumps_with_floor >= jumps_without


def test_every_pose_on_the_followed_path_is_actually_reached(arm):
    rng = np.random.default_rng(8)
    start = arm.random_config(rng, margin=0.9)
    q = start.copy()
    poses = []
    for _ in range(15):
        q = q + rng.uniform(-0.05, 0.05, 6)
        poses.append(arm.fk(q))

    path, failed = follow(arm, poses, start)

    assert failed == []
    for pose, chosen in zip(poses, path):
        assert np.allclose(arm.fk(chosen), pose, atol=1e-6)

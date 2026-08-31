"""Choose one inverse-kinematics solution, instead of returning all eight.

`analytical_ik` returns every closed-form solution for a pose, which is the
right answer to a mathematical question and the wrong answer to give a
controller. A robot executes one configuration. Something has to choose, and
the choice is where knowing the closed form turns into knowing why a controller
needs it.

THREE THINGS DECIDE IT, and they conflict, which is why this is a weighted cost
rather than a rule:

  joint travel        how far the arm must move from where it already is.
                      Cheapest to execute, and the only term most naive
                      selectors use.
  singularity margin  how close the configuration is to losing a degree of
                      freedom. Yoshikawa manipulability; near zero, small
                      Cartesian corrections need enormous joint rates, so a
                      pose that is fine to hold is dangerous to move through.
  limit margin        how much room is left before a joint runs out of travel.
                      A configuration sitting against a stop cannot be
                      corrected in one direction at all.

Picking purely on travel walks the arm into singularities, because the nearest
solution to a configuration near a singularity is usually another one. Picking
purely on manipulability throws the arm across its workspace between adjacent
waypoints. The weights are stated, not hidden, so they can be argued with.

THE 2-PI PROBLEM, and it is the part most implementations get wrong.
`analytical_ik` returns principal values in (-pi, pi]. A UR joint travels plus
or minus 2 pi, so for a joint currently at +3.0 rad, a solution reported as
-3.0 rad is the SAME arm pose reachable by moving 0.28 rad forward or 6.0 rad
backward. Selecting on the principal value alone therefore reports a wrist
unwind as the cheap option and hands the controller a six radian move to reach
a pose it was almost already in. Every candidate here is first shifted to the
2-pi equivalent nearest the current configuration, and it is the shifted
configuration that gets returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armik.ik import manipulability
from armik.robot import SerialArm

TAU = 2.0 * np.pi

#: Default weights, in radians of equivalent joint travel.
#:
#: THE FIRST VERSION OF THIS COST WAS WRONG, and the benchmark caught it, which
#: is worth recording because the mistake is the obvious one. The margin terms
#: were reciprocals, `weight / manipulability`, so at the median manipulability
#: of 1.6e-2 the singularity term contributed 0.6 / 0.016 = 37 against a travel
#: term of order 0.05 for an adjacent waypoint. Travel was arithmetically
#: irrelevant, the selector chose whichever branch was furthest from a
#: singularity at every step, and the resulting path was LESS continuous than
#: naively taking the first solution the closed form returned: worst joint step
#: 3.16 rad against 0.07 rad. A cost whose terms are not commensurate is not a
#: weighting, it is one term with decoration.
#:
#: Both margins are now bounded penalties in [0, 1]: zero while the quantity is
#: healthy, rising to one as it reaches the danger threshold. So travel decides
#: between comparable branches, and the margins only intervene when something
#: is actually close to going wrong, which is what the weighting was always
#: meant to express.
#:
#: The right values depend on the machine, so they are arguments rather than
#: constants and `select_branch` takes them.
TRAVEL_WEIGHT = 1.0
SINGULARITY_WEIGHT = 3.0
LIMIT_WEIGHT = 1.5

#: At or above this manipulability a configuration carries no singularity
#: penalty at all. It is the measured median over random UR5 configurations
#: (see SINGULAR_FLOOR), so a typical pose is simply "fine" rather than being
#: scored against an unreachable ideal.
SAFE_MANIPULABILITY = 1.6e-2

#: Radians of joint travel beyond which a limit is not worth worrying about.
SAFE_MARGIN = 0.5

#: Below this manipulability a configuration is treated as singular and
#: rejected outright rather than merely penalised. A cost term cannot express
#: "not at any price", and a controller asked to move through a singularity
#: does not produce a large error, it produces an unbounded joint rate.
#:
#: The value is measured rather than guessed. Over 4000 random UR5
#: configurations drawn inside the joint limits:
#:
#:     p50   1.64e-02        p5    2.16e-04
#:     p25   3.99e-03        p1    3.73e-05
#:
#: and the home pose, which is genuinely singular, is exactly 0. So 1e-4 sits
#: near the 2.5th percentile and about 160 times below the median: low enough
#: that a normal configuration never trips it, high enough to catch a pose the
#: arm should not be asked to move through. About 2.5% of random poses fall
#: below it, which is the rate at which `select_branch` correctly returns None.
SINGULAR_FLOOR = 1e-4


@dataclass(frozen=True)
class Branch:
    """One candidate configuration, and why it scored what it did."""

    q: np.ndarray
    #: Sum of absolute joint motion from the previous configuration, radians.
    travel: float
    #: Yoshikawa manipulability. Larger is further from a singularity.
    manipulability: float
    #: Radians of room at the tightest joint before it hits a limit.
    limit_margin: float
    cost: float
    feasible: bool
    #: Why it was rejected, or "" when it was not.
    rejected: str = ""


def nearest_equivalent(q: np.ndarray, reference: np.ndarray,
                       limits: np.ndarray) -> np.ndarray:
    """Shift each joint by a multiple of 2 pi to sit nearest `reference`.

    The same arm pose, expressed as the configuration the controller can reach
    most cheaply from where it is. A shift is only taken when the result stays
    inside the joint's travel; a wrist that would have to pass a limit to reach
    the equivalent keeps its principal value.
    """
    q = np.asarray(q, dtype=float).copy()
    reference = np.asarray(reference, dtype=float)

    for i in range(len(q)):
        turns = np.round((reference[i] - q[i]) / TAU)
        if turns == 0:
            continue
        shifted = q[i] + turns * TAU
        if limits[i, 0] <= shifted <= limits[i, 1]:
            q[i] = shifted
    return q


def _limit_margin(q: np.ndarray, limits: np.ndarray) -> float:
    """Radians of room at the tightest joint. Negative means outside."""
    return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))


def score_branch(arm: SerialArm, q: np.ndarray, previous: np.ndarray, *,
                 travel_weight: float = TRAVEL_WEIGHT,
                 singularity_weight: float = SINGULARITY_WEIGHT,
                 limit_weight: float = LIMIT_WEIGHT,
                 singular_floor: float = SINGULAR_FLOOR) -> Branch:
    """Cost one candidate, after shifting it to its nearest 2-pi equivalent."""
    shifted = nearest_equivalent(q, previous, arm.joint_limits)

    travel = float(np.sum(np.abs(shifted - previous)))
    margin = _limit_margin(shifted, arm.joint_limits)
    mu = manipulability(arm, shifted)

    if margin < 0:
        return Branch(shifted, travel, mu, margin, float("inf"), False,
                      "outside joint limits")
    if mu < singular_floor:
        return Branch(shifted, travel, mu, margin, float("inf"), False,
                      "singular")

    # Bounded penalties rather than reciprocals: zero while the quantity is
    # healthy, one at the danger threshold. See the note on the weights.
    singular_penalty = max(0.0, (SAFE_MANIPULABILITY - mu) / SAFE_MANIPULABILITY)
    limit_penalty = max(0.0, (SAFE_MARGIN - margin) / SAFE_MARGIN)

    cost = (travel_weight * travel
            + singularity_weight * singular_penalty
            + limit_weight * limit_penalty)
    return Branch(shifted, travel, mu, margin, float(cost), True)


def rank_branches(arm: SerialArm, solutions: np.ndarray, previous: np.ndarray,
                  **weights: float) -> list[Branch]:
    """Every candidate, scored and sorted cheapest first, rejects last."""
    scored = [score_branch(arm, q, previous, **weights) for q in solutions]
    return sorted(scored, key=lambda b: (not b.feasible, b.cost))


def select_branch(arm: SerialArm, solutions: np.ndarray, previous: np.ndarray,
                  **weights: float) -> Branch | None:
    """The configuration a controller should be given, or None if there is none.

    None rather than the least-bad infeasible answer. Handing a controller a
    configuration outside its joint limits, or one at a singularity, is worse
    than reporting that the pose cannot be reached from here: the first fails
    in the machine and the second fails in the planner, where it can be handled.
    """
    if len(solutions) == 0:
        return None
    best = rank_branches(arm, solutions, previous, **weights)[0]
    return best if best.feasible else None


def follow(arm: SerialArm, poses: list[np.ndarray], start: np.ndarray,
           **weights: float) -> tuple[list[np.ndarray], list[int]]:
    """Track a Cartesian path, choosing a branch at every waypoint.

    MEASURED, over 40 random smooth paths of 60 waypoints each, as the worst
    single joint step on each path (`apps/benchmark_ik.py`):

        first branch returned          worst 6.28 rad   median 0.139   11/40 over 1 rad
        chained, default weights       worst 3.21 rad   median 0.050    6/40 over 1 rad
        chained, singularity guards off worst 0.11 rad  median 0.050    0/40 over 1 rad

    Two things follow, and the second is the one worth knowing. Chaining alone
    cuts the median step by a factor of three and halves the number of paths
    with a discontinuity. And **every remaining jump is the singular floor
    doing its job**: turning it off removes all six, so the selector is not
    drifting between branches, it is refusing to track through a region where
    the arm loses a degree of freedom, and paying a large joint move to get out.

    Which of those two behaviours is correct is a property of the machine, not
    of this function, so both are reachable: pass `singular_floor=0.0` for pure
    continuity, and leave it alone to keep the refusal.

    Returns the chosen configurations and the indices of any waypoint where no
    feasible branch existed. Selection is chained: each waypoint is chosen
    relative to the one actually taken before it, which is what produces branch
    continuity rather than a sequence of independently reasonable jumps.
    """
    from armik.analytical import analytical_ik

    path: list[np.ndarray] = []
    failed: list[int] = []
    previous = np.asarray(start, dtype=float)

    for index, pose in enumerate(poses):
        chosen = select_branch(arm, analytical_ik(arm, pose), previous, **weights)
        if chosen is None:
            failed.append(index)
            continue
        path.append(chosen.q)
        previous = chosen.q
    return path, failed

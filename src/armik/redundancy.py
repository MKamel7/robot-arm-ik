"""Redundancy resolution: doing something useful with the joints you have spare.

WHAT REDUNDANCY IS, and why it needs a seventh joint to be interesting. A pose
is six numbers. A 6R arm has, generically, a finite set of configurations
reaching it, and `armik.select` already picks between them. A 7R arm has a
CONTINUUM: at every non-singular configuration there is a one-dimensional family
of joint velocities that move the joints and leave the end effector exactly
where it is. That family is the null space of the Jacobian, and it is free
motion in the only sense that matters here, it costs nothing in task space.

THE WHOLE IDEA IN ONE LINE:

    qdot = J^+ v  +  (I - J^+ J) z

The first term does the task. The second lives entirely in the null space, so it
CANNOT disturb the task however large `z` is, PROVIDED the projector is built
from the EXACT pseudo-inverse and not the damped one. Getting that wrong is the
mistake this module made first and its own tests caught. That term is where a
secondary objective goes: stay away from joint limits, stay away from singularities, avoid
an obstacle. On a 6R arm at full rank the projector is the zero matrix and the
second term vanishes, which is not a bug and is the honest answer: there is no
freedom to spend. A test asserts exactly that on the UR5e, because a redundancy
module that appeared to do something on a non-redundant arm would be computing
nonsense somewhere.

WHY THE PSEUDO-INVERSE IS DAMPED. Near a singularity J^+ has enormous entries
and the task term demands joint velocities no real arm can produce. Damping
trades a little task accuracy for a bounded command, which is the same
Levenberg-Marquardt argument `armik.ik` already makes, and it must be the same
damping or the two disagree about what the arm does at the same configuration.

THE TWO SECONDARY OBJECTIVES HERE CAN DISAGREE, and that is a real property
rather than an implementation limit. Pushing every joint toward the middle of
its range is not generally the same as maximising manipulability, and on the
Panda they pull in different directions in parts of the workspace. Combining
them is a weighted sum, the weights are the caller's, and `compare()` exists so
the trade can be measured rather than asserted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from armik.robot import SerialArm

#: Same damping as `armik.ik` uses, and it has to be: two modules disagreeing
#: about the damped inverse would make the arm behave differently depending on
#: which one issued the command near a singularity.
DEFAULT_DAMPING = 1e-2

#: Step used for the numerical manipulability gradient. Yoshikawa's measure has
#: no convenient closed-form derivative for a general DH arm, so this is a
#: central difference. Too small and it is dominated by floating-point noise in
#: the determinant; too large and it stops describing the local slope. 1e-5 rad
#: sits in the flat part of that trade for this arm and is a parameter so a
#: reader can watch it stop working.
GRADIENT_STEP = 1e-5


def damped_pseudo_inverse(J: np.ndarray,
                          damping: float = DEFAULT_DAMPING) -> np.ndarray:
    """J^T (J J^T + k^2 I)^-1, the damped right inverse.

    The right inverse rather than the left because a redundant arm has more
    columns than rows, so J J^T is the small square one.
    """
    rows = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(rows))


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    """I - J^+ J, built from the TRUE pseudo-inverse rather than the damped one.

    THE TWO INVERSES HERE ARE DELIBERATELY DIFFERENT, and the first version of
    this module got it wrong by using the damped one for both. The tests caught
    it: on a full-rank 6R arm the projector should be exactly zero and was not,
    and null-space motion on the Panda moved the tool.

    The reason is that damping is a deliberate approximation, and the two terms
    want opposite things from it:

      task term    damped, because near a singularity the exact inverse demands
                   joint velocities no arm can produce. Accepting a little task
                   error to keep the command bounded is the point.
      projector    exact, because the whole claim of null-space control is that
                   the secondary motion costs the task NOTHING. A projector
                   built from an approximate inverse leaks secondary motion
                   straight into task-space error, which is the one thing it
                   exists to prevent.

    `np.linalg.pinv` truncates small singular values, so this stays well
    behaved at a singularity: the arm has lost a task direction, and motion in
    that direction is correctly reported as free because nothing is asking for
    it any more.
    """
    n = J.shape[1]
    return np.eye(n) - np.linalg.pinv(J) @ J


def redundancy_dimension(arm: SerialArm, q: np.ndarray) -> int:
    """How many dimensions of free motion exist here.

    Zero on a 6R arm at full rank, one on a 7R arm at full rank, and MORE at a
    singularity, where the arm has lost a task direction rather than gained
    freedom. The number is the same either way, which is why it is reported
    beside the manipulability rather than on its own.
    """
    return arm.n - int(np.linalg.matrix_rank(arm.jacobian(q)))


def joint_limit_margin(arm: SerialArm, q: np.ndarray) -> float:
    """Distance to the nearest joint limit, normalised by each joint's range.

    Normalised because the Panda's joint 2 has a range of 3.5 rad and its joint
    4 about 3.0, and an unnormalised margin would call the same fractional
    distance from a limit "worse" on whichever joint happens to have the wider
    span. 0.5 means every joint sits exactly at its midpoint; 0 means one is on
    a hard stop.
    """
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    span = high - low
    return float(np.min(np.minimum(q - low, high - q) / span))


def joint_limit_gradient(arm: SerialArm, q: np.ndarray) -> np.ndarray:
    """Gradient of the classical distance-from-limits criterion.

    H(q) = -(1/2n) * sum(((q_i - mid_i) / span_i)^2), so ascending H walks every
    joint toward the middle of its own range. The gradient is analytic, which
    matters because this one is evaluated at every step of a trajectory and a
    finite difference here would double the Jacobian evaluations.

    Normalising by span again, for the same reason as above: without it the
    criterion pulls hardest on whichever joint has the largest absolute offset
    rather than the one closest to actually stopping.
    """
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    mid = 0.5 * (low + high)
    span = high - low
    return -(q - mid) / (arm.n * span**2)


def manipulability(arm: SerialArm, q: np.ndarray) -> float:
    """Yoshikawa's sqrt(det(J J^T)). Zero at a singularity.

    Re-exported from the same definition `armik.ik` uses rather than written
    again, because two manipulability measures in one library is how a
    singularity-avoidance controller ends up avoiding a different singularity
    from the one the metric reports.
    """
    from armik.ik import manipulability as _manipulability
    return float(_manipulability(arm, q))


def manipulability_gradient(arm: SerialArm, q: np.ndarray,
                            step: float = GRADIENT_STEP) -> np.ndarray:
    """Central-difference gradient of the manipulability measure.

    Central rather than forward: forward differences carry a first-order error
    that biases the direction, and near a singularity the measure is changing
    fast enough for that bias to point the controller the wrong way.
    """
    grad = np.zeros(arm.n)
    for i in range(arm.n):
        forward, backward = q.copy(), q.copy()
        forward[i] += step
        backward[i] -= step
        grad[i] = (manipulability(arm, forward)
                   - manipulability(arm, backward)) / (2.0 * step)
    return grad


#: A secondary objective: given an arm and a configuration, which way to move.
Objective = Callable[[SerialArm, np.ndarray], np.ndarray]


def combined(*objectives: tuple[Objective, float]) -> Objective:
    """Weighted sum of secondary objectives.

    Provided because the two here genuinely disagree in parts of the workspace,
    and a caller should be made to state the trade rather than discovering that
    one silently dominated.
    """
    def evaluate(arm: SerialArm, q: np.ndarray) -> np.ndarray:
        total = np.zeros(arm.n)
        for objective, weight in objectives:
            total = total + weight * objective(arm, q)
        return total
    return evaluate


def resolve(arm: SerialArm, q: np.ndarray, twist: np.ndarray, *,
            objective: Objective | None = None, gain: float = 1.0,
            damping: float = DEFAULT_DAMPING) -> np.ndarray:
    """Joint velocity achieving `twist`, with any spare freedom spent on `objective`.

    With `objective=None` this is the plain damped least-squares solution and
    the null space is left unused, which is the baseline `compare()` measures
    against.
    """
    J = arm.jacobian(q)
    qdot = damped_pseudo_inverse(J, damping) @ twist
    if objective is None:
        return qdot
    return qdot + nullspace_projector(J) @ (gain * objective(arm, q))


@dataclass(frozen=True)
class Trace:
    """What happened while following a path.

    Both the worst and the mean are kept for each measure. A controller that
    improves the average while driving one joint onto a stop has not helped,
    and only the worst value shows that.
    """

    configurations: np.ndarray
    margins: np.ndarray
    manipulabilities: np.ndarray
    task_errors: np.ndarray

    @property
    def worst_margin(self) -> float:
        return float(np.min(self.margins))

    @property
    def mean_manipulability(self) -> float:
        return float(np.mean(self.manipulabilities))

    @property
    def worst_task_error(self) -> float:
        """Largest deviation from the commanded twist, in the twist's own units.

        Reported so that any claimed improvement can be checked against what it
        cost the task. A secondary objective that quietly degrades the primary
        one is not a free improvement, and the null-space projector is only
        exactly task-preserving in the undamped limit.
        """
        return float(np.max(self.task_errors))


def follow(arm: SerialArm, q_start: np.ndarray,
           twists: Sequence[np.ndarray], *, dt: float = 1.0,
           objective: Objective | None = None, gain: float = 1.0,
           damping: float = DEFAULT_DAMPING) -> Trace:
    """Integrate a twist sequence, recording what the arm did along the way.

    Joint values are clipped to their limits. A controller whose whole purpose
    is limit avoidance must not be measured on a trajectory that quietly walked
    through a limit and came back.
    """
    q = np.asarray(q_start, dtype=float).copy()
    low, high = arm.joint_limits[:, 0], arm.joint_limits[:, 1]

    configurations, margins, manips, errors = [], [], [], []
    for twist in twists:
        qdot = resolve(arm, q, np.asarray(twist, dtype=float),
                       objective=objective, gain=gain, damping=damping)
        achieved = arm.jacobian(q) @ qdot
        errors.append(float(np.linalg.norm(achieved - np.asarray(twist))))
        q = np.clip(q + qdot * dt, low, high)
        configurations.append(q.copy())
        margins.append(joint_limit_margin(arm, q))
        manips.append(manipulability(arm, q))

    return Trace(configurations=np.array(configurations),
                 margins=np.array(margins),
                 manipulabilities=np.array(manips),
                 task_errors=np.array(errors))


def compare(arm: SerialArm, q_start: np.ndarray,
            twists: Sequence[np.ndarray], *, objective: Objective,
            gain: float = 1.0, dt: float = 1.0) -> tuple[Trace, Trace]:
    """The same path with and without the secondary objective.

    Returned as a pair rather than as a verdict. Whether a gain in margin is
    worth a loss in manipulability is a decision about a robot and a task, not
    something this function is entitled to make.
    """
    baseline = follow(arm, q_start, twists, dt=dt, objective=None)
    controlled = follow(arm, q_start, twists, dt=dt,
                        objective=objective, gain=gain)
    return baseline, controlled

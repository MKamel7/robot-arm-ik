"""Compare paths from different planners without flattering any of them.

Four planner families produce paths with wildly different shapes: an
interpolation returns two waypoints, a sampling planner returns a jagged
sequence of random steps, an optimiser returns a fixed-length smooth one. Any
metric computed straight off those waypoint lists measures the OUTPUT FORMAT
rather than the motion, which is the trap this module exists to avoid.

So every metric here is computed on a path resampled to a fixed number of
points at equal spacing along its own joint-space arc length. After that a
two-waypoint straight line and a 60-waypoint optimiser output are described in
the same terms, and a planner cannot look smoother by returning fewer points or
rougher by returning more.

The metrics are deliberately implementation independent, so they say the same
thing about a path from this repository's own planners and a path from MoveIt:

    joint_path_length   how far the joints travel, in radians summed over the
                        arm. The cost the motors pay.
    task_path_length    how far the tool travels, in metres. The cost the
                        process pays, and not the same ranking as the above.
    smoothness          mean squared second difference of the resampled joint
                        path. Unitless and only comparable between paths
                        resampled the same way, which is why the resampling is
                        part of this module rather than left to the caller.

Clearance is NOT here. It needs the collision world, which is a property of the
scene rather than of the path, so it belongs to whatever owns the geometry.
"""

from __future__ import annotations

import numpy as np

#: Resampling density. High enough that a curved path is not straightened by
#: sampling, low enough that a 20-problem sweep over eight planners stays quick.
DEFAULT_SAMPLES = 100


def _as_array(path) -> np.ndarray:
    q = np.asarray(path, dtype=float)
    if q.ndim != 2 or len(q) < 2:
        raise ValueError("a path needs at least two waypoints of equal width")
    return q


def arc_lengths(path) -> np.ndarray:
    """Cumulative joint-space distance along a path, starting at 0."""
    q = _as_array(path)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(path, samples: int = DEFAULT_SAMPLES) -> np.ndarray:
    """Resample a joint path to `samples` points at equal arc length.

    A degenerate path (start equals goal, or every waypoint identical) has no
    arc length to divide, so it is returned as a repeat of its first point
    rather than raising: a planner that was asked to move nowhere did nothing
    wrong.
    """
    q = _as_array(path)
    s = arc_lengths(q)
    if s[-1] <= 0.0:
        return np.repeat(q[:1], samples, axis=0)
    target = np.linspace(0.0, s[-1], samples)
    return np.column_stack([np.interp(target, s, q[:, j]) for j in range(q.shape[1])])


def joint_path_length(path) -> float:
    """Total joint travel in radians, summed over joints, in path order."""
    return float(arc_lengths(path)[-1])


def task_path_length(positions) -> float:
    """Total tool travel in metres along a sequence of Cartesian positions."""
    p = _as_array(positions)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def smoothness(path, samples: int = DEFAULT_SAMPLES) -> float:
    """Mean squared second difference of the resampled joint path.

    Lower is smoother. Zero for a straight line in joint space, whatever its
    length, which is the property that makes it a shape measure rather than a
    disguised length measure: a long straight move and a short straight move
    both score 0, and a wiggle scores by how sharply it wiggles.

    Reported unitless. It is not jerk: the paths compared here are geometric,
    and several of them arrive with no timing at all, so a time derivative
    would have to invent a time parameterisation and would then measure that
    invention.
    """
    q = resample(path, samples)
    if len(q) < 3:
        return 0.0
    second = np.diff(q, n=2, axis=0)
    return float(np.mean(np.sum(second**2, axis=1)))


def tool_positions(arm, path) -> np.ndarray:
    """Tool-tip positions along a joint path, through a forward kinematics."""
    q = _as_array(path)
    return np.array([arm.fk(row)[:3, 3] for row in q])


def link_points(arm, q, per_link: int = 6) -> np.ndarray:
    """Points sampled along the arm's link centre lines at configuration q.

    The joint origins alone are too sparse: a long link can pass straight
    through an obstacle with both of its ends well clear of it.
    """
    frames = arm.frames(np.asarray(q, dtype=float))
    origins = np.array([T[:3, 3] for T in frames])
    points = [origins[0]]
    for a, b in zip(origins[:-1], origins[1:]):
        for s in np.linspace(0.0, 1.0, per_link)[1:]:
            points.append(a + s * (b - a))
    return np.array(points)


def distance_to_box(points, centre, half_extents) -> float:
    """Smallest distance from a set of points to an axis-aligned box.

    Zero inside the box. Standard point-to-box distance: clamp the point to the
    box, measure what is left over.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    delta = np.abs(p - np.asarray(centre, dtype=float)) - np.asarray(half_extents, dtype=float)
    outside = np.maximum(delta, 0.0)
    return float(np.min(np.linalg.norm(outside, axis=1)))


def clearance(arm, path, boxes, per_link: int = 6, samples: int = DEFAULT_SAMPLES) -> float:
    """Smallest distance between the arm's link centre lines and any box.

    HONEST LIMIT, and it is why this is not called "distance to collision":
    link thickness is ignored, so the true clearance of a real arm is smaller
    than this by roughly a link radius. It is an upper bound, identical in
    construction for every planner, so it ranks paths fairly even though it
    would be the wrong number to quote as a safety margin.

    FCL through MoveIt would give the exact figure and was tried first. Asked
    for a distance with self-collision pairs allowed, it returns a constant
    0.000996 m for every configuration, including in an empty world, so it was
    reporting a sentinel rather than a measurement. The validity column in the
    benchmark still comes from FCL, where it answers the question it was asked.

    `boxes` is a sequence of (centre, half_extents).
    """
    q_path = resample(path, samples)
    worst = float("inf")
    for q in q_path:
        points = link_points(arm, q, per_link)
        for centre, half in boxes:
            worst = min(worst, distance_to_box(points, centre, half))
    return worst

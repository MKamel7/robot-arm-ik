"""Joint-space RRT-Connect motion planning.

Sampling-based planning for the case the straight-line paths in trajectory.py
can't handle: a workspace obstacle sits between q_start and q_goal, so no
interpolation of the two ever stays collision-free and something has to
search around it. Collision checking is injected as a callback rather than
hard-coded, so this module (and its tests) depend on nothing but numpy --
whatever calls it (a MuJoCo scene, a simple geometric check, ...) owns the
notion of "collision".

RRT-Connect grows two trees, one rooted at q_start and one at q_goal.  Each
iteration, the active tree takes one EXTEND step toward a random sample (with
occasional goal-biased sampling to pull it toward the other tree's root), and
then the other tree tries to CONNECT -- i.e. take repeated greedy steps, not
just one -- straight toward the node the active tree just added.  Growing two
trees toward each other like this, rather than one tree toward the goal, is
what makes RRT-Connect converge fast through the narrow passages a 6-DOF arm's
joint space tends to have. Every step, in both EXTEND and CONNECT, is
collision-checked along its full interpolated segment (not just the
endpoint), so a thin obstacle can't be tunnelled through between samples.

The raw path RRT-Connect returns is a jagged sequence of random steps that
happens to reach the goal -- not something you'd want to run on a real robot.
A shortcutting pass follows: repeatedly pick two non-adjacent waypoints and,
if the direct segment between them is collision-free, splice out everything
in between. This is what turns the random walk into a short, usable path.
"""

from dataclasses import dataclass, field

import numpy as np

_TRAPPED, _ADVANCED, _REACHED = range(3)


@dataclass
class _Tree:
    """Joint configs plus parent pointers, rooted at nodes[0] (parent -1)."""

    nodes: list = field(default_factory=list)
    parents: list = field(default_factory=list)

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        """Index of the tree node closest to q.

        Brute-force distance scan: the trees this planner grows (thousands of
        nodes at most, low-dimensional joint space) are far too small to
        justify a KD-tree, and this keeps the module dependency-free.
        """
        dists = [np.linalg.norm(n - q) for n in self.nodes]
        return int(np.argmin(dists))

    def path_to_root(self, i: int) -> list:
        """Nodes from the root to node i, inclusive, root-to-leaf order."""
        path = []
        while i != -1:
            path.append(self.nodes[i])
            i = self.parents[i]
        return path[::-1]


def _segment_free(q_from: np.ndarray, q_to: np.ndarray, collision_fn, resolution: float) -> bool:
    """True if every point sampled along q_from -> q_to at `resolution` spacing
    is collision-free (checks q_to; q_from is assumed already checked)."""
    dist = float(np.linalg.norm(q_to - q_from))
    n_checks = max(1, int(np.ceil(dist / resolution)))
    for i in range(1, n_checks + 1):
        q = q_from + (q_to - q_from) * (i / n_checks)
        if collision_fn(q):
            return False
    return True


def _steer(q_from: np.ndarray, q_target: np.ndarray, step_size: float):
    """One step from q_from toward q_target, capped at step_size.

    Returns (q_new, reached) where reached is True iff q_target itself was
    within step_size (so q_new == q_target exactly, not just close to it).
    """
    diff = q_target - q_from
    dist = float(np.linalg.norm(diff))
    if dist <= step_size:
        return q_target.copy(), True
    return q_from + diff * (step_size / dist), False


def _extend(tree: _Tree, q_target: np.ndarray, joint_limits: np.ndarray,
            collision_fn, step_size: float, resolution: float):
    """Grow `tree` one step toward q_target. Returns (status, new_index)."""
    i_near = tree.nearest(q_target)
    q_near = tree.nodes[i_near]
    q_new, reached_target = _steer(q_near, q_target, step_size)
    q_new = np.clip(q_new, joint_limits[:, 0], joint_limits[:, 1])
    if np.allclose(q_new, q_near):
        # Clipping collapsed the step back onto the node it started from
        # (q_near already sits on the limit boundary in the direction of
        # travel) -- treat as blocked rather than adding a duplicate node.
        return _TRAPPED, -1
    if not _segment_free(q_near, q_new, collision_fn, resolution):
        return _TRAPPED, -1
    i_new = tree.add(q_new, i_near)
    reached = reached_target and np.array_equal(q_new, q_target)
    return (_REACHED if reached else _ADVANCED), i_new


def _connect(tree: _Tree, q_target: np.ndarray, joint_limits: np.ndarray,
             collision_fn, step_size: float, resolution: float, max_steps: int):
    """Repeated EXTEND toward q_target until it's reached or blocked.

    This greedy multi-step "connect" (rather than a single "extend") is the
    move that gives RRT-Connect its name and its speed: once one tree gets
    close to the other, it lunges the rest of the way instead of taking one
    random-length step per outer iteration.
    """
    status, i_new = _ADVANCED, -1
    steps = 0
    while status == _ADVANCED and steps < max_steps:
        status, i_new = _extend(tree, q_target, joint_limits, collision_fn, step_size, resolution)
        steps += 1
    return status, i_new


def _shortcut(path: list, collision_fn, resolution: float, rng: np.random.Generator,
              iters: int = 200) -> list:
    """Path-shortcutting: repeatedly try to splice out a detour by connecting
    two non-adjacent waypoints directly, if the straight segment between them
    is collision-free. Raw RRT-Connect output is a jittery random walk; this
    is what makes it short and purposeful enough to run on a real arm.
    """
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.integers(0, len(path), size=2))
        if j - i < 2:
            continue
        if _segment_free(path[i], path[j], collision_fn, resolution):
            path = path[:i + 1] + path[j:]
    return path


def rrt_connect(q_start: np.ndarray, q_goal: np.ndarray, collision_fn, joint_limits: np.ndarray, *,
                 step_size: float = 0.1, max_iters: int = 5000, goal_bias: float = 0.05,
                 seed=None):
    """Plan a collision-free joint-space path from q_start to q_goal.

    collision_fn(q) -> bool, True means q is IN COLLISION (blocked). It is
    the only thing this function knows about the world, so it works equally
    well against a MuJoCo scene, an analytic obstacle, or a unit test double.

    joint_limits is (n, 2) of [lo, hi] per joint; sampling and every extension
    are kept inside this box.

    Returns a list of joint configs (waypoints, start to goal inclusive),
    already shortcut-smoothed, or None if no path was found within max_iters.
    Also returns None immediately (without spending any iterations) if
    q_start or q_goal is itself in collision -- that's a query error, not a
    search that should be given a budget.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    joint_limits = np.asarray(joint_limits, dtype=float)

    if collision_fn(q_start) or collision_fn(q_goal):
        return None

    resolution = step_size / 2.0  # finer than step_size so segment checks can't tunnel
    diag = float(np.linalg.norm(joint_limits[:, 1] - joint_limits[:, 0]))
    max_connect_steps = max(1, int(np.ceil(diag / step_size)) + 1)

    rng = np.random.default_rng(seed)
    tree_start = _Tree([q_start.copy()], [-1])
    tree_goal = _Tree([q_goal.copy()], [-1])

    active, other = tree_start, tree_goal
    active_is_start = True

    for _ in range(max_iters):
        bias_target = q_goal if active_is_start else q_start
        if rng.random() < goal_bias:
            q_rand = bias_target
        else:
            q_rand = rng.uniform(joint_limits[:, 0], joint_limits[:, 1])

        status, i_new = _extend(active, q_rand, joint_limits, collision_fn, step_size, resolution)
        if status != _TRAPPED:
            q_new = active.nodes[i_new]
            status2, i_other = _connect(other, q_new, joint_limits, collision_fn,
                                        step_size, resolution, max_connect_steps)
            if status2 == _REACHED:
                path_active = active.path_to_root(i_new)
                # other's connect walk ended exactly on q_new, so its root-to-
                # tip path shares that point with path_active's tip -- drop
                # the duplicate before splicing the two halves together.
                path_other = other.path_to_root(i_other)[::-1][1:]
                path = (path_active + path_other if active_is_start
                        else path_other[::-1] + path_active[::-1])
                return _shortcut(path, collision_fn, resolution, rng)

        active, other = other, active
        active_is_start = not active_is_start

    return None

"""RRT-Connect: path existence, collision safety, joint limits, and the
early-exit / no-hang failure modes. Uses a synthetic hypersphere obstacle in
low-dimensional joint space (no arm/robot geometry needed) so this stays a
pure-NumPy test of the planner itself."""

import time

import numpy as np

from armik import rrt_connect


def _sphere_collision(center, radius):
    """collision_fn: True (blocked) inside the sphere of `radius` about `center`."""
    center = np.asarray(center, dtype=float)

    def collision_fn(q):
        return float(np.linalg.norm(np.asarray(q) - center)) < radius

    return collision_fn


def _path_length(path):
    return sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))


def _densely_collision_free(path, collision_fn, resolution=0.01):
    """Interpolate every consecutive pair of waypoints at fine resolution and
    confirm collision_fn never fires -- the actual safety property a
    real robot needs from the returned path, not just "endpoints are clear"."""
    for a, b in zip(path[:-1], path[1:]):
        dist = np.linalg.norm(b - a)
        n_checks = max(1, int(np.ceil(dist / resolution)))
        for i in range(n_checks + 1):
            q = a + (b - a) * (i / n_checks)
            if collision_fn(q):
                return False
    return True


def test_unobstructed_finds_near_direct_path():
    q_start = np.array([-0.7, -0.7, -0.7])
    q_goal = np.array([0.7, 0.7, 0.7])
    joint_limits = np.array([[-1.0, 1.0]] * 3)
    collision_fn = lambda q: False   # noqa: E731

    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits, seed=0)

    assert path is not None
    assert np.allclose(path[0], q_start)
    assert np.allclose(path[-1], q_goal)
    direct = np.linalg.norm(q_goal - q_start)
    # With no obstacle, the shortcut pass should collapse the raw random-walk
    # path down close to the straight line -- "reasonably close", not the
    # jagged, much-longer raw RRT-Connect output.
    assert _path_length(path) < direct * 1.5


def test_blocked_direct_path_routes_around_obstacle():
    q_start = np.array([-0.8, 0.0, 0.0])
    q_goal = np.array([0.8, 0.0, 0.0])
    joint_limits = np.array([[-1.0, 1.0]] * 3)
    # A sphere centred on the straight line between start and goal, so naive
    # linear interpolation between them collides.
    collision_fn = _sphere_collision(center=(0.0, 0.0, 0.0), radius=0.5)
    assert collision_fn(0.5 * (q_start + q_goal))   # sanity: direct path is blocked

    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits,
                        step_size=0.1, max_iters=5000, seed=1)

    assert path is not None
    assert np.allclose(path[0], q_start)
    assert np.allclose(path[-1], q_goal)
    # The real safety property: every densely-interpolated segment between
    # consecutive returned waypoints must stay collision-free end to end.
    assert _densely_collision_free(path, collision_fn)


def test_respects_joint_limits():
    q_start = np.array([-0.4, -0.4, -0.4])
    q_goal = np.array([0.4, 0.4, 0.4])
    joint_limits = np.array([[-0.5, 0.5]] * 3)   # tight box around start/goal
    collision_fn = lambda q: False   # noqa: E731

    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits,
                        step_size=0.05, max_iters=3000, seed=2)

    assert path is not None
    for q in path:
        assert np.all(q >= joint_limits[:, 0] - 1e-9)
        assert np.all(q <= joint_limits[:, 1] + 1e-9)


def test_start_in_collision_returns_none_promptly():
    joint_limits = np.array([[-1.0, 1.0]] * 3)
    collision_fn = _sphere_collision(center=(-0.8, 0.0, 0.0), radius=0.3)
    q_start = np.array([-0.8, 0.0, 0.0])   # inside the sphere
    q_goal = np.array([0.8, 0.0, 0.0])

    t0 = time.time()
    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits, max_iters=5000)
    elapsed = time.time() - t0

    assert path is None
    # Must early-exit, not spend the whole max_iters budget searching from a
    # blocked start.
    assert elapsed < 2.0


def test_goal_in_collision_returns_none_promptly():
    joint_limits = np.array([[-1.0, 1.0]] * 3)
    collision_fn = _sphere_collision(center=(0.8, 0.0, 0.0), radius=0.3)
    q_start = np.array([-0.8, 0.0, 0.0])
    q_goal = np.array([0.8, 0.0, 0.0])   # inside the sphere

    t0 = time.time()
    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits, max_iters=5000)
    elapsed = time.time() - t0

    assert path is None
    assert elapsed < 2.0


def test_unsolvable_returns_none_without_hanging():
    """A deliberately-hard problem (obstacle spans nearly the whole box between
    start and goal) with a tiny iteration budget: must return None, not hang
    or raise, and must do so quickly."""
    q_start = np.array([-0.9, 0.0, 0.0])
    q_goal = np.array([0.9, 0.0, 0.0])
    joint_limits = np.array([[-1.0, 1.0]] * 3)
    # Large sphere leaves only thin slivers near the box corners passable --
    # combined with a tiny max_iters, this should not be solved in time.
    collision_fn = _sphere_collision(center=(0.0, 0.0, 0.0), radius=0.85)
    assert not collision_fn(q_start) and not collision_fn(q_goal)

    t0 = time.time()
    path = rrt_connect(q_start, q_goal, collision_fn, joint_limits,
                        step_size=0.1, max_iters=5, seed=3)
    elapsed = time.time() - t0

    assert path is None
    assert elapsed < 2.0

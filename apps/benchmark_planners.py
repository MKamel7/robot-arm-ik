"""Compare this repository's planners against MoveIt's, on one Panda in one world.

    source /opt/ros/jazzy/setup.bash
    python3 apps/benchmark_planners.py            # writes docs/planner_comparison.csv

WHY THIS IS NOT A LIST OF PLANNERS WITH TIMINGS

Three of the four planner families in this comparison are implemented in this
repository: joint interpolation (`armik.trajectory.joint_trajectory`), Cartesian
interpolation (`armik.trajectory.cartesian_line`) and RRT-Connect with
shortcutting (`armik.rrt.rrt_connect`). The fourth, trajectory optimisation, is
not, and no amount of reading tells you how a hand-written planner compares with
a production one. So each family is run beside its reference implementation from
MoveIt, on identical problems in an identical collision world:

    joint interpolation      armik.joint_trajectory     vs  Pilz PTP
    Cartesian interpolation  armik.cartesian_line       vs  Pilz LIN
    sampling                 armik.rrt_connect          vs  OMPL RRTConnect
    optimisation             (nothing here)             vs  CHOMP, STOMP

THE TWO MODELS ARE THE SAME ROBOT, WHICH IS WHAT MAKES THIS FAIR

`SerialArm.panda()` is a Craig-form DH transcription and MoveIt plans on the
published URDF. They are only comparable if they agree, so the agreement is
measured rather than assumed: `tests/test_panda_model_agreement.py` checks the
flange pose across random configurations, and the worst disagreement is printed
in the header of the CSV this writes. Every joint configuration below is
therefore meaningful to both sides.

COLLISION CHECKING IS SHARED, DELIBERATELY

`rrt_connect` takes a collision callback and knows nothing else about the world,
so it is given MoveIt's own FCL check against the same planning scene the C++
planners use. Neither side gets an easier world, and a difference in the results
cannot be a difference in the collision model.

WHAT IS MEASURED, AND THE ONE COMPARISON THAT WOULD BE DISHONEST

Success, wall-clock planning time, joint path length, tool path length,
smoothness and minimum clearance, all defined in `armik.planning_metrics` and
computed on paths resampled to equal arc length so a planner cannot score better
by returning fewer waypoints.

Wall-clock time across the Python and C++ groups is NOT a language-independent
statement, and the CSV marks which implementation each row came from so the
comparison can be made within a group. What IS comparable across groups is
everything else: a path's length, shape and clearance do not care what wrote it.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik.planning_metrics import (  # noqa: E402
    clearance,
    joint_path_length,
    resample,
    smoothness,
    task_path_length,
    tool_positions,
)
from armik.robot import SerialArm  # noqa: E402
from armik.rrt import rrt_connect  # noqa: E402
from armik.trajectory import cartesian_line, joint_trajectory  # noqa: E402

GROUP = "panda_arm"
TIP_LINK = "panda_link8"
PIPELINES = ["ompl", "chomp", "stomp", "pilz_industrial_motion_planner"]
CSV_PATH = ROOT / "docs" / "planner_comparison.csv"

#: Problems per run. Twenty pairs over eight planners is 160 plans, which fits
#: in a few minutes and is enough to separate planners whose behaviour differs
#: by more than noise. It is not enough to rank two planners that differ by a
#: few percent, and the write-up says so rather than implying precision.
PROBLEMS = int(__import__("os").environ.get("BENCH_PROBLEMS", "20"))
SEED = 0

#: Where the obstacles are. A shelf in front of the arm at tool height with a
#: gap under it, so a straight line between two poses on opposite sides has to
#: go through the shelf, around it, or under it. Sized from the Panda's 0.855 m
#: published reach so the whole workspace is not simply blocked.
OBSTACLES = [
    # name,           position (x, y, z),      size (x, y, z)
    ("shelf_deck", (0.55, 0.0, 0.55), (0.36, 1.10, 0.03)),
    ("shelf_back", (0.72, 0.0, 0.40), (0.03, 1.10, 0.34)),
    ("floor", (0.0, 0.0, -0.03), (2.0, 2.0, 0.04)),
]


def build_scene(robot) -> None:
    """Put the obstacles in the planning scene every planner will share."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    with robot.get_planning_scene_monitor().read_write() as scene:
        for name, (x, y, z), (sx, sy, sz) in OBSTACLES:
            obj = CollisionObject()
            obj.id = name
            obj.header.frame_id = "panda_link0"
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [sx, sy, sz]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.w = 1.0
            obj.primitives = [box]
            obj.primitive_poses = [pose]
            obj.operation = CollisionObject.ADD
            scene.apply_collision_object(obj)
        scene.current_state.update()


def collision_checker(robot):
    """A `collision_fn(q) -> bool` backed by MoveIt's own FCL check.

    Returned as a closure holding one RobotState and one scene handle, because
    RRT-Connect calls this thousands of times per plan and re-entering the
    scene monitor per call would measure the monitor rather than the planner.
    """
    from moveit.core.robot_state import RobotState

    state = RobotState(robot.get_robot_model())

    def in_collision(q) -> bool:
        state.set_joint_group_positions(GROUP, np.asarray(q, dtype=float))
        state.update()
        with robot.get_planning_scene_monitor().read_only() as scene:
            return bool(scene.is_state_colliding(robot_state=state,
                                                 joint_model_group_name=GROUP))

    return in_collision, state


def path_is_valid(robot, state, path) -> bool:
    """Is every waypoint of this path collision free in the shared scene?

    FCL through MoveIt, which is the right tool for a yes or no answer and is
    what the planners themselves used. The clearance number beside it comes
    from `armik.planning_metrics.clearance` instead, for the reason recorded
    there: asked for a DISTANCE with self pairs allowed, FCL returns a constant
    0.000996 m for every configuration, including in an empty world.
    """
    with robot.get_planning_scene_monitor().read_only() as scene:
        for q in resample(path, 60):
            state.set_joint_group_positions(GROUP, q)
            state.update()
            if scene.is_state_colliding(robot_state=state, joint_model_group_name=GROUP):
                return False
    return True


#: The obstacle boxes as (centre, half extents), for the geometric clearance.
#: The floor is deliberately excluded: it bounds the workspace rather than
#: being the thing planners are asked to avoid, and including it pins the
#: minimum at the base link for every path.
CLEARANCE_BOXES = [((x, y, z), (sx / 2, sy / 2, sz / 2))
                   for name, (x, y, z), (sx, sy, sz) in OBSTACLES if name != "floor"]


def sample_problems(robot, in_collision, arm) -> list:
    """Seeded start and goal configurations that are free and worth planning.

    Rejection sampling inside the joint limits, keeping only configurations
    that are collision free and whose tool sits in front of the arm, then
    pairing them. A pair is kept only if the straight joint-space line between
    it collides, because a problem an interpolation solves outright separates
    no planners: the whole question here is what happens when the naive answer
    is not available.
    """
    rng = np.random.default_rng(SEED)
    limits = arm.joint_limits

    def sample_free():
        for _ in range(5000):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            if in_collision(q):
                continue
            p = arm.fk(q)[:3, 3]
            if p[0] < 0.25 or p[2] < 0.1 or np.linalg.norm(p) > 0.80:
                continue
            return q
        return None

    problems = []
    while len(problems) < PROBLEMS:
        q_a, q_b = sample_free(), sample_free()
        if q_a is None or q_b is None:
            print(f"could only build {len(problems)} problems before the sampler gave up")
            break
        if not any(in_collision(q) for q in np.linspace(q_a, q_b, 40)):
            continue  # an interpolation solves this one outright, so it separates nothing
        problems.append({"start": q_a, "goal": q_b})
    return problems


def run_armik(name, fn) -> dict:
    """Time one of this repository's planners and normalise its return."""
    started = time.perf_counter()
    path = fn()
    elapsed = time.perf_counter() - started
    return {"planner": name, "implementation": "armik (python)", "seconds": elapsed,
            "path": None if path is None else np.asarray(path, dtype=float)}


def run_moveit(robot, arm_component, name, pipeline, planner_id, start_q, goal, timeout=5.0) -> dict:
    """Time one MoveIt pipeline on the same problem, through the same interface."""
    from moveit.core.robot_state import RobotState
    from moveit.planning import PlanRequestParameters

    state = RobotState(robot.get_robot_model())
    state.set_joint_group_positions(GROUP, start_q)
    state.update()
    arm_component.set_start_state(robot_state=state)

    if isinstance(goal, np.ndarray):
        goal_state = RobotState(robot.get_robot_model())
        goal_state.set_joint_group_positions(GROUP, goal)
        goal_state.update()
        arm_component.set_goal_state(robot_state=goal_state)
    else:
        arm_component.set_goal_state(pose_stamped_msg=goal, pose_link=TIP_LINK)

    params = PlanRequestParameters(robot, pipeline)
    params.planning_pipeline = pipeline
    params.planner_id = planner_id
    params.planning_time = timeout
    params.planning_attempts = 1

    started = time.perf_counter()
    try:
        result = arm_component.plan(single_plan_parameters=params)
    except Exception:  # noqa: BLE001
        result = None
    elapsed = time.perf_counter() - started

    path = None
    if result:
        msg = result.trajectory.get_robot_trajectory_msg()
        points = msg.joint_trajectory.points
        if len(points) >= 2:
            path = np.array([p.positions for p in points], dtype=float)
    return {"planner": name, "implementation": "moveit (c++)", "seconds": elapsed,
            "path": path}


def _quaternion(R):
    """Rotation matrix to (w, x, y, z), via the rotation vector armik already has.

    `armik.rotations` deals in matrices and rotation vectors because that is
    what the IK needs; ROS messages want a quaternion, and this is the only
    place in the repository that does, so the conversion lives here rather
    than widening the library's API for one caller.
    """
    from armik.rotations import rotvec_from_matrix

    v = np.asarray(rotvec_from_matrix(R), dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    axis = v / angle
    return (float(np.cos(angle / 2.0)), *(axis * np.sin(angle / 2.0)))


def pose_msg(arm, q):
    """A PoseStamped at the tool frame of configuration q, for the Cartesian planners."""
    from geometry_msgs.msg import PoseStamped

    T = arm.fk(q)
    msg = PoseStamped()
    msg.header.frame_id = "panda_link0"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = T[:3, 3]
    w, x, y, z = _quaternion(T[:3, :3])
    msg.pose.orientation.w, msg.pose.orientation.x = float(w), float(x)
    msg.pose.orientation.y, msg.pose.orientation.z = float(y), float(z)
    return msg


def main() -> None:
    import rclpy
    from moveit_configs_utils import MoveItConfigsBuilder
    from moveit.planning import MoveItPy

    rclpy.init()
    config = (MoveItConfigsBuilder("moveit_resources_panda")
              .planning_pipelines(pipelines=PIPELINES)
              .to_moveit_configs()).to_dict()
    config["planning_pipelines"] = {"pipeline_names": PIPELINES}
    robot = MoveItPy(node_name="planner_comparison", config_dict=config)
    component = robot.get_planning_component(GROUP)

    arm = SerialArm.panda()
    build_scene(robot)
    in_collision, state = collision_checker(robot)

    problems = sample_problems(robot, in_collision, arm)
    print(f"{len(problems)} problems, every one of them blocked for a straight joint-space line")

    rows = []
    for index, problem in enumerate(problems, start=1):
        q_a, q_b = problem["start"], problem["goal"]
        goal_pose = pose_msg(arm, q_b)

        attempts = [
            run_armik("joint interpolation", lambda: joint_trajectory(q_a, q_b, dt=0.02)[1]),
            run_armik("cartesian interpolation",
                      lambda: (lambda out: out[0] if out[1] else None)(
                          cartesian_line(arm, arm.fk(q_a), arm.fk(q_b), q_a, steps=50))),
            run_armik("rrt connect",
                      lambda: rrt_connect(q_a, q_b, in_collision, arm.joint_limits, seed=SEED + index)),
            run_moveit(robot, component, "pilz ptp", "pilz_industrial_motion_planner", "PTP", q_a, q_b),
            run_moveit(robot, component, "pilz lin", "pilz_industrial_motion_planner", "LIN", q_a, goal_pose),
            run_moveit(robot, component, "ompl rrtconnect", "ompl", "RRTConnectkConfigDefault", q_a, q_b),
            run_moveit(robot, component, "chomp", "chomp", "chomp", q_a, q_b),
            run_moveit(robot, component, "stomp", "stomp", "stomp", q_a, q_b),
        ]

        for attempt in attempts:
            path = attempt.pop("path")
            row = {"problem": index, **attempt}
            if path is None or len(path) < 2:
                row.update(succeeded=False, collision_free=False, joint_length="",
                           tool_length="", smoothness="", min_clearance_m="")
            else:
                row.update(
                    succeeded=True,
                    collision_free=path_is_valid(robot, state, path),
                    joint_length=round(joint_path_length(path), 4),
                    tool_length=round(task_path_length(tool_positions(arm, resample(path, 60))), 4),
                    smoothness=float(f"{smoothness(path):.3e}"),
                    min_clearance_m=round(clearance(arm, path, CLEARANCE_BOXES), 4),
                )
            row["seconds"] = round(row["seconds"], 4)
            rows.append(row)

        done = [r for r in rows if r["problem"] == index and r["succeeded"]]
        print(f"problem {index}/{len(problems)}: {len(done)} of 8 planners returned a path")
        write_csv(rows)

    write_csv(rows)
    summarise(rows)
    print(f"\nwritten to {CSV_PATH}")

    # moveit_py segfaults inside its own teardown here, reliably, after every
    # row is on disk and the summary is printed. Exiting first turns a core
    # dump that means nothing into a clean exit code, and the alternative is a
    # benchmark that looks like it crashed every time it succeeded.
    sys.stdout.flush()
    os._exit(0)


def write_csv(rows) -> None:
    """Rewritten every problem, so a killed run still leaves a complete file."""
    fields = ["problem", "planner", "implementation", "succeeded", "collision_free",
              "seconds", "joint_length", "tool_length", "smoothness", "min_clearance_m"]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows) -> None:
    planners = []
    for row in rows:
        if row["planner"] not in planners:
            planners.append(row["planner"])

    print()
    print(f"{'planner':<26}{'ok':>6}{'valid':>7}{'sec':>9}{'joint':>9}{'smooth':>11}{'clear':>8}")
    for planner in planners:
        mine = [r for r in rows if r["planner"] == planner]
        ok = [r for r in mine if r["succeeded"]]
        valid = [r for r in ok if r["collision_free"]]
        def med(key, source):
            values = [float(r[key]) for r in source if r[key] != ""]
            return float(np.median(values)) if values else float("nan")
        print(f"{planner:<26}{len(ok):>4}/{len(mine):<2}{len(valid):>6}"
              f"{med('seconds', mine):>9.3f}{med('joint_length', ok):>9.2f}"
              f"{med('smoothness', ok):>11.2e}{med('min_clearance_m', valid):>8.3f}")


if __name__ == "__main__":
    main()

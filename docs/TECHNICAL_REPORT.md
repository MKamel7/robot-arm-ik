# Robot Arm IK: technical report

A from-scratch 6-DOF kinematics library grown into an industrial, perception-driven
palletizing cell. This report summarises the architecture, methods, results, and
honest limitations across the four development phases.

## 1. Overview

The project starts from the mathematics of manipulation, forward kinematics,
Jacobians, and inverse kinematics written in plain NumPy, and builds up to a
production-style robot cell on the framework industry deploys (ROS 2 + MoveIt 2),
with a camera closing the loop.

| Phase | Capability | Stack |
| --- | --- | --- |
| 0 | Repo maturity: tests, config, benchmark, CI | NumPy |
| 1 | Real planning & control: RRT-Connect, time-optimal trajectories, physics execution | NumPy + MuJoCo |
| 2 | Industrial palletizing cell (UR5e + Robotiq 2F-85) | ROS 2 Jazzy + MoveIt 2 |
| 3 | Perception-driven bin picking | + Gazebo RGB-D |

## 2. Kinematics core (Phase 0/1)

- **Forward kinematics** from standard Denavit-Hartenberg parameters; the UR5e FK
  is cross-validated against the MuJoCo Menagerie model in `tests/test_ur5e_mujoco.py`,
  agreeing to 1.5 mm worst case and 0.98 mm mean over 5000 random configurations,
  with orientation to 4e-6 degrees. The residual is the Menagerie XML's
  millimetre-rounded link lengths, not an error in either model.
- **Inverse kinematics** two ways: a damped-least-squares numerical solver
  (`dq = J^T (J J^T + lambda^2 I)^-1 e`) that stays stable through singularities,
  and a closed-form analytic solver returning all eight branches. Over 2000 random
  poses, every analytic branch reproduces the target through FK to ~1e-13, and the
  numerical solver is verified to land on one of them.
- **Motion planning**: a joint-space **RRT-Connect** (two-tree EXTEND/CONNECT +
  shortcutting) replaces a lift-over heuristic when the direct path is blocked,
  proven to actually engage via a monkeypatch-spy test on a harder scenario.
- **Execution**: PD + gravity-compensation control (`tau = kp*e + kd*edot +
  qfrc_bias`) tracks the planned trajectory under real MuJoCo dynamics; grasp is a
  weld equality constraint. Kinematic playback remains the default.

Suite: 52 tests. A randomised-layout benchmark reports 100% success over 50
layouts with ~0.093 mm placement residual (the IK solver's own error).

## 3. Industrial palletizing cell (Phase 2)

The same UR5e runs on ROS 2 Jazzy + MoveIt 2 as a pedestal-mounted palletizing
cell with a Robotiq 2F-85 gripper. The design follows production practice rather
than one general-purpose sampling planner (see `ros2/docs/ENGINEERING_PLAN.md`):

- **Deterministic descents** with the **Pilz Industrial Motion Planner** (`LIN`,
  straight-line Cartesian) for approach and retreat.
- **Obstacle-avoiding transfers** with **OMPL**, to a fixed top-down joint
  configuration, via a high waypoint over the separator wall.
- **Consistent top-down grasp**: every station's joint configuration comes from a
  single `/compute_ik` seed, so the tool always arrives straight down in the same
  posture, no wrist flips or behind-the-base swings.
- **Attach on contact**: the part attaches to the gripper at the true grasp offset
  and follows the tool (mock hardware has no physics grasp).
- Reachability pre-check, back-to-front filling, and printed production metrics.

An initial naive version (OMPL, position-only goals) ran at ~100 s/part and 2/4;
the industrial rewrite is **4/4, 0 re-plans, ~10 s/part**, an order of magnitude
faster and reliable. The arm is mounted on a pedestal so it works downward over
the cell, the standard industrial layout.

## 4. Perception-driven bin picking (Phase 3)

A downward RGB-D camera (Gazebo) views the supply bin. A detector segments each
part by colour, samples the depth at the blob centroid, deprojects the pixel into
the camera optical frame, and transforms to the world frame with the known camera
pose, classical, honest perception with no learned model. It also estimates each
part's orientation by PCA of the colour blob.

The palletizer subscribes to the detected poses, transforms them into the arm's
base frame, applies a reachability pre-check, and picks each part at its detected
position and grasp yaw. Detection accuracy is 1-20 mm against ground truth (part
cube is 40 mm). Demonstrated on scrambled part positions: the arm palletizes what
it sees and does not fake undetected parts.

Closed loop: **RGB-D camera -> colour+depth detection -> MoveIt planning ->
pick-and-place**, 4/4 with per-part grasp orientation.

## 5. Honest limitations

- Mock hardware in the ROS stack: the grasp is represented in the planning scene
  (attached collision object), not force-controlled contact; the physics-mode
  execution (Phase 1, MuJoCo) is a separate, opt-in path.
- Perception is classical colour+depth on saturated primaries; it does not handle
  texture, clutter, or occlusion, and cube yaw is only defined modulo 90 degrees.
- Cycle time (~10 s/part) is dominated by OMPL planning on a CPU, not by the
  (scaled-down) execution speed.

## 6. Reproducing

- Core + physics: `uv run --group sim python apps/palletizing_cell.py`

The ROS 2 commands below run against the `armik_moveit` workspace, which lives
in [moveit-ur5-pick-place](https://github.com/MKamel7/moveit-ur5-pick-place)
rather than in this repository. It used to be duplicated here under `ros2/`,
byte-identical, which is how a fix could land in one copy and not the other.

- ROS 2 cell: `ros2 launch armik_moveit ur5e_gripper_moveit.launch.py` then
  `ros2 run armik_moveit palletize`
- Perception: `ros2 launch armik_moveit perception_palletizing.launch.py` then
  `PICK_SOURCE=perception ros2 run armik_moveit palletize`

Verification records for each capability are under `ros2/docs/`.

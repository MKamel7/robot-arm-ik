# ROS 2 + MoveIt 2: industrial palletizing cell (Phase 2)

The ROS 2 side of this project: the same UR5e, now driven by the framework
industry actually deploys. Where `../apps` solves and animates the kinematics in
pure NumPy, this stands up a full **industrial palletizing cell** on **ROS 2
Jazzy + MoveIt 2**, a pedestal-mounted UR5e with a Robotiq 2F-85 gripper that
transfers parts from a supply bin, over a divider wall, onto a pallet.

Targets ROS 2 Jazzy on Ubuntu 24.04. Runs on mock hardware (no physical robot).

## The cell

A pedestal-mounted UR5e works downward over a table (the standard palletizing
layout), with:

- a **supply bin** of four colour-coded parts,
- a **pallet** filled back-to-front,
- a **separator wall** standing between them,
- a **Robotiq 2F-85** gripper doing real top-down grasps.

```
ros2 launch armik_moveit ur5e_gripper_moveit.launch.py   # RViz opens
ros2 run   armik_moveit palletize                        # run the cell
```

## How it moves the arm (the industrial approach)

This is built the way a production cell is, not with one general-purpose
sampling planner (see [`docs/ENGINEERING_PLAN.md`](docs/ENGINEERING_PLAN.md)):

- **Deterministic descents.** The straight-down approach and straight-up retreat
  are **Pilz Industrial Motion Planner `LIN`** moves (linear Cartesian), the same
  motion profile a real palletizer runs.
- **Obstacle-avoiding transfers.** **OMPL** plans the bin-to-pallet transfer
  around the separator wall, to a fixed joint configuration (a via-point over the
  wall keeps each plan short and reliable).
- **Consistent top-down grasp.** Every station's joint configuration comes from a
  single `/compute_ik` seed, so the tool always arrives pointing straight down in
  the same elbow-up posture, no wonky angles, wrist flips, or behind-the-base
  swings.
- **Attach on contact.** The part attaches to the gripper at the true grasp
  offset and rigidly follows the tool (mock hardware has no physics grasp).
- **Reachability pre-check** rejects out-of-workspace slots instead of faking
  them; the pallet fills **back-to-front** so no placement reaches over a placed
  part; **production metrics** (placed / rejected / re-plans / cycle time) print
  at the end.

## Results (mock hardware, headless-verified)

| Metric | Value |
| --- | --- |
| Parts placed | **4 / 4** |
| Re-plans | 0-2 |
| Cycle time | **~9-10 s/part** |
| Grasp | consistent top-down |

An earlier naive version (OMPL, position-only goals) ran at ~100 s/part and
2/4; the industrial rewrite is ~10x faster and reliable. The story of that
rewrite (deterministic planning, defined grasps, attach-on-contact) is the
engineering content, see the plan and the verification records in `docs/`.

## Package `armik_moveit`

```
ros2/src/armik_moveit/
  description/ur5e_robotiq.urdf.xacro   UR5e + 2F-85 + pedestal (one model for control + MoveIt)
  config/                               SRDF, kinematics, joint/cartesian limits, controllers, RViz
  launch/
    ur5e_gripper_moveit.launch.py       one-command bringup (control + MoveIt, RViz optional)
    ur5e_moveit.launch.py               bare-arm bringup (step 1)
    view_ur5e_robotiq.launch.py         description-only model viewer
  armik_moveit/
    palletizing.py        the palletizing cell            (ros2 run armik_moveit palletize)
    scene.py              cell geometry + planning scene   (populate_scene)
    scene_routing_check.py collision-aware routing test    (scene_routing_check)
    plan_execute_smoke.py headless Plan-and-Execute check
```

## Build

```bash
cd ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select armik_moveit
source install/setup.bash
```

Requires: `ros-jazzy-desktop`, `ros-jazzy-ur`, `ros-jazzy-moveit`,
`ros-jazzy-ur-moveit-config`, `ros-jazzy-robotiq-description`,
`ros-jazzy-robotiq-controllers`.

## Verification

Each capability has a recorded check under `docs/`:

- `phase2_step1_verification.txt` — UR5e MoveIt bringup, Plan and Execute
- `phase2_gripper_verification.txt` — 2F-85 open/close as a MoveIt end effector
- `phase2_planning_scene_verification.txt` — collision-aware routing around the wall
- `phase2_palletizing_verification.txt` — full 4/4 palletizing run + metrics

Two Jazzy specifics baked into the launch (easy to miss): the mock-hardware flag
is `use_mock_hardware`, and `ur_control` can leave the trajectory controller
inactive so the launch re-activates it.

## Factory integration

Beyond the palletizing cell, the same arm runs a **colour-sorting cell** with the
production-integration layers a factory needs:

- **Colour sorting to three outfeed lanes** (`color_sort`), one belt per colour,
  fanned out radially from the robot so each colour leaves the cell in its own
  direction. Random part placement, and a 3-button GUI (`sort_gui`) that
  validates orders (refuses double-orders and absent colours).
- **OPC UA fieldbus** (`opcua_server`, `opc.tcp://:4840/cell/`) so a PLC/SCADA
  commands sorts and reads live process values and the safety state.
- **Live production dashboard** (`dashboard`, http://localhost:8080): parts sorted,
  throughput, cycle time, per-colour counts and a safety banner, plus a `/control`
  room view with the live robot. These are process metrics, not a true OEE figure
  (no availability or quality factors).
- **Functional safety** (`safety_supervisor`) — latched e-stop, guard interlock,
  speed-and-separation monitoring (ISO/TS 15066), watchdog, reset. See
  [`docs/SAFETY.md`](docs/SAFETY.md).
- **Real-hardware ready** — `use_mock_hardware:=false robot_ip:=<ip>` switches to
  a physical UR5e + Robotiq 2F-85. See [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Status and roadmap

- Done: bringup, 2F-85 integration, planning scene, industrial palletizing cell,
  pedestal-mounted layout, perception-driven bin picking, colour sorting, OPC UA
  + dashboard, functional safety, hardware-ready description.
- Advanced next: MoveIt Task Constructor for task-level grasp planning;
  continuous conveyor tracking (pick-on-the-fly) on real hardware.

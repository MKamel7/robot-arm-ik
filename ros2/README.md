# Phase 2: ROS 2 + MoveIt 2 (UR5e)

This directory is the ROS 2 side of the project: a colcon workspace that brings up the UR5e under [MoveIt 2](https://moveit.picknik.ai/) so motions are planned by MoveIt/OMPL and executed on `ros2_control`, instead of the standalone MuJoCo playback used by the pure-Python demos in `../apps`. The kinematics library (`armik`) stays untouched; this is the industry-standard framework the same arm runs on.

It targets **ROS 2 Jazzy on Ubuntu 24.04**.

## Package: `armik_moveit`

Phase 2 lands in stages. Step 1 (this commit) is the bringup: a single launch file that stands up the UR5e with mock hardware and MoveIt, and a headless smoke test that plans and executes a motion to prove the toolchain end to end.

```
ros2/
  src/armik_moveit/
    launch/ur5e_moveit.launch.py      one-command UR5e + MoveIt bringup (mock hardware)
    armik_moveit/plan_execute_smoke.py  headless Plan-and-Execute verification
```

## Build

```bash
cd ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select armik_moveit
source install/setup.bash
```

## Run

Bring up the arm and MoveIt (RViz on by default; drag the interactive marker and hit **Plan & Execute**):

```bash
ros2 launch armik_moveit ur5e_moveit.launch.py
```

Headless (no display), then verify planning + execution from the command line:

```bash
ros2 launch armik_moveit ur5e_moveit.launch.py launch_rviz:=false
ros2 run armik_moveit plan_execute_smoke     # prints RESULT: PASS on success
```

The smoke test reads the current joint state, asks MoveGroup (OMPL) to plan to a nearby reachable target, executes it on the mock hardware, and checks the arm reached the target (MoveItErrorCode SUCCESS and joint error under 0.05 rad).

## Two Jazzy-specific fixes baked into the launch

Getting a bare UR5e MoveIt bringup working on Jazzy needed two corrections that are easy to miss:

1. **Mock hardware flag renamed.** The argument is `use_mock_hardware` (it was `use_fake_hardware` on earlier distros). Without it the driver tries to reach a real robot over TCP and the controllers never come up.
2. **Trajectory controller left inactive.** `ur_control.launch.py` intermittently finishes with `scaled_joint_trajectory_controller` in the `inactive` state after its consistent-controller-set spawn, so MoveIt has nothing to execute on. The launch activates it (only if it is not already active) once bringup settles.

A third, cosmetic one: when `ur_moveit.launch.py` is *included* (rather than run directly) it must not be wrapped in a `TimerAction`, or its `declare_arguments()` scoping breaks and `warehouse_sqlite_path` stops resolving.

## Status

- **Step 1 (bringup + Plan & Execute):** done, verified headless (`plan_execute_smoke` returns PASS).
- Next: fixture/bin/pallet as a MoveIt planning scene, a palletizing action/node, and a recorded RViz demo.

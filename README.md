# Robot Arm IK

[![CI](https://github.com/MKamel7/robot-arm-ik/actions/workflows/ci.yml/badge.svg)](https://github.com/MKamel7/robot-arm-ik/actions/workflows/ci.yml)

Inverse kinematics and trajectory planning for a 6-DOF serial manipulator (the Universal Robots UR5/UR5e), written from the kinematics up in NumPy. Given a target pose for the tool, the planner finds the joint angles that reach it, plans a smooth timed motion to get there, and animates the whole thing in 3D.

![UR5e palletizing cell in MuJoCo](docs/palletizing_cell.gif)

*A UR5e palletizing cell, rendered in the [MuJoCo](https://mujoco.org) physics engine (`apps/palletizing_cell.py`): the robot transfers parts from a supply bin into a pallet grid with a Robotiq 2F-85 gripper, while a heads-up display reports the production metrics an automation engineer cares about (cycle time, throughput, placement accuracy). Every joint angle comes from this library: the UR5e forward kinematics is cross-validated against the MuJoCo model in the test suite, and agrees to 1.5 mm worst case, and the placement accuracy shown is the IK solver's own residual. A single pick-and-place (`apps/pick_and_place_mujoco.py`) and a dependency-light matplotlib version (`apps/pick_and_place.py`, below) also ship.*

![pick and place animation](docs/pick_and_place.gif)

## What it does

Four independent pieces, each a distinct capability:

1. **Forward kinematics** (`src/armik/robot.py`). The arm is defined by standard Denavit-Hartenberg parameters (real, manufacturer-published numbers). `SerialArm.fk(q)` composes the per-link homogeneous transforms to give the tool pose for any joint configuration. Both `SerialArm.ur5()` (the default) and `SerialArm.ur5e()` are provided; the UR5e FK is cross-validated against the MuJoCo Menagerie model with an identity joint mapping, in `tests/test_ur5e_mujoco.py`: position agrees to **1.5 mm worst case and 0.98 mm mean** over 5000 random configurations, orientation to 4e-6 degrees. The residual is not an error in either model, it is the Menagerie XML rounding the link lengths to the millimetre where the DH table uses the datasheet values to a tenth.

2. **Geometric Jacobian** (`SerialArm.jacobian(q)`). Maps joint velocities to the tool's spatial velocity, `[v; omega] = J(q) q_dot`. Its conditioning reveals singular configurations. The test suite verifies the analytic Jacobian against a finite-difference of forward kinematics, and confirms that the UR5's home pose is genuinely singular (rank drops below 6).

3. **Inverse kinematics** (`src/armik/ik.py`). The hard part. A numerical solver using **damped least squares**:

   ```
   dq = J^T (J J^T + lambda^2 I)^-1 e
   ```

   where `e` is the 6D pose error (position, plus orientation as a rotation vector). The damping term keeps the joint step bounded near singularities, where a plain pseudo-inverse would demand near-infinite joint speeds and blow up. Per-iteration step clamping keeps the linearisation honest, and each iterate is clamped to the joint limits. The solver reports whether it converged and the residual position and orientation error.

3b. **Closed-form inverse kinematics** (`src/armik/analytical.py`). `analytical_ik(arm, T)` returns *all* solutions in closed form: a generic reachable pose has eight (two shoulder, two elbow, two wrist branches), and the solver drops branches that are genuinely unreachable rather than faking them. This is both a capability (no seed, no iteration, every branch at once) and the strongest possible correctness check: over 2000 random poses, every analytic solution reproduces the target pose through forward kinematics to ~1e-13, and the numerical solver is verified to land on one of these branches.

4. **Trajectory planning** (`src/armik/trajectory.py`). A synchronised trapezoidal profile: all joints move together along a straight line in joint space, driven by a single time-scaling `s(t)` whose velocity and acceleration limits are chosen so no joint exceeds its bounds. The motion starts and ends at rest with a clean trapezoidal velocity profile. A Cartesian straight-line planner is also included (linear position, SLERP orientation, IK at each step).

The pick-and-place demo (`apps/pick_and_place.py`) ties it together: reach to a pick location, grasp, carry, release, and return home, with the gripper state shown on the tool tip.

## Architecture

```mermaid
flowchart LR
    subgraph CORE["armik core (pure NumPy)"]
        direction TB
        FK["Forward kinematics<br/>robot.py"]
        JAC["Geometric Jacobian<br/>robot.py"]
        IK["IK: damped least squares<br/>+ closed-form analytic<br/>ik.py / analytical.py"]
        TRAJ["Trajectory planning<br/>trajectory.py"]
        FK --> IK
        JAC --> IK
        IK --> TRAJ
    end

    PLANNER["Planner<br/>apps/palletizing_cell.py<br/>collision-aware routing (MuJoCo contact queries)<br/>+ reachability validation"]
    SCENE["MuJoCo scene<br/>UR5e + Robotiq 2F-85<br/>(scene config)"]
    OUT["GIF / MP4<br/>docs/*.gif"]

    CORE --> PLANNER --> SCENE --> OUT
```

armik never touches MuJoCo directly: it solves poses and timing in plain NumPy, and the Planner is the only place that queries MuJoCo (contact checks for re-routing, forward kinematics for reachability) before handing the resulting joint path to the scene for rendering. *Scene parameters (table height, grid layout, home pose, and the like) are being consolidated out of inline constants into a single scene config, so the same Planner/scene pipeline can be re-targeted without editing code.*

## Photoreal demos (MuJoCo)

Two optional demos render the kinematics in the [MuJoCo](https://mujoco.org) physics engine with a real UR5e and a Robotiq 2F-85 gripper. In both, armik does all the kinematics (`SerialArm.ur5e()` forward kinematics + damped-least-squares IK solve each waypoint, `joint_trajectory` builds the timed motion) and MuJoCo only renders the result and animates the gripper.

- **`apps/palletizing_cell.py`** (hero animation above) — an industrial palletizing cell that goes past a happy-path animation. It does **collision-aware routing** (a machine fixture stands between the bin and the pallet; the planner checks the direct path with MuJoCo contact queries and re-routes up-and-over when it is blocked), **multi-layer palletizing** (parts stack into a 2x2x2 pallet), and **failure handling** (one requested slot is outside the arm's reach; a reachability check rejects it on screen and the cell carries on). A heads-up display reports parts placed, cycle time, throughput, re-plans, rejections, and placement accuracy (the IK solver's own residual). This mirrors real end-of-line automation and intralogistics work.
- **`apps/pick_and_place_mujoco.py`** — a single pick-and-place, the same task as the matplotlib demo.

The UR5e and gripper models come from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), vendored under `assets/` with their licenses (see `assets/ATTRIBUTION.md`). These are optional extras, so the core library stays pure-NumPy.

## ROS 2 + MoveIt 2: industrial palletizing cell

The same UR5e also runs on the framework industry actually deploys: a full **industrial palletizing cell** on **ROS 2 Jazzy + MoveIt 2**, a pedestal-mounted UR5e with a Robotiq 2F-85 gripper that picks colour-coded parts from a supply bin, routes over a divider wall, and stacks them onto a pallet.

It is built the way a production cell is: **Pilz Industrial Motion Planner** `LIN` moves for the straight-down approach and retreat, **OMPL** for the obstacle-avoiding transfer, a **consistent top-down grasp** solved from one IK seed (no wonky wrist flips), attach-on-contact so parts follow the tool, a reachability pre-check, back-to-front filling, and printed production metrics. It places **4/4 at ~10 s/part**. A camera then closes the loop, **perception-driven bin picking** (RGB-D colour+depth detection feeds real grasp poses).

**The runnable workspace lives in [moveit-ur5-pick-place](https://github.com/MKamel7/moveit-ur5-pick-place), not here.** It used to be checked in under `ros2/` as well, byte-identical to the copy in that repository, which meant two copies of the same nodes drifting apart and a fix landing in one of them only. This repository keeps the engineering record and hands the code to the one place it is maintained:

- [`ros2/docs/ENGINEERING_PLAN.md`](ros2/docs/ENGINEERING_PLAN.md) is the write-up, and [`ros2/docs/`](ros2/docs/) holds the phase verification records, [`SAFETY.md`](ros2/docs/SAFETY.md), [`HARDWARE.md`](ros2/docs/HARDWARE.md) and [`DEMO_VIDEO.md`](ros2/docs/DEMO_VIDEO.md).
- The [technical report](docs/TECHNICAL_REPORT.md) covers all four phases.
- To run the cell, follow the README in `moveit-ur5-pick-place`.

## Run it

```bash
uv run --group dev pytest                        # 111 tests with the sim extras, 93 without
uv run --group dev python apps/pick_and_place.py --save   # matplotlib animation -> docs/pick_and_place.gif

uv run --group sim python apps/palletizing_cell.py --save         # the palletizing cell GIF
uv run --group sim python apps/pick_and_place_mujoco.py --save     # the single pick-and-place GIF
```

(or the classic path: `pip install numpy matplotlib pytest`, then `python apps/pick_and_place.py`; add `pip install mujoco imageio pillow` for the MuJoCo demos.)

## Why damped least squares

The clean way to invert the Jacobian is the Moore-Penrose pseudo-inverse, `dq = J^+ e`. It works right up until the arm approaches a singularity, where the Jacobian loses rank, `J J^T` becomes ill-conditioned, and the commanded joint velocities explode. Damped least squares trades a small amount of tracking accuracy for stability: the `lambda^2 I` term bounds the step no matter how singular the configuration. The UR5's fully-extended home pose is a real singularity, so the solver is tested seeded from exactly there and required to stay finite and make progress, not diverge.

## Layout

```
src/armik/
  robot.py        DH model, forward kinematics, geometric Jacobian
  rotations.py    rotation-vector <-> matrix, SLERP (robust at 0 and pi)
  ik.py           damped-least-squares IK, manipulability measure
  analytical.py   closed-form UR5 IK (all 8 branches)
  trajectory.py   synchronised trapezoidal + Cartesian straight-line
apps/
  pick_and_place.py         matplotlib 3D animated demo (NumPy only)
  pick_and_place_mujoco.py  photoreal MuJoCo pick-and-place (UR5e + Robotiq 2F-85)
  palletizing_cell.py       industrial palletizing cell with a production-metrics HUD
assets/                     vendored MuJoCo Menagerie models (UR5e, 2F-85) + licenses
tests/
  test_kinematics.py     FK validity, Jacobian vs finite-difference, singularity
  test_ik.py             round-trip accuracy, stability near singularity
  test_analytical_ik.py  every branch reaches the pose, 8-branch count, numeric agrees
  test_trajectory.py     boundary conditions, velocity limits, synchronisation
  test_ur5e.py           UR5e FK golden (vs MuJoCo), IK round-trip
```

## Choosing one solution, and measuring whether it helps

`analytical_ik` returns every closed-form solution for a pose, which is the right answer to a mathematical question and the wrong thing to hand a controller: a robot executes one configuration. `armik.select` scores the candidates on joint travel, singularity margin and joint-limit margin, and returns one.

**The 2-pi problem is the part most implementations get wrong.** The closed form returns principal values in (-pi, pi]. A UR joint travels ±2pi, so for a joint at +3.0 rad a solution reported as -3.0 rad is the *same arm pose* reachable by moving 0.28 rad. Selecting on the principal value hands the controller a six-radian wrist unwind to reach a pose it was almost already in. Every candidate is shifted to the 2-pi equivalent nearest the current configuration, and the shifted configuration is what gets returned.

![branch continuity](docs/ik_branch_continuity.png)

Worst single joint step on each of 40 random 60-waypoint Cartesian paths (`apps/benchmark_ik.py`):

| selector | worst step | median | paths with a jump over 1 rad |
|---|---|---|---|
| first branch returned | 6.28 rad | 0.139 | 11 of 40 |
| chained, default weights | 3.21 rad | 0.050 | 6 of 40 |
| chained, singularity guards off | **0.11 rad** | 0.050 | **0 of 40** |

Chaining alone cuts the median step threefold and halves the discontinuous paths. **Every remaining jump is the singular floor doing its job**: disabling it removes all six, so the selector is not drifting between branches, it is refusing to track through a region where the arm loses a degree of freedom and paying a large joint move to leave. Which behaviour is correct depends on the machine, so both are reachable: `singular_floor=0.0` for pure continuity, the default to keep the refusal.

**The first version of this cost was wrong, and the benchmark is what caught it.** The margin terms were reciprocals, so at the median manipulability of 1.6e-2 the singularity term contributed 0.6/0.016 = 37 against a travel term of order 0.05 for an adjacent waypoint. Travel was arithmetically irrelevant and the resulting path was *less* continuous than the naive selector, at 3.16 rad against 0.07. Both margins are now bounded penalties in [0, 1]. A cost whose terms are not commensurate is not a weighting, it is one term with decoration.

## Solver benchmarks

Every figure is generated from `docs/ik_benchmark.csv`, so a number in the report and a point on a plot cannot disagree.

![iterations against conditioning](docs/ik_iterations_vs_condition.png)
![success rate by manipulability](docs/ik_success_vs_manipulability.png)

Over 600 random poses, seeded 0.6 rad away from the answer: **95% converge**, median 7 iterations and p95 92, p95 position error **0.099 mm**. The analytic solver returns all branches in a median 0.7 ms against 1.6 ms for damped least squares, and the interesting part is the tail rather than the median: DLS p95 is 44 ms, because a badly conditioned pose costs an order of magnitude more than a typical one. **Failure is not spread evenly**: it lives almost entirely in the lowest manipulability band, which is the argument for reporting the distribution rather than one success rate.

## Redundancy: a seventh joint, and what to do with it

A pose is six numbers. A 6R arm reaches it in a finite set of configurations,
and the section above picks between them. A **7R arm has a continuum**: at every
non-singular configuration there is a one-dimensional family of joint velocities
that move the joints and leave the tool exactly where it is. `armik.redundancy`
spends that freedom on a secondary objective:

    qdot = J+ v  +  (I - J+ J) z

`SerialArm.panda()` is the arm, in **modified (Craig) DH** because that is the
convention Franka publishes. The transcription is checked against two things
outside this repository: the flange at the standard ready pose lands within a
millimetre of the published [0.307, 0, 0.590], and the sampled reach from the
shoulder is 0.858 m against a published 0.855 m.

### What it buys, measured

From `apps/benchmark_redundancy.py`, over a 60-step Cartesian drag that walks the
arm toward its limits. Full sweep in `docs/redundancy_benchmark.csv`.

| | worst joint-limit margin | mean manipulability |
|---|---|---|
| no null-space control | 0.088 | 0.0561 |
| limit avoidance, gain 5 | **0.192** | 0.0320 |
| manipulability, gain 0.2 | 0.063 | **0.0571** |

**The two objectives genuinely conflict.** Doubling the limit margin costs 43% of
the manipulability, and buying 1.7% more manipulability costs 29% of the margin.
Neither is free, `compare()` returns both traces rather than a verdict, and which
trade is right is a decision about a robot and a task.

### Both objectives are worse than nothing past their optimal gain

![limit avoidance gain sweep](docs/redundancy_limit_gain.png)

The controller integrates in discrete steps, so a large null-space step
overshoots the hill it is climbing. Limit avoidance peaks at gain 5 and decays
above it; **at gain 300 the controller whose entire purpose is staying off the
stops puts a joint on one.** Tests assert both of those, because the finding is
more useful than the tuned number.

Task error stays below 4.3e-4 across every gain, which is the check that the
null-space term really is free.

### A bug worth recording

The first version built the projector from the **damped** inverse, the same one
the task term uses. Its own tests caught it: on a full-rank 6R arm the projector
should be exactly zero and was not, and null-space motion on the Panda moved the
tool. Damping is a deliberate approximation that belongs in the task term, where
trading a little accuracy for a bounded command near a singularity is the point.
Putting it in the projector leaks secondary motion straight into task error,
which is the one thing the projector exists to prevent.

## Planners: the ones here against the ones people ship

Three of the four planner families below are implemented in this repository.
The comparison is not "which planner is best", it is whether a hand-written
implementation does the same thing as the production one, measured on identical
problems in an identical collision world.

`apps/benchmark_planners.py` runs it against a sourced ROS 2 Jazzy with MoveIt.
It is a script and not a package: the duplicated colcon workspace was removed
from this repository deliberately, and nothing here brings one back. 20 seeded
start and goal pairs on the
Panda, each one chosen so a straight joint-space line between them is blocked
by a shelf. Full sweep in `docs/planner_comparison.csv`, drawn by
`apps/plot_planner_comparison.py`.

**The two models are the same robot, and that is checked rather than assumed.**
`SerialArm.panda()` is a Craig-form DH transcription and MoveIt plans on the
published URDF. `tests/test_panda_model_agreement.py` compares the flange pose
over 200 random configurations inside the joint limits: worst disagreement
below 1e-6 m and 1e-6 in every rotation entry.

| planner | written by | solved | collision free | median s | joint travel | smoothness | clearance |
|---|---|---|---|---|---|---|---|
| joint interpolation | this repo | 20/20 | **0** | 0.000 | 4.96 | straight | n/a |
| Pilz PTP | MoveIt | 20/20 | **0** | 0.000 | 4.96 | straight | n/a |
| Cartesian interpolation | this repo | 0/20 | 0 | 1.71 | n/a | n/a | n/a |
| Pilz LIN | MoveIt | 1/20 | 0 | 0.001 | 2.44 | 5.8e-07 | n/a |
| RRT-Connect | this repo | **20/20** | **20** | 0.055 | 6.26 | 2.2e-05 | 0.086 |
| OMPL RRTConnect | MoveIt | 16/20 | 13 | 0.019 | 6.68 | 2.3e-05 | 0.111 |
| CHOMP | MoveIt | 14/20 | 14 | 0.408 | 5.22 | 3.0e-06 | 0.096 |
| STOMP | MoveIt | 13/20 | 13 | 0.258 | 5.01 | 3.2e-07 | 0.102 |

Medians over the problems each planner solved. Joint travel is radians summed
over the arm, clearance is metres to the shelf, smoothness is the mean squared
second difference of the path after resampling to equal arc length, so a
planner cannot score better by returning fewer waypoints.

**Time is not comparable across the two groups and the table should not be read
that way.** This repository's planners are NumPy and MoveIt's are C++. Path
length, shape and clearance do not care what wrote them; seconds do.

### What the numbers say

**The two joint interpolations are the same function.** Not similar: the path
lengths agree to four decimals on every one of the 20 problems (6.1724, 4.5935,
8.2930, and so on). Both also produce a colliding path on all 20, which is
correct behaviour for a primitive that does not take a collision world as an
argument, and is the reason planners exist.

**A straight line for the tool is usually not available.** Cartesian
interpolation solved none of the 20 and Pilz LIN solved one, for different
reasons: this repository's version fails when IK stops converging along the
line, and LIN refuses when the joint acceleration limit would be violated.
Neither is a defect. A straight tool path between two arbitrary reachable
configurations leaves the reachable set almost immediately.

**Optimisers buy smoothness, not clearance.** STOMP's paths are about 70 times
smoother than either sampling planner and CHOMP's about 7 times, while the
clearance of all four is within 3 cm of each other. They pay 5 to 20 times the
compute time for it, and they solve fewer problems: an optimiser initialised
with a straight line that is deep in collision has a poor starting point.

### The finding worth the run: OMPL returned paths with collisions in them

**Six of 52 OMPL paths across three repeats contain a collision**, and none of
them collide at OMPL's own waypoints. Checked at 400 samples along the path, 2
to 4 samples are inside the shelf; checked at the waypoints, zero are. The
collisions are strictly between waypoints.

That is the textbook tunnelling case rather than a defect in OMPL. The shelf
here is 30 mm thick, and `moveit_resources_panda_moveit_config` does not set
`longest_valid_segment_fraction`, so the segment check runs at the MoveIt
default. This repository's `rrt.py` checks every segment at `step_size / 2`,
which is 0.05 rad, and its docstring gives that as the reason: "so a thin
obstacle can't be tunnelled through between samples". The measurement says the
docstring is right, and says nothing at all about the quality of OMPL, which
would find these collisions too if it were asked at a finer resolution.

**So the 20/20 in the table is not a claim that this repository's RRT-Connect
beats OMPL's.** It solves more of these problems and validates all of them, at
three times the compute cost per solve and with a finer collision check. Those
are the trade-offs, not a ranking.

![planner comparison](docs/planner_comparison.png)

## Roadmap

- **Set `longest_valid_segment_fraction` and re-run the comparison.** The planner table above found six of 52 OMPL paths carrying a collision strictly between waypoints, against 30 mm obstacles, at the MoveIt default. The fix is one parameter and the measurement to check it against already exists, which makes this the cheapest real experiment left here.
- **Give the optimisers a better initial guess.** CHOMP and STOMP solved 14 and 13 of 20 while both sampling planners solved more, and an optimiser handed a straight line that is deep in collision is being asked to start from the worst possible place. Seeding them from an RRT-Connect path would separate "the optimiser is weak" from "the initialisation was".

Not doing: **no second ROS workspace here.** The duplicated one was removed and `moveit-ur5-pick-place` owns that story. This repository answers one question, whether the manipulator mathematics is understood.

## License

MIT

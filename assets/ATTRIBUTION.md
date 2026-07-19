# Vendored assets

The MuJoCo pick-and-place demo (`apps/pick_and_place_mujoco.py`) uses two robot
models from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
vendored here so the demo runs without a separate download.

## `ur5e/` — Universal Robots UR5e
- Source: `mujoco_menagerie/universal_robots_ur5e`
- License: see `ur5e/LICENSE` (BSD-3-Clause).

## `robotiq_2f85/` — Robotiq 2F-85 gripper
- Source: `mujoco_menagerie/robotiq_2f85`
- License: see `robotiq_2f85/LICENSE`.

Each model's original `LICENSE` file is kept alongside its assets. These models
are unmodified Menagerie releases; the demo composes them (attaches the gripper
to the arm flange) and adds a table, a box, and a place pad at runtime via the
MuJoCo `mjSpec` API — no source meshes are altered.

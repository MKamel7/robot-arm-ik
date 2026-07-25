"""Headless benchmark for the palletizing cell planner.

Runs `build_plan` over N randomized cell layouts (bin part count/spacing,
machine-fixture position) and reports aggregate planning stats: success rate,
cycle time, re-plan count, and IK placement error. No MuJoCo rendering (never
calls `render()`/saves a GIF/MP4), but `build_plan` itself still does
per-waypoint MuJoCo collision queries for the lift-traverse-lower re-routing,
so it is not free: ~1-1.4 s per run, ~60 s for the N=50 default. Useful for
regression-checking the planner (e.g. after touching the collision-routing
logic in `Planner.move`) without eyeballing a GIF.

Each run keeps `pallet_grid` and `unreachable_xy` at their `CellConfig`
defaults, so every run's request list has the same shape: 4 base-layer slots
+ 1 deliberately-unreachable slot + 4 top-layer slots (see
`CellConfig.requests`). That means the bin must supply at least as many parts
as there are reachable pallet slots (8), or `build_plan` will index past the
end of `cfg.bin` -- the randomized bin grid is clamped to respect that.

Run:
    python apps/benchmark_palletizing.py --n 50 --seed 0
"""

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import SerialArm, solve_ik   # noqa: E402

from palletizing_cell import (           # noqa: E402
    DEFAULT_CONFIG, Planner, build_cell, build_plan, tool_down_pose,
)

# number of reachable pallet slots requested per run, given pallet_grid and
# unreachable_xy stay at their CellConfig defaults (see CellConfig.requests)
N_REACHABLE_REQUESTS = 2 * len(DEFAULT_CONFIG.pallet_xy)


def gripper_length(mujoco, model, arm, cfg):
    """Same computation as palletizing_cell.main(): the fixed offset from the
    attachment site to the gripper's grasp centre, read off a throwaway pose."""
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    res = solve_ik(arm, tool_down_pose(*cfg.bin[0], 0.20), cfg.ik_seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    return data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])


def random_config(rng):
    """A fresh CellConfig with a randomized bin grid and fixture position;
    everything else (pallet, home pose, part/table geometry, reach limits)
    stays at the CellConfig defaults."""
    cx, cy, _, _, sx0, sy0 = DEFAULT_CONFIG.bin_grid

    nx = int(rng.integers(2, 5))          # 2..4 parts wide
    ny = int(rng.integers(2, 4))          # 2..3 parts deep
    if nx * ny < N_REACHABLE_REQUESTS:
        # build_plan pulls one bin part per reachable request with no
        # wraparound -- too few parts is an IndexError, not a failed run.
        ny = -(-N_REACHABLE_REQUESTS // nx)   # ceil division

    sx = sx0 * (1.0 + rng.uniform(-0.20, 0.20))
    sy = sy0 * (1.0 + rng.uniform(-0.20, 0.20))

    fx0, fy0 = DEFAULT_CONFIG.fixture_xy
    fixture_xy = (fx0 + rng.uniform(-0.03, 0.03), fy0 + rng.uniform(-0.06, 0.06))

    return replace(DEFAULT_CONFIG, bin_grid=(cx, cy, nx, ny, sx, sy), fixture_xy=fixture_xy)


def run_once(mujoco, arm, cfg):
    model = build_cell(mujoco, cfg)
    gl = gripper_length(mujoco, model, arm, cfg)
    planner = Planner(mujoco, model, arm, gl, cfg=cfg)

    t0 = time.perf_counter()
    frames, meta = build_plan(planner)
    wall_time_s = time.perf_counter() - t0

    reachable = meta["total"] - 1     # the one deliberately-unreachable slot
    success_rate = meta["placed"] / reachable if reachable else float("nan")
    return dict(
        nx=cfg.bin_grid[2], ny=cfg.bin_grid[3],
        sx=cfg.bin_grid[4], sy=cfg.bin_grid[5],
        fixture_x=cfg.fixture_xy[0], fixture_y=cfg.fixture_xy[1],
        n_parts=cfg.n_parts, placed=meta["placed"], rejected=meta["rejected"],
        total=meta["total"], replans=meta["replans"], accuracy_mm=meta["accuracy_mm"],
        wall_time_s=wall_time_s, n_frames=len(frames), success_rate=success_rate,
    )


def summarize(rows):
    success = np.array([r["success_rate"] for r in rows])
    wall = np.array([r["wall_time_s"] for r in rows])
    placed = np.array([r["placed"] for r in rows])
    replans = np.array([r["replans"] for r in rows])
    err = np.array([r["accuracy_mm"] for r in rows])
    s_per_part = np.divide(wall, placed, out=np.full_like(wall, np.nan), where=placed > 0)

    print(f"\n{'=' * 60}\nBenchmark summary over {len(rows)} runs\n{'=' * 60}")
    print(f"Success rate      mean {success.mean():6.1%}   min {success.min():6.1%}   "
          f"max {success.max():6.1%}")
    print(f"Cycle time (wall) mean {wall.mean():6.3f} s   min {wall.min():6.3f} s   "
          f"max {wall.max():6.3f} s")
    print(f"                  mean {np.nanmean(s_per_part):6.3f} s/part")
    print(f"Re-plans          mean {replans.mean():6.2f}   min {replans.min():d}   "
          f"max {replans.max():d}")
    print(f"Placement error   min {err.min():6.3f} mm   mean {err.mean():6.3f} mm   "
          f"max {err.max():6.3f} mm   p95 {np.percentile(err, 95):6.3f} mm")
    print()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "nx", "ny", "sx", "sy", "fixture_x", "fixture_y", "n_parts",
                  "placed", "rejected", "total", "replans", "accuracy_mm",
                  "wall_time_s", "success_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"run": i, **{k: r[k] for k in fieldnames if k != "run"}})
    print(f"wrote {path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="number of randomized runs")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for layout randomization")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmark_results.csv",
                        help="CSV output path")
    args = parser.parse_args()

    import mujoco

    rng = np.random.default_rng(args.seed)
    arm = SerialArm.ur5e()   # kinematic chain is fixed, reuse across runs

    rows = []
    for i in range(args.n):
        cfg = random_config(rng)
        row = run_once(mujoco, arm, cfg)
        rows.append(row)
        print(f"run {i:3d}/{args.n}  bin {row['nx']}x{row['ny']} ({row['n_parts']} parts)  "
              f"placed {row['placed']}/{row['total'] - 1}  replans {row['replans']}  "
              f"err {row['accuracy_mm']:.3f} mm  {row['wall_time_s'] * 1000:.0f} ms")

    summarize(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()

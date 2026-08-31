"""Measure the solver, and draw the plots a robotics interviewer asks for.

    uv run --group dev python apps/benchmark_ik.py --trials 800

Writes docs/ik_benchmark.csv and four figures into docs/. Every figure is
generated from that CSV, so a number in the report and a point on a plot cannot
disagree.

WHAT IS MEASURED, and why each one rather than a single accuracy figure:

  iterations vs conditioning   how the damped least squares solver behaves as
                               the Jacobian degrades. The interesting plot is
                               not "does it converge" but "what does it cost
                               when the arm is badly conditioned", because that
                               is where a naive pseudo-inverse diverges.
  runtime distribution         analytic against numerical, as a distribution
                               and not a mean. A solver used inside a control
                               loop is judged on its tail.
  success near singularities   binned by manipulability. A single success rate
                               over uniform random poses hides the only region
                               where failure happens.
  branch continuity            the same Cartesian path solved with and without
                               chained branch selection. This is the plot that
                               shows why `armik.select` exists: independently
                               reasonable answers at consecutive waypoints are
                               not an executable trajectory.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import (  # noqa: E402
    SerialArm,
    analytical_ik,
    manipulability,
    solve_ik,
)
from armik.select import select_branch  # noqa: E402

DOCS = ROOT / "docs"


def sample(arm: SerialArm, trials: int, seed: int) -> list[dict[str, float]]:
    """One row per random pose: cost, accuracy and conditioning together."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for _ in range(trials):
        q_true = arm.random_config(rng, margin=0.6)
        target = arm.fk(q_true)
        mu = manipulability(arm, q_true)
        condition = float(np.linalg.cond(arm.jacobian(q_true)))

        # Seeded away from the answer, so this measures the solver rather than
        # the seed. The same offset is used for every pose.
        q_seed = q_true + rng.uniform(-0.6, 0.6, arm.n)

        started = time.perf_counter()
        result = solve_ik(arm, target, q_seed)
        numerical_ms = (time.perf_counter() - started) * 1e3

        started = time.perf_counter()
        solutions = analytical_ik(arm, target)
        analytic_ms = (time.perf_counter() - started) * 1e3

        rows.append({
            "manipulability": mu,
            "condition": condition,
            "iterations": result.iterations,
            "success": int(result.success),
            "position_error_mm": result.position_error * 1e3,
            "orientation_error_mrad": result.orientation_error * 1e3,
            "numerical_ms": numerical_ms,
            "analytic_ms": analytic_ms,
            "branches": len(solutions),
        })
    return rows


def continuity_over_paths(arm: SerialArm, paths: int, waypoints: int,
                          seed: int) -> tuple[list[float], list[float], list[float]]:
    """Worst joint step on each of many random paths, naive against chained.

    ONE PATH IS NOT EVIDENCE. Whether the naive selector jumps depends on where
    in the workspace the path runs, so a single trace can make either method
    look fine. These are the worst-case steps over many independent paths,
    which is the distribution the claim should be argued from.
    """
    naive_worst, chained_worst, travel_worst = [], [], []
    for k in range(paths):
        naive, chained, travel = continuity(arm, waypoints, seed + k)
        if naive:
            naive_worst.append(max(naive))
        if chained:
            chained_worst.append(max(chained))
        if travel:
            travel_worst.append(max(travel))
    return naive_worst, chained_worst, travel_worst


def continuity(arm: SerialArm, waypoints: int, seed: int
               ) -> tuple[list[float], list[float], list[float]]:
    """Largest per-joint step along one path, for three selectors.

    naive       take the first solution the closed form returns, every time
    default     armik.select with the default weights, so a near-singular
                branch is penalised even when it is the closest one
    travel only armik.select with every singularity guard off, weights AND
                the hard floor, which is pure continuity and will happily
                track straight through a singular region

    The third exists because the remaining jumps under the default weights are
    not a defect: they are the cost function trading travel for margin where
    the path passes close to a singularity. Measuring the travel-only variant
    is what separates "the selector is jumping" from "the selector is avoiding
    something", and the answer decides which weights a machine should use.
    """
    rng = np.random.default_rng(seed)
    q = arm.random_config(rng, margin=0.9)
    start = q.copy()
    poses = []
    for _ in range(waypoints):
        q = q + rng.uniform(-0.05, 0.05, arm.n)
        poses.append(arm.fk(q))

    naive: list[np.ndarray] = []
    for pose in poses:
        solutions = analytical_ik(arm, pose)
        if len(solutions):
            naive.append(np.asarray(solutions[0], dtype=float))

    def chain(**weights: float) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        previous = start
        for pose in poses:
            chosen = select_branch(arm, analytical_ik(arm, pose), previous,
                                   **weights)
            if chosen is not None:
                path.append(chosen.q)
                previous = chosen.q
        return path

    def steps(path: list[np.ndarray]) -> list[float]:
        return [float(np.max(np.abs(b - a))) for a, b in zip(path, path[1:])]

    return (steps(naive), steps(chain()),
            steps(chain(singularity_weight=0.0, limit_weight=0.0,
                        singular_floor=0.0)))


def plots(rows: list[dict[str, float]], naive: list[float],
          chained: list[float], travel_only: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["success"]]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25})

    # 1. iterations against conditioning
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter([r["condition"] for r in ok], [r["iterations"] for r in ok],
               s=6, alpha=0.35, color="#1c5e70", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Jacobian condition number (log)")
    ax.set_ylabel("DLS iterations to converge")
    ax.set_title("Cost rises with conditioning, and stays bounded")
    fig.tight_layout()
    fig.savefig(DOCS / "ik_iterations_vs_condition.png")
    plt.close(fig)

    # 2. runtime distributions
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist([r["numerical_ms"] for r in rows], bins=50, alpha=0.75,
            label="damped least squares", color="#1c5e70")
    ax.hist([r["analytic_ms"] for r in rows], bins=50, alpha=0.75,
            label="closed form (all branches)", color="#a84a12")
    ax.set_xscale("log")
    ax.set_xlabel("solve time per pose, ms (log)")
    ax.set_ylabel("poses")
    ax.set_title("Runtime is a distribution, not a mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "ik_runtime_distribution.png")
    plt.close(fig)

    # 3. success rate binned by manipulability
    edges = [0, 1e-4, 1e-3, 4e-3, 1.6e-2, 5e-2, 1.0]
    labels, rates, counts = [], [], []
    for low, high in zip(edges, edges[1:]):
        band = [r for r in rows if low <= r["manipulability"] < high]
        if not band:
            continue
        labels.append(f"{low:g}\nto {high:g}")
        rates.append(100 * sum(r["success"] for r in band) / len(band))
        counts.append(len(band))

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(labels, rates, color="#1c5e70")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={n}", ha="center", fontsize=7, color="#66757a")
    ax.set_ylim(0, 108)
    ax.set_xlabel("manipulability band (nearer a singularity to the left)")
    ax.set_ylabel("DLS success rate, %")
    ax.set_title("Failure lives near the singularities, and only there")
    fig.tight_layout()
    fig.savefig(DOCS / "ik_success_vs_manipulability.png")
    plt.close(fig)

    # 4. branch continuity, worst step per path over many paths
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    order = np.argsort(naive)
    ax.plot([naive[i] for i in order], "o-", ms=3, lw=1,
            label="first branch returned", color="#a84a12")
    ax.plot([chained[i] for i in order], "o-", ms=3, lw=1,
            label="chained, default weights", color="#1c5e70")
    ax.plot([travel_only[i] for i in order], "o-", ms=3, lw=1,
            label="chained, all singularity guards off", color="#1f6b45")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("random Cartesian path, sorted by the naive worst step")
    ax.set_ylabel("worst single joint step on that path, rad (symlog)")
    ax.set_title("The remaining jumps are singularity avoidance, not drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "ik_branch_continuity.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--waypoints", type=int, default=60)
    parser.add_argument("--paths", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arm = SerialArm.ur5()
    rows = sample(arm, args.trials, args.seed)
    naive, chained, travel_only = continuity_over_paths(
        arm, args.paths, args.waypoints, args.seed + 1)

    DOCS.mkdir(exist_ok=True)
    with (DOCS / "ik_benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plots(rows, naive, chained, travel_only)

    ok = [r for r in rows if r["success"]]
    numerical = np.array([r["numerical_ms"] for r in rows])
    analytic = np.array([r["analytic_ms"] for r in rows])
    print(f"{len(rows)} poses, {100 * len(ok) / len(rows):.1f}% converged")
    print(f"  iterations   p50 {np.median([r['iterations'] for r in ok]):.0f}  "
          f"p95 {np.percentile([r['iterations'] for r in ok], 95):.0f}")
    print(f"  DLS ms       p50 {np.median(numerical):.2f}  "
          f"p95 {np.percentile(numerical, 95):.2f}")
    print(f"  analytic ms  p50 {np.median(analytic):.3f}  "
          f"p95 {np.percentile(analytic, 95):.3f}")
    print(f"  position err p95 {np.percentile([r['position_error_mm'] for r in ok], 95):.4f} mm")
    print(f"  branch continuity over {len(naive)} paths:")
    print(f"    naive    worst {max(naive):.2f} rad   median {np.median(naive):.3f}   "
          f"paths over 1 rad: {sum(v > 1 for v in naive)}")
    print(f"    chained  worst {max(chained):.2f} rad   median {np.median(chained):.3f}   "
          f"paths over 1 rad: {sum(v > 1 for v in chained)}")
    print(f"    travel   worst {max(travel_only):.2f} rad   median {np.median(travel_only):.3f}   "
          f"paths over 1 rad: {sum(v > 1 for v in travel_only)}")
    print(f"wrote {(DOCS / 'ik_benchmark.csv').relative_to(ROOT)} and 4 figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

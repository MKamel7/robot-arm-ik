"""Measure what null-space control on a 7R arm actually buys, and what it costs.

    uv run python apps/benchmark_redundancy.py

Writes docs/redundancy_benchmark.csv and two figures into docs/. Every number in
the README's redundancy section comes from this file, so a claim there can be
regenerated rather than believed.

THE QUESTION. A 7R arm has one spare degree of freedom at every non-singular
configuration. Spending it on a secondary objective is free in task space, so
the interesting question is not "does it work" but "how much gain, and what does
the other objective lose while one of them is being served".

THE ANSWER, and it is the reason this sweep exists rather than a single number:
BOTH OBJECTIVES HAVE AN OPTIMAL GAIN AND ARE WORSE THAN NOTHING PAST IT. The
controller integrates in discrete steps, so a large null-space step overshoots
the hill it is climbing. Anyone reaching for a bigger gain should see the curve
turn over first.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from armik.redundancy import (
    follow,
    joint_limit_gradient,
    manipulability_gradient,
)
from armik.robot import SerialArm

DOCS = Path(__file__).resolve().parents[1] / "docs"

#: The standard Panda ready configuration, which is also where the FK is
#: validated against the published flange position.
READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2,
                  np.pi / 4])

#: A straight-line Cartesian drag chosen because it walks the arm toward its
#: joint limits. A path that stayed comfortably in the middle of the workspace
#: would let limit avoidance look effective by never being needed.
STEPS = 60
TWIST = np.array([0.012, 0.0, -0.004, 0.0, 0.0, 0.0])

LIMIT_GAINS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0)
MANIP_GAINS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 4.0, 8.0)


def sweep(objective, gains) -> list[dict[str, float]]:
    arm = SerialArm.panda()
    twists = [TWIST] * STEPS
    rows: list[dict[str, float]] = []
    for gain in gains:
        trace = follow(arm, READY, twists,
                       objective=None if gain == 0.0 else objective, gain=gain)
        rows.append({
            "gain": gain,
            "worst_margin": trace.worst_margin,
            "mean_manipulability": trace.mean_manipulability,
            "worst_task_error": trace.worst_task_error,
        })
    return rows


def write_csv(limit_rows, manip_rows) -> Path:
    path = DOCS / "redundancy_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "gain", "worst_joint_limit_margin",
                         "mean_manipulability", "worst_task_error"])
        for name, rows in (("joint_limit_avoidance", limit_rows),
                           ("manipulability_maximisation", manip_rows)):
            for row in rows:
                writer.writerow([name, f"{row['gain']:g}",
                                 f"{row['worst_margin']:.6f}",
                                 f"{row['mean_manipulability']:.6f}",
                                 f"{row['worst_task_error']:.3e}"])
    return path


def figures(limit_rows, manip_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for rows, name, measure, label, filename in (
        (limit_rows, "joint-limit avoidance", "worst_margin",
         "worst joint-limit margin", "redundancy_limit_gain.png"),
        (manip_rows, "manipulability maximisation", "mean_manipulability",
         "mean manipulability", "redundancy_manipulability_gain.png"),
    ):
        gains = [r["gain"] for r in rows]
        served = [r[measure] for r in rows]
        # The other objective, so the cost of serving this one is visible on
        # the same axes. Reporting only the served measure would make every
        # secondary objective look free, which is the claim being tested.
        other_key = ("mean_manipulability" if measure == "worst_margin"
                     else "worst_margin")
        other = [r[other_key] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(gains, served, "o-", label=label)
        ax.plot(gains, other, "s--", label=other_key.replace("_", " "))
        ax.axhline(served[0], color="grey", linewidth=0.8,
                   label="baseline, no null-space control")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("null-space gain")
        ax.set_ylabel("value")
        ax.set_title(f"{name}: the gain has an optimum")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DOCS / filename, dpi=140)
        plt.close(fig)


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    limit_rows = sweep(joint_limit_gradient, LIMIT_GAINS)
    manip_rows = sweep(manipulability_gradient, MANIP_GAINS)

    path = write_csv(limit_rows, manip_rows)
    try:
        figures(limit_rows, manip_rows)
    except ImportError:
        print("matplotlib is absent, so the CSV was written without figures")

    best_limit = max(limit_rows, key=lambda r: r["worst_margin"])
    best_manip = max(manip_rows, key=lambda r: r["mean_manipulability"])
    baseline = limit_rows[0]

    print(f"wrote {path.relative_to(DOCS.parent)}")
    print(f"baseline                 margin {baseline['worst_margin']:.4f}  "
          f"mean w {baseline['mean_manipulability']:.5f}")
    print(f"best limit avoidance     gain {best_limit['gain']:g}  "
          f"margin {best_limit['worst_margin']:.4f}  "
          f"mean w {best_limit['mean_manipulability']:.5f}")
    print(f"best manipulability      gain {best_manip['gain']:g}  "
          f"margin {best_manip['worst_margin']:.4f}  "
          f"mean w {best_manip['mean_manipulability']:.5f}")
    print(f"worst task error overall "
          f"{max(r['worst_task_error'] for r in limit_rows + manip_rows):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

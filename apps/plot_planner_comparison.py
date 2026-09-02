"""Draw the planner comparison from its CSV.

    uv run python apps/plot_planner_comparison.py

Separate from `benchmark_planners.py` because that one runs against a sourced
ROS 2 and MoveIt, while this runs in the project's own uv environment where
matplotlib lives. Splitting them also means a figure can be redrawn without
re-running an hour of planning, and that the CSV, not the plot, stays the
artefact everything else is checked against.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parents[1] / "docs"
CSV_PATH = DOCS / "planner_comparison.csv"

#: Plot order, grouped by family so the pairs sit side by side. The comparison
#: is armik against its reference, and a bar chart sorted by score would break
#: exactly the pairing the study is about.
ORDER = [
    ("joint interpolation", "armik"),
    ("pilz ptp", "reference"),
    ("cartesian interpolation", "armik"),
    ("pilz lin", "reference"),
    ("rrt connect", "armik"),
    ("ompl rrtconnect", "reference"),
    ("chomp", "reference"),
    ("stomp", "reference"),
]
COLOURS = {"armik": "#1f77b4", "reference": "#9aa5b1"}


def load():
    with CSV_PATH.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def column(rows, planner, key, only_valid=False):
    out = []
    for row in rows:
        if row["planner"] != planner or row["succeeded"] != "True":
            continue
        if only_valid and row["collision_free"] != "True":
            continue
        if row[key] not in ("", None):
            out.append(float(row[key]))
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load()
    problems = len({row["problem"] for row in rows})
    labels = [name for name, _ in ORDER]
    colours = [COLOURS[kind] for _, kind in ORDER]
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    valid = [len(column(rows, name, "joint_length", only_valid=True)) for name, _ in ORDER]
    returned = [len(column(rows, name, "joint_length")) for name, _ in ORDER]
    axes[0].bar(x, returned, color=colours, alpha=0.35, label="returned a path")
    axes[0].bar(x, valid, color=colours, label="path is collision free")
    axes[0].set_ylabel(f"problems (of {problems})")
    axes[0].set_title("What each planner actually delivered")
    axes[0].legend(frameon=False, fontsize=8)

    # A straight line in joint space has a second difference of zero, so the two
    # interpolations score around 1e-31, which is numerical zero rather than a
    # measurement. Plotted on a shared log axis they stretch it over 27 decades
    # and flatten the range where the planners actually differ, so the axis is
    # floored and they are labelled for what they are: straight by construction.
    SMOOTHNESS_FLOOR = 1e-9

    for axis, key, title, log in (
        (axes[1], "smoothness", "Smoothness of the path (lower is smoother)", True),
        (axes[2], "min_clearance_m", "Clearance to the shelf, metres (higher is safer)", False),
    ):
        data = [column(rows, name, key, only_valid=(key == "min_clearance_m")) for name, _ in ORDER]
        drawn = [d if d else [np.nan] for d in data]
        parts = axis.boxplot(drawn, positions=x, widths=0.6, patch_artist=True,
                             medianprops={"color": "black"})
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.8)
        if log:
            axis.set_yscale("log")
            axis.set_ylim(bottom=SMOOTHNESS_FLOOR)
        axis.set_title(title)
        for i, values in enumerate(data):
            if not values:
                axis.text(i, axis.get_ylim()[0], "none", ha="center", va="bottom",
                          fontsize=7, rotation=90)
            elif log and max(values) < SMOOTHNESS_FLOOR:
                axis.text(i, SMOOTHNESS_FLOOR * 1.6, "straight by\nconstruction",
                          ha="center", va="bottom", fontsize=7)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle(f"Panda, {problems} problems where a straight joint-space line is blocked\n"
                 "blue: implemented in this repository, grey: reference implementation",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DOCS / "planner_comparison.png"
    fig.savefig(out, dpi=140)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

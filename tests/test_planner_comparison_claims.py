"""The planner table in the README is the one in the CSV.

The README quotes eight planners with four numbers each, from a sweep that
needs a sourced ROS 2 and several minutes to reproduce. `test_documented_counts`
already argues the general case: a number a human has to remember to update is a
number that will be wrong. These are worse than a test count, because nobody
reading the README can tell a stale median from a fresh one.

So the CSV is committed and this recomputes every published figure from it.
No ROS needed, which is the point: CI has none, and the claim is checkable
anyway.

Verify by falsification: change a digit in the README table and this goes red.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CSV = ROOT / "docs" / "planner_comparison.csv"

#: README label to the planner name in the CSV.
NAMES = {
    "joint interpolation": "joint interpolation",
    "Pilz PTP": "pilz ptp",
    "Cartesian interpolation": "cartesian interpolation",
    "Pilz LIN": "pilz lin",
    "RRT-Connect": "rrt connect",
    "OMPL RRTConnect": "ompl rrtconnect",
    "CHOMP": "chomp",
    "STOMP": "stomp",
}


@pytest.fixture(scope="module")
def rows():
    if not CSV.exists():
        pytest.skip("docs/planner_comparison.csv has not been generated")
    with CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def table():
    """The README's planner table, as {label: [cells]}."""
    text = README.read_text(encoding="utf-8")
    section = re.search(r"## Planners: the ones here against the ones people ship(.*?)\n## ",
                        text, re.S)
    assert section, "the README has no planner comparison section"
    parsed = {}
    for line in section.group(1).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in NAMES:
            parsed[cells[0]] = cells
    assert set(parsed) == set(NAMES), f"table rows {sorted(parsed)} != {sorted(NAMES)}"
    return parsed


def _for(rows, planner):
    return [r for r in rows if r["planner"] == planner]


def _clean(cell):
    return cell.replace("**", "").strip()


def test_every_solved_count_matches_the_csv(rows, table):
    for label, cells in table.items():
        solved, total = _clean(cells[2]).split("/")
        mine = _for(rows, NAMES[label])
        assert len(mine) == int(total), f"{label}: table says {total} problems"
        succeeded = sum(1 for r in mine if r["succeeded"] == "True")
        assert succeeded == int(solved), f"{label}: table says {solved} solved, csv says {succeeded}"


def test_every_collision_free_count_matches_the_csv(rows, table):
    """The column that separates a returned path from a usable one."""
    for label, cells in table.items():
        claimed = int(_clean(cells[3]))
        actual = sum(1 for r in _for(rows, NAMES[label]) if r["collision_free"] == "True")
        assert claimed == actual, f"{label}: table says {claimed} collision free, csv says {actual}"


def test_the_published_medians_match_the_csv(rows, table):
    """Joint travel and clearance, to the precision the README prints them."""
    for label, cells in table.items():
        mine = [r for r in _for(rows, NAMES[label]) if r["succeeded"] == "True"]
        for column, key, only_valid in ((5, "joint_length", False),
                                        (7, "min_clearance_m", True)):
            claimed = _clean(cells[column])
            if claimed in ("n/a", "straight"):
                continue
            source = [r for r in mine if not only_valid or r["collision_free"] == "True"]
            values = [float(r[key]) for r in source if r[key] != ""]
            assert values, f"{label}: README quotes {claimed} where the csv has nothing"
            assert float(claimed) == pytest.approx(float(np.median(values)), abs=0.006), (
                f"{label} {key}: README says {claimed}, csv median is {np.median(values):.4f}")


def test_the_interpolations_really_did_collide_every_time(rows):
    """The README's strongest claim about this repository's own code.

    It says joint interpolation returns a colliding path on all 20 problems.
    That is a statement against interest and the one most worth pinning.
    """
    for planner in ("joint interpolation", "pilz ptp"):
        mine = _for(rows, planner)
        assert mine, f"no {planner} rows"
        assert all(r["succeeded"] == "True" for r in mine)
        assert not any(r["collision_free"] == "True" for r in mine)


def test_the_two_joint_interpolations_agree_path_by_path(rows):
    """The validation result: the same function, written twice."""
    mine = {r["problem"]: r["joint_length"] for r in _for(rows, "joint interpolation")}
    reference = {r["problem"]: r["joint_length"] for r in _for(rows, "pilz ptp")}

    assert mine and mine.keys() == reference.keys()
    for problem, length in mine.items():
        assert float(length) == pytest.approx(float(reference[problem]), abs=1e-4), (
            f"problem {problem}: this repo {length}, Pilz {reference[problem]}")

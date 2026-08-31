"""Every test count the documents claim must be the real one.

WHY THIS EXISTS. Within a single day this repository carried two different
wrong numbers: the README said 24 tests and the technical report said 52, while
the suite was 67. One of those was created by adding tests after updating the
document, which is the ordinary way a number goes stale and is invisible to
every other check here.

`fault-injection-harness` has a gate like this and it caught three stale counts
in one afternoon. The repositories without one drifted. That is the whole
argument: a number a human has to remember to update is a number that will be
wrong, and the fix is not more care, it is a check.

WHY COLLECTION RATHER THAN A RUN. It is fast, it needs no threshold to be met,
and the count is exactly what the documents claim. Collection does not execute
tests, so this cannot recurse.

THE COUNT DEPENDS ON THE ENVIRONMENT, which the first version of this file got
wrong and CI caught. `test_palletizing_cell.py` and `test_ur5e_mujoco.py` are
skipped at module level without the `sim` extras, so they are never collected:
53 items on the plain `test` job against 70 with MuJoCo installed. A single
documented number cannot be true in both, so the documents state the FULL
suite and this gate only enforces where the full suite exists. The `mujoco-sim`
job is where that is, which is the same job that already sets
ARMIK_REQUIRE_MUJOCO=1 to stop the cross-validation quietly skipping.

A number that is deliberately historical should not be written as "N tests" at
all. Reword it instead of exempting it: an exemption is a second place for the
truth to live.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ("README.md", "docs/TECHNICAL_REPORT.md")


def collected_tests() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    found = re.search(r"^(\d+) tests collected", result.stdout, re.M)
    assert found is not None, (
        f"could not count the suite:\n{result.stdout[-1500:]}")
    return int(found.group(1))


def full_suite_available() -> bool:
    """Is every test collectable here, or are the sim-only files skipped?"""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_stated_test_count_is_the_real_one(document: str) -> None:
    path = ROOT / document
    if not path.is_file():
        pytest.skip(f"{document} is not in this repository")
    if not full_suite_available():
        pytest.skip(
            "mujoco is absent, so the sim-only files are not collected and the "
            "count here is not the one the documents state. The mujoco-sim CI "
            "job enforces this.")

    actual = collected_tests()
    claimed = {int(n) for n in re.findall(r"(\d+)\s+tests\b",
                                          path.read_text(encoding="utf-8"))}

    wrong = sorted(n for n in claimed if n != actual)
    assert not wrong, (
        f"{document} says {wrong} tests; the suite collects {actual}. "
        f"Update the document, or reword the claim if it is historical.")


def test_the_documents_state_the_count_at_all() -> None:
    """A gate over a claim nobody makes passes for the wrong reason.

    If both documents stopped mentioning the suite size this file would go
    green forever while saying nothing, which is the shape of check this
    repository is otherwise careful to avoid.
    """
    stated = [d for d in DOCUMENTS
              if (ROOT / d).is_file()
              and re.search(r"\d+\s+tests\b", (ROOT / d).read_text(encoding="utf-8"))]

    assert stated, "no document states a test count, so this gate guards nothing"

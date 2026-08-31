"""Cross-validate the UR5e forward kinematics against the MuJoCo Menagerie model.

The README and the technical report both state that this library's UR5e FK
agrees with the MuJoCo Menagerie `universal_robots_ur5e` model. Until this
file existed, nothing checked it: the only UR5e FK test asserted a hardcoded
position that had been produced by hand, so the cross-validation was a claim
about a comparison that no longer happened anywhere.

This file performs the comparison. It loads the model that ships in
`assets/ur5e/`, drives it with the same joint vector the DH chain gets, and
compares the `attachment_site` pose against `SerialArm.ur5e().fk(q)`.

What the agreement actually is, measured over 5000 random configurations:

    position     max 1.489 mm, mean 0.983 mm, min 0.303 mm
    orientation  max 4.2e-6 degrees

**The residual is expected and is not an error in either model.** The
Menagerie XML rounds the UR5e link lengths to the millimetre, while the DH
table uses the Universal Robots datasheet values to a tenth of a millimetre:
0.163 against 0.1625 at the shoulder, 0.392 against 0.3922 at the elbow, 0.1
against 0.0996 at the wrist. Those offsets partly cancel depending on the
configuration, which is why the error varies between 0.3 mm and 1.5 mm rather
than being constant. The orientation is exact because none of the rounding
touches an axis direction.

So the bound below is 2 mm, not 1 mm: a tolerance the geometry cannot meet
would be a test that only ever reports the rounding.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from armik import SerialArm

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "ur5e" / "ur5e.xml"

# Measured bounds, with margin. See the module docstring for the real figures.
POSITION_TOLERANCE_M = 2.0e-3
MEAN_POSITION_TOLERANCE_M = 1.2e-3
ORIENTATION_TOLERANCE_DEG = 1.0e-4

N_CONFIGS = 500
SEED = 12345


def _mujoco():
    """Import mujoco, or skip; unless the environment says it must be here.

    Skipping on a developer machine without the `sim` extras is reasonable.
    Skipping in the CI job whose entire purpose is to install those extras
    would mean this check silently stopped running, which is the failure mode
    this file was written to close. That job sets ARMIK_REQUIRE_MUJOCO=1.
    """
    try:
        import mujoco
    except ImportError:  # pragma: no cover - exercised by the absence itself
        if os.environ.get("ARMIK_REQUIRE_MUJOCO") == "1":
            pytest.fail(
                "ARMIK_REQUIRE_MUJOCO=1 but mujoco is not importable. "
                "The MuJoCo cross-validation did not run."
            )
        pytest.skip("mujoco not installed (uv sync --group sim)")
    return mujoco


@pytest.fixture(scope="module")
def errors():
    """Position and orientation error against MuJoCo over N_CONFIGS poses."""
    mujoco = _mujoco()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    assert site >= 0, "the UR5e model has no attachment_site"

    arm = SerialArm.ur5e()
    rng = np.random.default_rng(SEED)

    position, orientation = [], []
    for _ in range(N_CONFIGS):
        q = rng.uniform(-np.pi, np.pi, arm.n)

        data.qpos[: arm.n] = q
        mujoco.mj_kinematics(model, data)

        T = arm.fk(q)
        position.append(float(np.linalg.norm(data.site_xpos[site] - T[:3, 3])))

        r_mujoco = data.site_xmat[site].reshape(3, 3)
        cos_angle = (np.trace(r_mujoco.T @ T[:3, :3]) - 1.0) / 2.0
        orientation.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))))

    return np.array(position), np.array(orientation)


def test_the_model_ships_with_the_repository():
    # The cross-validation is worthless if it silently falls back to a model
    # downloaded at test time, which would not be the one the demos drive.
    assert MODEL_PATH.is_file(), f"missing {MODEL_PATH}"


def test_zero_pose_matches_the_model_not_just_the_golden():
    """Tie the hardcoded golden in test_ur5e.py back to MuJoCo.

    test_ur5e_fk_golden asserts a frozen constant. That constant is only
    meaningful if it is where the MuJoCo model actually puts the flange, and
    this is the test that says so.
    """
    mujoco = _mujoco()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    data.qpos[:6] = 0.0
    mujoco.mj_kinematics(model, data)

    golden = np.array([-0.8172, -0.2329, 0.0628])
    assert np.linalg.norm(data.site_xpos[site] - golden) < POSITION_TOLERANCE_M


def test_fk_position_agrees_across_the_workspace(errors):
    position, _ = errors

    assert position.max() < POSITION_TOLERANCE_M, (
        f"worst-case position error {position.max() * 1e3:.3f} mm exceeds "
        f"{POSITION_TOLERANCE_M * 1e3:.1f} mm"
    )
    # A regression that doubles the error while staying under the worst-case
    # bound would slip past the line above. This one catches it.
    assert position.mean() < MEAN_POSITION_TOLERANCE_M, (
        f"mean position error {position.mean() * 1e3:.3f} mm exceeds "
        f"{MEAN_POSITION_TOLERANCE_M * 1e3:.1f} mm"
    )


def test_fk_orientation_agrees_essentially_exactly(errors):
    _, orientation = errors

    # Two orders of magnitude tighter than the position bound, because no
    # part of the millimetre rounding in the XML changes an axis direction.
    assert orientation.max() < ORIENTATION_TOLERANCE_DEG, (
        f"worst-case orientation error {orientation.max():.3e} deg exceeds "
        f"{ORIENTATION_TOLERANCE_DEG:.1e} deg"
    )


def test_the_identity_joint_mapping_is_load_bearing():
    """Feeding the model a permuted q must blow the tolerance.

    The README claims an identity mapping between this library's joint vector
    and the model's qpos. Agreement under the identity mapping is only
    evidence for that if disagreement follows from breaking it, otherwise the
    2 mm bound could be wide enough to swallow a wrong mapping and nobody
    would learn anything from the test above.
    """
    mujoco = _mujoco()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    arm = SerialArm.ur5e()
    q = np.array([0.3, -0.7, 1.1, -0.4, 0.9, 0.2])

    # Swap the shoulder-lift and elbow joints, the mildest realistic mix-up.
    data.qpos[:6] = q[[0, 2, 1, 3, 4, 5]]
    mujoco.mj_kinematics(model, data)

    error = np.linalg.norm(data.site_xpos[site] - arm.fk(q)[:3, 3])
    assert error > 10 * POSITION_TOLERANCE_M, (
        f"a permuted joint mapping still agreed to {error * 1e3:.3f} mm, so "
        "the tolerance is too loose to say anything about the mapping"
    )

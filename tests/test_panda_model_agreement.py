"""The DH Panda and the URDF Panda must be the same robot.

`apps/benchmark_planners.py` compares this repository's planners against
MoveIt's by handing both the same joint configurations. That comparison means
nothing unless `SerialArm.panda()`, a Craig-form DH transcription, and the
published URDF MoveIt plans on agree about where those configurations put the
tool. `robot.py` used to say the Panda was "NOT cross-validated against a
simulator, unlike ur5e()". This is that cross-validation.

It needs a ROS 2 environment for the URDF and is skipped without one, which is
also why the number it checks is printed into the benchmark's own output: CI
here is a plain uv environment with no ROS, so a reader of the CSV should not
have to take the agreement on faith either.
"""

import numpy as np
import pytest

from armik.robot import SerialArm

pytest.importorskip("moveit.core.robot_model",
                    reason="needs a sourced ROS 2 Jazzy with moveit_py")

READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
TIP = "panda_link8"


@pytest.fixture(scope="module")
def urdf_state():
    import tempfile

    from moveit.core.robot_model import RobotModel
    from moveit.core.robot_state import RobotState
    from moveit_configs_utils import MoveItConfigsBuilder

    config = MoveItConfigsBuilder("moveit_resources_panda").to_moveit_configs()
    paths = []
    for text, suffix in ((config.robot_description["robot_description"], ".urdf"),
                         (config.robot_description_semantic[
                             "robot_description_semantic"], ".srdf")):
        handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        handle.write(text)
        handle.close()
        paths.append(handle.name)
    return RobotState(RobotModel(urdf_xml_path=paths[0], srdf_xml_path=paths[1]))


def _urdf_flange(state, q):
    state.set_joint_group_positions("panda_arm", q)
    state.update()
    return np.array(state.get_global_link_transform(TIP))


def test_the_ready_pose_matches_the_published_flange_position(urdf_state):
    """The pose the README already quotes, now checked against the URDF too."""
    dh = SerialArm.panda().fk(READY)
    urdf = _urdf_flange(urdf_state, READY)

    assert np.allclose(dh[:3, 3], urdf[:3, 3], atol=1e-6)
    assert np.allclose(dh[:3, 3], [0.307, 0.0, 0.590], atol=1e-3)


def test_the_two_models_agree_across_the_workspace(urdf_state):
    """One pose agreeing could be luck. Two hundred random ones could not."""
    arm = SerialArm.panda()
    rng = np.random.default_rng(0)
    limits = arm.joint_limits

    worst_position = 0.0
    worst_rotation = 0.0
    for _ in range(200):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        dh = arm.fk(q)
        urdf = _urdf_flange(urdf_state, q)
        worst_position = max(worst_position,
                             float(np.linalg.norm(dh[:3, 3] - urdf[:3, 3])))
        worst_rotation = max(worst_rotation,
                             float(np.abs(dh[:3, :3] - urdf[:3, :3]).max()))

    assert worst_position < 1e-6, f"worst flange disagreement {worst_position * 1000:.4f} mm"
    assert worst_rotation < 1e-6, f"worst orientation disagreement {worst_rotation:.2e}"

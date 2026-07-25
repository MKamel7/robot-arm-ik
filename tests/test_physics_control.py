"""Physics-control app: PD + gravity-compensation trajectory tracking.

Validates apps/physics_control.py in isolation against a tiny synthetic
MuJoCo model (a double pendulum), not the full UR5e -- the controller is
generic and doesn't care which robot it drives; the pendulum is cheap to
step thousands of times and its dynamics (nonlinear, gravity-loaded,
coupled) are enough to exercise real PD + gravity-comp behaviour.

Needs the optional `sim` extras (mujoco); skipped cleanly when mujoco isn't
installed (the fast pure-NumPy CI job never sees this file run).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps"))

from physics_control import (        # noqa: E402
    PDGravityCompController, hold_pose, track_trajectory,
)
from armik import joint_trajectory   # noqa: E402

# A two-link pendulum swinging under gravity: one hinge joint per link, one
# torque ("motor") actuator per joint, actuator order == joint order. A
# small physics timestep (1 ms) and modest joint damping keep the explicit
# force application (PD torques are external forces, not implicitly
# integrated) numerically well-behaved at the gains used below.
PENDULUM_XML = """
<mujoco model="test_double_pendulum">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
      <body name="link2" pos="0 0 -0.3">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1"/>
    <motor name="motor2" joint="joint2"/>
  </actuator>
</mujoco>
"""

KP, KD = 40.0, 8.0       # picked empirically against this model: stable and
                          # accurate at dt=0.001 (see tuning sweep in the task)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(PENDULUM_XML)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


@pytest.fixture(scope="module")
def joint_ids(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("joint1", "joint2")]


def test_tracks_trapezoidal_move(model, data, joint_ids):
    """Command a trapezoidal joint move and confirm the controller actually
    tracks it through real physics stepping (mj_step), not teleportation."""
    q0 = np.array([0.0, 0.0])
    q1 = np.array([1.0, -0.7])
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    _, q_ref, qd_ref = joint_trajectory(q0, q1, v_max=1.2, a_max=2.5, dt=0.02)

    q_ach, qd_ach = track_trajectory(mujoco, model, data, joint_ids, KP, KD,
                                     q_ref, qd_ref, dt=0.02)

    err = np.abs(q_ach - q_ref)
    assert np.all(np.isfinite(q_ach)) and np.all(np.isfinite(qd_ach))
    assert err[-1].max() < 0.03, f"final tracking error too large: {err[-1]}"
    assert err.max() < 0.08, f"tracking error exceeded loose bound somewhere: {err.max()}"


def test_holds_static_pose_against_gravity(model, data, joint_ids):
    """A static reference pose must be held against gravity, not sag -- this
    specifically tests that the qfrc_bias feedforward is doing real work
    (a PD term alone sags noticeably at these gains; see the docstring in
    physics_control.py for why qfrc_bias is the correct fix)."""
    q_hold = np.array([np.pi / 2, 0.0])   # link1 horizontal: worst-case gravity torque
    data.qpos[:] = q_hold
    data.qvel[:] = 0.0

    q_ach, qd_ach = hold_pose(mujoco, model, data, joint_ids, KP, KD, q_hold, steps=2000)

    sag = np.abs(q_ach - q_hold)
    assert np.all(np.isfinite(q_ach))
    assert sag.max() < 0.01, f"sagged too far under gravity: max {sag.max()}"
    assert sag[-1].max() < 0.005, f"did not settle back onto the held pose: {sag[-1]}"


def test_gravity_compensation_reduces_sag_versus_pd_alone(model, joint_ids):
    """Direct check that qfrc_bias feedforward is pulling its weight: a bare
    PD term (no bias) sags substantially more than PD + gravity comp at the
    same gains and the same held pose."""
    q_hold = np.array([np.pi / 2, 0.0])

    data_gc = mujoco.MjData(model)
    data_gc.qpos[:] = q_hold
    q_ach, _ = hold_pose(mujoco, model, data_gc, joint_ids, KP, KD, q_hold, steps=1000)
    sag_with_gc = np.abs(q_ach[-1] - q_hold).max()

    data_pd = mujoco.MjData(model)
    data_pd.qpos[:] = q_hold
    ctrl = PDGravityCompController(mujoco, model, data_pd, joint_ids, KP, KD)
    for _ in range(1000):
        tau = KP * (q_hold - ctrl.q) + KD * (0.0 - ctrl.qd)     # PD only, no qfrc_bias
        data_pd.ctrl[joint_ids] = tau
        mujoco.mj_step(model, data_pd)
    sag_without_gc = np.abs(ctrl.q - q_hold).max()

    assert sag_with_gc < sag_without_gc / 10


def test_no_blowup_from_large_initial_error(model, data, joint_ids):
    """Start well away from the reference (a nontrivial initial error) and
    confirm the controller converges rather than diverging: qpos/qvel stay
    finite and bounded throughout, and the final error is small."""
    q_start = np.array([2.5, -2.0])
    q_target = np.array([0.3, -0.3])
    data.qpos[:] = q_start
    data.qvel[:] = 0.0
    _, q_ref, qd_ref = joint_trajectory(q_start, q_target, v_max=1.2, a_max=2.5, dt=0.02)

    q_ach, qd_ach = track_trajectory(mujoco, model, data, joint_ids, KP, KD,
                                     q_ref, qd_ref, dt=0.02)

    assert np.all(np.isfinite(q_ach)) and np.all(np.isfinite(qd_ach))
    assert np.abs(q_ach).max() < 50.0, "joint angles blew up"
    assert np.abs(qd_ach).max() < 50.0, "joint velocities blew up"
    final_err = np.abs(q_ach[-1] - q_ref[-1])
    assert final_err.max() < 0.03, f"did not converge from a large initial error: {final_err}"

"""Closed-loop joint tracking control for MuJoCo: PD + gravity compensation.

`palletizing_cell.py` and `pick_and_place_mujoco.py` both animate a planned
trajectory by teleporting the arm every frame (`data.qpos[:6] = q`) -- MuJoCo
only renders and runs collision queries, physics never actually steps the
arm's own dynamics. This module is the other half: a small, reusable
controller that tracks a reference joint trajectory by stepping physics
forward for real (`mj_step`), the way a real robot's joint-level servo loop
would, so tracking error is a real closed-loop quantity, not zero by
construction.

    tau = kp * (q_ref - q) + kd * (qd_ref - qd) + qfrc_bias

`qfrc_bias` is MuJoCo's per-DOF gravity + Coriolis + centrifugal torque, i.e.
exactly the torque needed to hold the current state -- using it directly as
the feedforward term is the standard gravity-compensation approach (no
hand-derived gravity model). The PD term then only has to correct the
tracking error, not fight gravity too, which is what keeps a lightly-damped
arm from sagging under its own weight at rest.

Torques are written to `data.ctrl`, so the MJCF actuators driving the tracked
joints must be raw torque (`<motor>`) actuators, not MuJoCo's built-in
position/velocity servos (those already close their own loop and would just
compete with this one).

Reference trajectories are whatever `armik.trajectory.joint_trajectory` /
`multi_waypoint_trajectory` return: q of shape (T, n), qd of shape (T, n),
sampled at a fixed `dt`. This module doesn't depend on armik or on any
particular robot -- it only needs joint/actuator ids and a MuJoCo
model/data, so it plugs into any MJCF model, robot or test rig alike.
"""

import numpy as np


class PDGravityCompController:
    """PD + gravity-compensation torque controller for a set of hinge/slide
    joints in a MuJoCo model.

    Parameters
    ----------
    mujoco : the mujoco module (passed in, not imported at module scope, so
        importing this file never requires mujoco to be installed).
    model, data : the MjModel / MjData being simulated.
    joint_ids : int array-like, MuJoCo joint ids of the tracked joints
        (e.g. `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)`).
        Must be 1-DOF joints (hinge or slide).
    kp, kd : scalar or length-n array-like, per-joint proportional/derivative
        gains.
    ctrl_ids : int array-like, `data.ctrl` indices the computed torques are
        written to, one per tracked joint. Defaults to `joint_ids`, which is
        only correct when the model's torque actuators happen to be indexed
        identically to the joints they drive; pass the actual actuator ids
        (`mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)`)
        whenever actuator order differs from joint order.
    """

    def __init__(self, mujoco, model, data, joint_ids, kp, kd, ctrl_ids=None):
        self.mj = mujoco
        self.model = model
        self.data = data
        self.joint_ids = np.asarray(joint_ids, dtype=int)
        n = len(self.joint_ids)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.ctrl_ids = np.asarray(
            joint_ids if ctrl_ids is None else ctrl_ids, dtype=int)
        self.kp = np.broadcast_to(kp, (n,)).astype(float)
        self.kd = np.broadcast_to(kd, (n,)).astype(float)

    @property
    def q(self):
        return self.data.qpos[self.qpos_ids]

    @property
    def qd(self):
        return self.data.qvel[self.dof_ids]

    def torque(self, q_ref, qd_ref):
        """PD + gravity/Coriolis feedforward torque for the current state."""
        bias = self.data.qfrc_bias[self.dof_ids]
        return self.kp * (np.asarray(q_ref) - self.q) \
            + self.kd * (np.asarray(qd_ref) - self.qd) + bias

    def step(self, q_ref, qd_ref):
        """Apply one control torque and step physics forward by one
        `model.opt.timestep`. Returns the achieved (q, qd) after the step."""
        self.data.ctrl[self.ctrl_ids] = self.torque(q_ref, qd_ref)
        self.mj.mj_step(self.model, self.data)
        return self.q.copy(), self.qd.copy()


def track_trajectory(mujoco, model, data, joint_ids, kp, kd, q_ref, qd_ref, dt,
                      ctrl_ids=None):
    """Track a reference joint trajectory by stepping physics forward.

    q_ref, qd_ref : arrays of shape (T, n) sampled every `dt` seconds -- the
        format `armik.trajectory.joint_trajectory` /
        `multi_waypoint_trajectory` return.
    dt : the reference trajectory's sample spacing. Physics is stepped at
        `model.opt.timestep`; each reference sample is held constant
        (zero-order hold) for `round(dt / model.opt.timestep)` physics
        substeps, with torque recomputed from the live state every substep
        (feedback runs at the physics rate, reference at the trajectory
        rate) -- so a coarse trajectory dt still gets a numerically stable,
        high-rate control loop.

    Returns (q_achieved, qd_achieved), each shape (T, n): the controller's
    actual `data.qpos` / `data.qvel` for the tracked joints at each reference
    sample time (after that sample's substeps have run), so tracking error
    can be measured directly against q_ref / qd_ref.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    qd_ref = np.asarray(qd_ref, dtype=float)
    controller = PDGravityCompController(mujoco, model, data, joint_ids, kp, kd, ctrl_ids)
    n_sub = max(1, round(dt / model.opt.timestep))

    T, n = q_ref.shape
    q_hist = np.empty((T, n))
    qd_hist = np.empty((T, n))
    for i in range(T):
        for _ in range(n_sub):
            controller.step(q_ref[i], qd_ref[i])
        q_hist[i] = controller.q
        qd_hist[i] = controller.qd
    return q_hist, qd_hist


def hold_pose(mujoco, model, data, joint_ids, kp, kd, q_hold, steps, ctrl_ids=None):
    """Convenience wrapper: hold a static reference pose for `steps` physics
    steps (qd_ref = 0 throughout). Useful for testing gravity-compensation
    sag / static disturbance rejection in isolation from trajectory tracking.
    Returns (q_achieved, qd_achieved) of shape (steps, n)."""
    q_hold = np.asarray(q_hold, dtype=float)
    n = len(np.atleast_1d(q_hold))
    q_ref = np.tile(q_hold, (steps, 1))
    qd_ref = np.zeros((steps, n))
    return track_trajectory(mujoco, model, data, joint_ids, kp, kd, q_ref, qd_ref,
                            dt=model.opt.timestep, ctrl_ids=ctrl_ids)

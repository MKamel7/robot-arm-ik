"""Palletizing cell: a UR5e transfers parts from a supply bin into a multi-layer
pallet grid, driven entirely by this library's inverse kinematics, rendered in
MuJoCo. This is the industrial version of the pick-and-place demo.

It goes beyond a happy-path animation and shows three things a real automation
cell has to do:

* Collision-aware routing. A machine fixture stands between the bin and the
  pallet. Before each transfer the planner checks the direct joint-space path
  for collisions (MuJoCo contact queries); when it is blocked it escalates
  through two re-routing strategies: a cheap lift-traverse-lower Cartesian
  detour over the obstacle at a handful of fixed clearance heights, and, only
  if that also fails to clear it, a general RRT-Connect joint-space search.
  Either way, the winning route is re-timed through real joint velocity/
  acceleration limits rather than played back at a fixed interpolation-step
  rate. The heads-up display counts how many moves were re-planned and shows
  which strategy fired.
* Multi-layer palletizing. Parts are stacked into a 2x2x2 pallet, so the place
  height changes per layer.
* Failure handling. One requested pallet slot is deliberately outside the arm's
  reachable workspace. A reachability check catches it, the slot is rejected on
  screen, and the cell carries on with the reachable ones instead of faking it.

Every joint angle comes from armik: `SerialArm.ur5e()` forward kinematics, the
damped-least-squares IK (`solve_ik`) for each waypoint, `joint_trajectory` /
`multi_waypoint_trajectory` for time-limited joint moves, `cartesian_line` for
the straight-line detour segments, and `rrt_connect` for the general fallback
route. The reported placement accuracy is the IK position residual. MuJoCo
renders the cell and animates the Robotiq 2F-85 gripper.

Maps to real German factory-automation work (end-of-line palletizing, machine
tending, intralogistics): Krones, Siemens, KUKA, BMW plant logistics, and
warehouse-robotics companies.

Run (needs the optional `sim` extras: mujoco, imageio, pillow):
    python apps/palletizing_cell.py             write a still
    python apps/palletizing_cell.py --save      write docs/palletizing_cell.gif
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import (                                                        # noqa: E402
    SerialArm, solve_ik, joint_trajectory, cartesian_line,
    multi_waypoint_trajectory, path_trajectory, rrt_connect,
)
from armik.rotations import matrix_from_rotvec                            # noqa: E402

UR5E_SCENE = ROOT / "assets" / "ur5e" / "scene.xml"
GRIPPER = ROOT / "assets" / "robotiq_2f85" / "2f85.xml"

# The 6 arm joints/actuators, in the order ur5e.xml declares them. Used by
# physics-mode execution (see build_cell(physics=True) and render(physics=True)
# below) to look up joint/actuator ids and to convert the default position-
# servo actuators into raw-torque ones.
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow",
                  "wrist_1", "wrist_2", "wrist_3"]
# PD + gravity-comp gains for physics.track the reference trajectory (tuned
# against build_cell(physics=True)'s torque actuators, see this task's
# write-up for the tracking-error numbers these produce): higher for the
# size3 shoulder/elbow joints (bigger links, forcerange +-150 Nm), lower for
# the size1 wrists (forcerange +-28 Nm).
PHYSICS_KP = np.array([250.0, 250.0, 250.0, 40.0, 40.0, 40.0])
PHYSICS_KD = np.array([25.0, 25.0, 25.0, 6.0, 6.0, 6.0])


def grid(cx, cy, nx, ny, sx, sy):
    return [(cx + (i - (nx - 1) / 2) * sx, cy + (j - (ny - 1) / 2) * sy)
            for j in range(ny) for i in range(nx)]


@dataclass
class CellConfig:
    """Every scene, geometry, and planning constant for the palletizing cell,
    gathered in one place so the scene builder, planner, and renderer all read
    from the same source of truth (and a benchmark script can vary it later)."""

    table_h: float = 0.50
    part: float = 0.02                          # part half-size (0.04 m cube)
    z_pick: float = 0.038                       # grasp-centre height at a part on the table
    z_safe: float = 0.26                        # local lift/lower height
    dt: float = 0.04
    home_xy: tuple = (0.42, -0.27)              # ready pose above the supply bin (a palletizer waits here)
    home_z: float = 0.30                        # (the zero pose points behind the base, avoid it)
    ik_seed: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0]))
    ik_seed_alt: np.ndarray = field(                                        # elbow/wrist-flipped
        default_factory=lambda: np.array([0.0, -1.2, -1.5, 0.3, 1.57, 0.0]))  # branch, see ik_clear
    grip_joints: list = field(default_factory=lambda: [
        "g_right_driver_joint", "g_right_coupler_joint", "g_right_spring_link_joint",
        "g_right_follower_joint", "g_left_driver_joint", "g_left_coupler_joint",
        "g_left_spring_link_joint", "g_left_follower_joint",
    ])
    ur_links: set = field(default_factory=lambda: {
        "base", "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link"})
    fixture_xy: tuple = (0.41, 0.0)             # machine fixture, between bin and pallet
    fixture_half_h: float = 0.15                # fixture column half-height (0.30 m tall)
    fixture_half_w: tuple = (0.05, 0.06)        # fixture column half-extents in x, y
    bin_grid: tuple = (0.46, -0.30, 4, 2, 0.058, 0.075)      # cx, cy, nx, ny, sx, sy: 8 supply cells
    # 2x2 pallet footprint. y is 0.36 rather than 0.30 to keep the pallet out of
    # the fixture's shadow: at y=0.30 the corner nearest the fixture has no
    # collision-free IK solution on the arm's usual branch, so ik_clear had to
    # jump to a branch ~7 rad away to place there and jump back afterwards --
    # twice per plan, which swung shoulder_pan across 160 deg instead of the
    # 87 deg band the demo ran in before. 6 cm further out, the ordinary branch
    # is clear, ik_clear never fires, and the posture stays consistent.
    pallet_grid: tuple = (0.37, 0.36, 2, 2, 0.075, 0.075)
    unreachable_xy: tuple = (0.90, 0.30)        # outside the arm's workspace
    reach_max: float = 0.82                     # UR5e workspace radius for a downward grasp
    lift_heights: tuple = (0.48, 0.54, 0.60, 0.66)  # lift-over heuristic's candidate clearances
    rrt_step_size: float = 0.1                  # see Planner.move for the sizing rationale
    rrt_max_iters: int = 5000
    rrt_goal_bias: float = 0.05
    rrt_seed: object = None                     # fixed int -> deterministic (tests); None in prod
    v_max: float = 2.6                          # joint vel/accel limits shared by every re-timing
    a_max: float = 6.5                          # pass (direct moves, lift-over detour, RRT route)
    corner_rest_angle: float = 0.5              # rad; path_trajectory rests at turns sharper than this
    branch_gap_max: float = 0.1                 # rad; see Planner._move_impl's detour branch check
    physics_clearance_margin: float = 0.02      # metres; see build_cell(physics=True)'s docstring.
    # Planner.move's collision-aware routing only ever checks the KINEMATIC
    # reference path (Planner._collides teleports `q` into a fresh
    # mj_forward, zero tracking error by construction). Under real physics
    # execution (render(physics=True)), PD + gravity-comp tracking error
    # pushes the actual simulated arm off that reference -- a transient
    # ~0.15-0.26 rad wrist error was measured during a fast collision-
    # avoidance reroute segment (this task's write-up) -- and the real arm
    # clipped fixture_lamp even though the reference path it was tracking was
    # provably collision-free. Retuning PHYSICS_KD alone (tried up to 8x the
    # wrist gains, and 3x across every joint) never closed the gap: tracking
    # error only floors out around 0.12-0.14 rad before gains large enough to
    # start degrading tracking again (the 3x-every-joint case was *worse*
    # than baseline -- early sign of instability, not more margin to spend),
    # and the fixture_lamp contact persisted at every gain setting tried. So
    # the fix here is this margin, not gains: build_cell(physics=True) gives
    # the fixture geoms a MuJoCo contact `margin` of this size (only when
    # physics=True; a no-op for the default kinematic-teleport path, which
    # has zero tracking error and so nothing to buffer against), which makes
    # Planner._collides -- and so the whole lift-over/RRT escalation -- treat
    # "within this distance of the fixture" as blocked, not just "touching
    # it". That pushes every physics-mode route far enough from the fixture
    # to absorb the tracking error above with room to spare, without
    # touching the real contact-force physics real execution feels (the
    # geoms' `gap` is set equal to `margin`, so the force-enforcement
    # threshold stays at the true zero-clearance surface -- see build_cell).
    #
    # 0.02 m was picked empirically, not guessed: re-running the plan's full
    # 8-part physics-mode execution while sweeping this value, 0.015-0.025 m
    # all fully eliminate the real (zero-clearance) fixture contact with
    # PHYSICS_KP/PHYSICS_KD untouched, while 0.03 m already makes a couple of
    # tight bin/pallet-corner legs geometrically infeasible for the lift-over
    # heuristic and RRT-Connect both (Planner._move_impl gives up -- "NO
    # ROUTE FOUND" -- rather than finding a margin-respecting route). 0.02 m
    # keeps a comfortable margin below that infeasibility wall while being
    # ~30% past the smallest value tested that already worked.

    @property
    def layer_dz(self) -> float:
        return 2 * self.part                     # vertical pitch between pallet layers

    @property
    def bin(self) -> list:
        return grid(*self.bin_grid)              # 8 supply cells

    @property
    def pallet_xy(self) -> list:
        return grid(*self.pallet_grid)

    @property
    def requests(self) -> list:
        # requested pallet slots: base layer, one unreachable slot, then the top layer
        return ([(xy, 0) for xy in self.pallet_xy]
                + [(self.unreachable_xy, 0)]
                + [(xy, 1) for xy in self.pallet_xy])

    @property
    def n_parts(self) -> int:
        return len(self.bin)


DEFAULT_CONFIG = CellConfig()


def tool_down_pose(x, y, z):
    T = np.eye(4)
    T[:3, :3] = matrix_from_rotvec(np.array([0.0, np.pi, 0.0]))
    T[:3, 3] = [x, y, z]
    return T


def place_grasp_z(cfg, layer):
    return 0.05 + layer * cfg.layer_dz


def rest_centre_z(cfg, layer):
    return 0.032 + layer * cfg.layer_dz


def free_joint_qadr(mujoco, model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return model.jnt_qposadr[model.body_jntadr[bid]]


def build_cell(mujoco, cfg=DEFAULT_CONFIG, physics=False):
    """Build the cell's MjSpec/model.

    physics=False (default): the scene used by the fast kinematic teleport
    path (render(physics=False)) -- untouched, byte-for-byte the same MJCF
    this always compiled, on ur5e.xml's original <general> position-servo
    arm actuators (gaintype=fixed, biastype=affine, i.e. MuJoCo's built-in
    stiff PD-to-setpoint formula). Nothing in this function's physics=True
    branch runs.

    physics=True: an opt-in alternate build for real physics-tracked
    execution (render(physics=True) / apps/physics_control.py). Two changes,
    both additive (nothing about the default path is affected since it's a
    separate physics=False build):

    1. The 6 arm actuators are rewritten from position-servo <general>
       actuators to raw-torque ones (gaintype=fixed gainprm=1, biastype=none)
       -- exactly MuJoCo's <motor> actuator, just expressed as a <general>
       since MjSpec edits the actuators scene.xml already declared rather
       than replacing them. physics_control.py's PDGravityCompController
       writes torque to data.ctrl and assumes exactly this; the stock
       <general> actuators would instead reinterpret that torque value as a
       position setpoint for their own built-in servo, which is wrong (see
       this task's write-up for the fuller argument for why this path (a)
       was chosen over adapting to the stock affine-bias actuators).
    2. One inactive weld equality constraint per part, `g_base_mount` (the
       gripper's own base, rigidly fixed to the wrist -- see the `g_` attach
       prefix below) <-> `part{k}`. render(physics=True) flips a part's
       weld active/inactival at grasp/release time so a held part is
       actually pinned to the gripper by a real MuJoCo constraint under
       physics stepping, instead of the kinematic path's per-frame position
       snap to grasp_centre().
    """
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER))
    site = next(s for s in spec.sites if s.name == "attachment_site")
    site.attach_body(gripper.body("base_mount"), "g_", "")
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
    except Exception:
        pass
    spec.body("base").pos = [0, 0, cfg.table_h]

    wb = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER

    wb.add_geom(type=box, pos=[0.30, 0.0, cfg.table_h - 0.02], size=[0.6, 0.6, 0.02],
                rgba=[0.42, 0.44, 0.48, 1])
    for sx in (-0.25, 0.85):
        for sy in (-0.55, 0.55):
            wb.add_geom(type=box, pos=[sx, sy, (cfg.table_h - 0.04) / 2],
                        size=[0.03, 0.03, (cfg.table_h - 0.04) / 2], rgba=[0.2, 0.21, 0.23, 1])

    # supply bin
    bx, by, bw, bl, bh, th = 0.46, -0.30, 0.16, 0.13, 0.045, 0.006
    for dx, dy, sxu, syu in [(bw, 0, th, bl), (-bw, 0, th, bl), (0, bl, bw, th), (0, -bl, bw, th)]:
        wb.add_geom(type=box, pos=[bx + dx, by + dy, cfg.table_h + bh], size=[sxu, syu, bh],
                    rgba=[0.85, 0.65, 0.15, 1])
    # pallet
    wb.add_geom(type=box, pos=[0.37, 0.30, cfg.table_h + 0.006], size=[0.13, 0.13, 0.006],
                rgba=[0.55, 0.40, 0.24, 1])

    # machine fixture between bin and pallet (the obstacle to route around).
    # Column height/width come from cfg so a harder scenario (see the RRT test
    # in test_palletizing_cell.py) can grow it tall/wide enough that the
    # lift-over heuristic's fixed candidate heights can't clear it; the cap and
    # lamp stay sized and stacked relative to the column so the defaults here
    # reproduce the original fixed geometry exactly.
    #
    # physics=True gives these three geoms a contact `margin` (see
    # cfg.physics_clearance_margin's docstring for why): MuJoCo reports a
    # contact -- and so Planner._collides flags a "collision" -- once another
    # geom comes within `margin` of the fixture, not only on actual touch.
    # `gap` is set equal to `margin` so this only widens *detection*; the
    # force-enforcement threshold (dist < margin - gap = 0) is unchanged, so
    # real physics stepping (render(physics=True)'s mj_step) still only feels
    # a genuine contact force at genuine (zero-clearance) contact -- the
    # margin affects routing, not dynamics. physics=False gets margin=gap=0
    # (MuJoCo's default), i.e. this is a no-op for kinematic-mode building.
    margin = cfg.physics_clearance_margin if physics else 0.0
    fx, fy = cfg.fixture_xy
    hh = cfg.fixture_half_h
    hw_x, hw_y = cfg.fixture_half_w
    wb.add_geom(type=box, name="fixture_col", pos=[fx, fy, cfg.table_h + hh],
                size=[hw_x, hw_y, hh], rgba=[0.30, 0.33, 0.40, 1], margin=margin, gap=margin)
    wb.add_geom(type=box, name="fixture_cap", pos=[fx, fy, cfg.table_h + 2 * hh + 0.01],
                size=[hw_x + 0.03, hw_y + 0.03, 0.02], rgba=[0.20, 0.22, 0.28, 1],
                margin=margin, gap=margin)
    wb.add_geom(type=cyl, name="fixture_lamp", pos=[fx, fy, cfg.table_h + 2 * hh + 0.05],
                size=[0.02, 0.02, 0], rgba=[0.15, 0.7, 0.9, 1], margin=margin, gap=margin)

    # a marker at the unreachable requested slot (stays unfilled -> rejected)
    ux, uy = cfg.requests[4][0]
    wb.add_geom(type=cyl, pos=[ux, uy, cfg.table_h + 0.002], size=[0.035, 0.002, 0],
                rgba=[0.85, 0.15, 0.12, 0.6], contype=0, conaffinity=0)

    for k, (x, y) in enumerate(cfg.bin):
        b = wb.add_body(name=f"part{k}", pos=[x, y, cfg.table_h + cfg.part])
        b.add_freejoint()
        b.add_geom(type=box, size=[cfg.part, cfg.part, cfg.part], rgba=[0.72, 0.76, 0.80, 1], mass=0.05)

    # industrial dressing: floor hazard border + controller cabinet + stack light
    hz = [0.90, 0.75, 0.10, 1]
    wb.add_geom(type=box, pos=[0.30, -0.72, 0.006], size=[0.78, 0.03, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[0.30, 0.72, 0.006], size=[0.78, 0.03, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[-0.50, 0.0, 0.006], size=[0.03, 0.72, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[1.10, 0.0, 0.006], size=[0.03, 0.72, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[-0.15, -0.9, 0.32], size=[0.14, 0.11, 0.32], rgba=[0.23, 0.26, 0.32, 1])
    for i, col in enumerate([[0.9, 0.15, 0.1, 1], [0.95, 0.8, 0.1, 1], [0.15, 0.8, 0.25, 1]]):
        wb.add_geom(type=cyl, pos=[-0.15, -0.9, 0.72 + i * 0.045], size=[0.028, 0.022, 0], rgba=col)

    if physics:
        for a in spec.actuators:
            if a.name in ARM_ACTUATORS:
                a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                a.biastype = mujoco.mjtBias.mjBIAS_NONE
                a.gainprm[0], a.gainprm[1], a.gainprm[2] = 1.0, 0.0, 0.0
                a.biasprm[:] = 0.0
                a.ctrlrange = a.forcerange     # ctrl is now torque, not a position setpoint
        # One inactive weld per part; render(physics=True) computes the
        # current relative pose and flips data.eq_active on grasp/release
        # (see build_cell's docstring). The data= values here are
        # placeholders -- overwritten before the constraint is ever
        # activated -- so identity is as good as any other guess.
        for k in range(cfg.n_parts):
            spec.add_equality(name=f"weld_part{k}", type=mujoco.mjtEq.mjEQ_WELD,
                              objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1="g_base_mount", name2=f"part{k}", active=0,
                              data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

    return spec.compile()


def gripper_open_closed(mujoco, model, arm_q, grip_adr, grip_act):
    g0 = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    data.ctrl[:6] = arm_q
    data.ctrl[grip_act] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    open_g = data.qpos[grip_adr].copy()
    data.ctrl[grip_act] = 200
    for _ in range(700):
        mujoco.mj_step(model, data)
    closed_g = data.qpos[grip_adr].copy()
    model.opt.gravity[:] = g0
    return open_g, closed_g


class Planner:
    """Builds the palletizing frame list with collision-aware routing and a
    reachability check. Uses MuJoCo only for collision queries during planning."""

    def __init__(self, mujoco, model, arm, gripper_len, cfg=DEFAULT_CONFIG):
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.arm = arm
        self.gl = gripper_len
        self.cfg = cfg
        self.lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
        self.rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
        self.part_adr = [free_joint_qadr(mujoco, model, f"part{k}") for k in range(cfg.n_parts)]
        self.fixture = [i for i in range(model.ngeom)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("fixture")]
        self.mover_bodies = set()
        for i in range(model.nbody):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if n in cfg.ur_links or n.startswith("g_") or n.startswith("part"):
                self.mover_bodies.add(i)

    def ik(self, xy, z, seed):
        return solve_ik(self.arm, tool_down_pose(xy[0], xy[1], z + self.gl), seed)

    def front_seed(self, xy):
        """cfg.ik_seed with the base joint pre-aimed at the target's bearing.

        cfg.ik_seed pins the base at 0 rad, which is not a neutral choice: a
        damped-least-squares solve started there can settle on the shoulder
        branch that reaches the target *backwards*, swinging the base ~180 deg
        so the arm comes over its own shoulder instead of simply turning to
        face the work. Because every later waypoint is seeded from the previous
        configuration for continuity, whichever branch the FIRST solve picks
        propagates through the entire plan -- the cell used to run with
        shoulder_pan between 49 and 209 deg for bin and pallet targets whose
        actual bearings are only -42 to +45 deg.

        Aiming the base at atan2(y, x) starts the solve in the half of joint
        space that faces the target, so it converges on the natural front
        posture. That also clears the pallet corner nearest the fixture, which
        collides at rest on the backward branch and is the sole reason
        ik_clear ever had to go hunting for another one.

        NOT wired into build_plan yet, and the reason is a cell-layout problem
        rather than a kinematics one. Facing the work means swinging the base
        through bearing 0, which is exactly where the fixture stands (r = 0.41,
        inside both the bin at r ~ 0.56 and the pallet at r ~ 0.45), so the
        forearm sweeps through the column: measured 14 RRT fallbacks, 4
        transfers with no route at all, and 51 colliding frames, against 0/0/0
        on the backward branch. The backward reach is ugly but it is what the
        current obstacle placement leaves room for. Shortening the fixture
        makes it worse (104 and 143 colliding frames at 0.20 m and 0.16 m,
        since the cheap lift-over then succeeds on lower paths that still clip
        it) and raising lift_heights changes nothing at all. Moving the
        fixture outward to x = 0.58 was the best of the sweep and still failed
        (4 no-route, 32 colliding). Making this usable needs the bin, pallet
        and fixture repositioned together so the obstacle sits off the sweep
        while still genuinely blocking the direct path."""
        seed = np.array(self.cfg.ik_seed, dtype=float)
        seed[0] = np.arctan2(xy[1], xy[0])
        return seed

    def ik_clear(self, xy, z, seed):
        """Like ik(), but for grasp/place poses whose joint solution becomes a
        fixed waypoint the four short bin/pallet legs actually rest at (not
        just pass through). A damped-least-squares solve seeded from wherever
        the arm happens to be coming from can converge onto an elbow/wrist
        branch that is itself, at rest, in self-collision with the fixture --
        e.g. the pallet corner nearest the fixture pulls a wide range of
        seeds into a backward-base-rotation branch whose forearm grazes the
        fixture cap even though the tool pose is nowhere near it. That is a
        goal-state collision, not a path one: Planner.move's lift-over/RRT
        escalation can route *around* an obstacle on the way to a given qb,
        but it can never rescue a qb that collides while standing still,
        since every strategy still has to land on that exact joint target.
        So fix it at the source -- retry from a second, differently-branched
        seed (cfg.ik_seed_alt) and take it only if it both succeeds and
        clears the fixture at rest; otherwise fall back to the primary
        result unchanged, same as a plain ik() call.

        Known cosmetic cost: at the problem pallet corner the branch this
        lands on is ~5.12 rad from the incoming configuration, so the arm
        visibly contorts to place that part and unwinds again. Picking the
        NEAREST collision-free branch instead (analytical_ik returns up to 8;
        the closest clear one there is 3.48 rad) was tried and reverted --
        changing qb changed which lift-over detours could land on it, and
        under the physics clearance margin some transfers then failed to
        route at all. The contortion is really a scene-geometry problem: that
        corner sits in the fixture's shadow (pallet x 0.408 vs fixture x
        0.41) and EVERY branch there is >= 3.39 rad from the working
        configuration, so it wants fixing in the cell layout rather than by
        swapping IK branches underneath the planner."""
        res = self.ik(xy, z, seed)
        if not res.success or not self._collides(res.q, -1):
            return res

        alt = self.ik(xy, z, self.cfg.ik_seed_alt)
        if alt.success and not self._collides(alt.q, -1):
            return alt
        return res

    def grasp_centre(self):
        return 0.5 * (self.d.xpos[self.lp] + self.d.xpos[self.rp])

    def _collides(self, q, carried):
        self.d.qpos[:6] = q
        self.mj.mj_forward(self.m, self.d)
        if carried >= 0:
            c = self.grasp_centre()
            self.d.qpos[self.part_adr[carried]:self.part_adr[carried] + 7] = [c[0], c[1], c[2], 1, 0, 0, 0]
            self.mj.mj_forward(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            f1, f2 = c.geom1 in self.fixture, c.geom2 in self.fixture
            if not (f1 or f2):
                continue
            other = c.geom2 if f1 else c.geom1
            if self.m.geom_bodyid[other] in self.mover_bodies:
                return True
        return False

    def _path_collides(self, path, carried):
        return any(self._collides(q, carried) for q in path)

    def _joint_seg_qqd(self, qa, qb):
        """Like _joint_seg, but also returns the per-sample qd joint_trajectory
        already computes -- physics-mode PD tracking needs the reference
        velocity too; _joint_seg (and everything collision-checking-only)
        keeps discarding it since it never needed it."""
        _, seg, qd = joint_trajectory(qa, qb, v_max=self.cfg.v_max, a_max=self.cfg.a_max,
                                      dt=self.cfg.dt)
        return list(seg), list(qd)

    def _joint_seg(self, qa, qb):
        seg, _qd = self._joint_seg_qqd(qa, qb)
        return seg

    @staticmethod
    def _decimate(path, max_step):
        """Thin a dense geometric path down to waypoints at most `max_step`
        (joint-space distance) apart, always keeping the first and last
        point and never dropping a point that's already farther than
        max_step from the last one kept.

        cartesian_line's output is ~50 IK samples spaced only to keep the
        *tool* on a straight Cartesian line -- each one is not a meaningful
        kinematic milestone. multi_waypoint_trajectory pauses at rest at
        every waypoint it's given (by design, see its docstring), so handing
        it all ~50 of those samples turns a 3-leg lift/traverse/lower detour
        into ~50 pointless micro stop-starts instead of a handful of pauses
        at the corners that actually matter. This keeps the geometric path
        close to what was collision-checked (it's a strict subset of the
        checked points) while giving the re-timing pass waypoints worth
        pausing at. RRT-Connect's own output is already this coarse after
        its shortcutting pass, so decimation is a no-op there.
        """
        path = [np.asarray(p, dtype=float) for p in path]
        if len(path) <= 2:
            return path
        kept = [path[0]]
        for q in path[1:-1]:
            if np.linalg.norm(q - kept[-1]) >= max_step:
                kept.append(q)
        kept.append(path[-1])
        return kept

    def _retime_qqd(self, waypoints, carried):
        """Like _retime, but also returns the per-sample reference qd.

        A detour's waypoints are samples of a path to FLOW along, not stops
        that matter, so this uses path_trajectory (one accel / cruise / decel
        over the whole route) rather than multi_waypoint_trajectory, which
        rests at every waypoint by design. Feeding a ~50-sample Cartesian
        detour to the latter produced a stop-start lurch at every kept
        waypoint -- measured at 137 mid-move near-stops across the default
        plan, with 41% of moving time spent below a quarter of v_max.

        This also removes the reason _decimate existed. Decimating traded
        geometric fidelity for fewer pauses, and the chords it cut between
        kept points could bow into obstacles the fine path avoided, which is
        why the old code needed a collision-checked backoff ladder.
        path_trajectory instead interpolates *between adjacent* input points,
        so its output lies on the already-checked polyline. The collision
        re-check stays as a cheap invariant, not a fallback ladder.
        """
        cfg = self.cfg
        pts = [np.asarray(w, dtype=float) for w in waypoints]
        if len(pts) < 2:
            return pts, [np.zeros_like(p) for p in pts]
        _, q, qd = path_trajectory(pts, v_max=cfg.v_max, a_max=cfg.a_max, dt=cfg.dt,
                                   corner_rest=cfg.corner_rest_angle)
        return list(q), list(qd)

    def _retime(self, waypoints, carried):
        """Re-parameterize a search's geometric waypoints (lift-over Cartesian
        detour or RRT-Connect route) through the real joint velocity/
        acceleration limits, instead of playing them back at however many
        interpolation/planning steps the search happened to produce as if
        that step count were meaningful timing.

        multi_waypoint_trajectory samples each of its chained trapezoidal
        segments at cfg.dt, same as _joint_seg's single segment does, so the
        render loop's `idx * cfg.dt` sim-time and the HUD's cycle-time math
        stay valid on the result without any further resampling.

        _decimate only drops points, so its output is a strict subset of an
        already collision-checked path -- but the straight joint-space chord
        multi_waypoint_trajectory draws between two kept points can still bow
        away from the original (checked) polyline enough to clip something
        the fine path narrowly missed. So the retimed result is re-verified
        here and, if that happens, backed off to a finer decimation and
        finally to the full, undecimated fine path -- which is exactly the
        path move() already proved collision-free, just resampled at cfg.dt
        -- as a guaranteed-safe last resort. Never hand back a path that
        hasn't actually been checked.
        """
        q, _qd = self._retime_qqd(waypoints, carried)
        return q

    def _move_impl(self, qa, qb, carried):
        """Collision-aware move, escalating through three strategies:

        1. the direct joint path (_joint_seg, already trapezoidal/time-
           limited) -- unchanged fast path for the common unblocked case;
        2. a lift-traverse-lower Cartesian detour over the fixture, tried at
           a handful of fixed clearance heights (cfg.lift_heights) -- cheap,
           and already resolves this cell's own fixture, so it's kept as the
           first fallback;
        3. RRT-Connect, a general sampling-based fallback for whatever those
           fixed clearance heights can't clear (a taller/wider obstacle, a
           different layout, ...) instead of silently giving up.

        Whichever strategy actually finds a collision-free path has its
        geometric waypoints re-timed through the shared joint limits
        (_retime) before being returned.

        Returns (path, qd, replanned, caption). `qd` is the reference
        velocity paired with `path` -- move()'s public wrapper drops it for
        callers that only need the geometric path; physics-mode execution
        (build_plan) needs it for PD tracking's derivative term, and this is
        where it's already computed (joint_trajectory /
        multi_waypoint_trajectory produce it alongside q for free), so it's
        threaded through here instead of being finite-differenced later from
        a q-only path.

        `caption` says which strategy produced the path -- including the
        case where none did and `path` is the still-colliding direct segment
        returned as a genuine last resort -- so the HUD (and any other
        caller) can tell "found a real collision-free route" from "gave up"
        instead of the two looking identical.
        """
        cfg = self.cfg
        seg, seg_qd = self._joint_seg_qqd(qa, qb)
        if not self._path_collides(seg, carried):
            return seg, seg_qd, False, ""

        pa, pb = self.arm.fk(qa)[:3, 3], self.arm.fk(qb)[:3, 3]
        for zc in cfg.lift_heights:
            Ta, Tb = tool_down_pose(pa[0], pa[1], zc), tool_down_pose(pb[0], pb[1], zc)
            p1, o1 = cartesian_line(self.arm, self.arm.fk(qa), Ta, qa, steps=14)
            if not o1:
                continue
            p2, o2 = cartesian_line(self.arm, Ta, Tb, p1[-1], steps=26)
            if not o2:
                continue
            p3, o3 = cartesian_line(self.arm, Tb, self.arm.fk(qb), p2[-1], steps=14)
            if not o3:
                continue
            path = list(p1) + list(p2[1:]) + list(p3[1:])
            # cartesian_line solves IK for the tool POSE at each sample, so its
            # last config need only reproduce fk(qb) -- it can sit on a
            # different IK branch than qb itself (same tool pose, elbow or
            # wrist flipped). The caller then carries on from qb, so a detour
            # that quietly ends somewhere else leaves a gap between this
            # segment and the next: measured as a single-frame 180 deg wrist
            # flip, a teleport no velocity limit was ever applied to. Reject
            # such a route rather than splicing a branch-change move onto the
            # end of it: that move is a large unplanned reconfiguration, and
            # forcing it through made previously-routable transfers fail
            # outright ("NO ROUTE FOUND") once the physics clearance margin
            # was active. Falling through to the next clearance height, or to
            # RRT (which plans in joint space and so lands exactly on qb), is
            # both simpler and strictly safer.
            # Distinguish the two cases by size: a numerical residual from the
            # DLS solve is a tiny fraction of a radian and is closed by ending
            # exactly on qb, whereas a different branch is radians away and the
            # route has to be rejected. (Demanding an exact match instead
            # rejected every detour, since cartesian_line's IK never reproduces
            # qb to machine precision, and pushed all 17 transfers onto RRT.)
            gap = float(np.linalg.norm(np.asarray(path[-1], dtype=float)
                                       - np.asarray(qb, dtype=float)))
            if gap > cfg.branch_gap_max:
                continue
            if gap > 1e-9:
                path = path + [np.asarray(qb, dtype=float)]
            if not self._path_collides(path, carried):
                q, qd = self._retime_qqd(path, carried)
                return q, qd, True, "COLLISION AVOIDED  re-routing"

        # The lift-over heuristic's fixed candidate heights all failed --
        # fall back to a general sampling-based search instead of returning
        # the still-colliding direct segment. step_size=0.1 rad (cfg default)
        # keeps the segment-collision check (resolution = step_size/2 = 0.05
        # rad) fine enough not to tunnel through this cell's obstacles -- the
        # fixture's own half-extents are ~0.05-0.06 m, and a 0.05 rad step at
        # the arm's typical working radius (~0.4-0.8 m) sweeps a comparable
        # few-centimetre Cartesian arc -- while still being 10-20x smaller
        # than the ~1-2 rad joint deltas these blocked moves actually need to
        # cover, so it converges in well under a second rather than crawling
        # (see the RRT probe in this task's write-up: worst case ~0.8 s over
        # the fixture geometry actually in this cell).
        path = rrt_connect(qa, qb, lambda q: self._collides(q, carried), self.arm.joint_limits,
                           step_size=cfg.rrt_step_size, max_iters=cfg.rrt_max_iters,
                           goal_bias=cfg.rrt_goal_bias, seed=cfg.rrt_seed)
        if path is not None:
            q, qd = self._retime_qqd(path, carried)
            return q, qd, True, "RRT-CONNECT ROUTE"

        return seg, seg_qd, True, "NO ROUTE FOUND  collision not avoided"    # genuine last resort

    def move(self, qa, qb, carried):
        """Collision-aware move; see _move_impl for the full escalation
        strategy. Thin wrapper that drops the reference qd _move_impl also
        computes -- kept as the public entry point (path, replanned, caption)
        since most callers, and the existing tests exercising it directly,
        only need the geometric path."""
        q, _qd, replanned, caption = self._move_impl(qa, qb, carried)
        return q, replanned, caption


def build_plan(planner):
    cfg = planner.cfg
    # [q, grip_frac, carried_idx, snap(layer,xy) or None, caption, replans,
    #  rejected, qd] -- qd is the reference joint velocity paired with q
    # (zero for the stationary grasp/release/reject holds); physics-mode
    # execution (render(physics=True)) needs it for PD tracking's derivative
    # term, the kinematic teleport path (render(physics=False)) ignores it.
    frames = []
    replans = rejected = placed = rrt_routes = failed_routes = 0
    max_err = 0.0
    home_q = planner.ik(cfg.home_xy, cfg.home_z, cfg.ik_seed).q   # ready pose
    # NOTE: seeding this with planner.front_seed(cfg.home_xy) instead gives the
    # natural front-facing posture the whole plan inherits -- but the current
    # cell layout cannot be routed that way. See front_seed's docstring.
    q = home_q
    part_i = 0
    zero_qd = np.zeros(6)

    def emit(seq, qd_seq, gf, carried, caption):
        for qs, qds in zip(seq, qd_seq):
            frames.append([qs, gf, carried, None, caption, replans, rejected, qds])

    def tally(caption):
        # Planner.move's caption tells us which escalation tier actually
        # fired; count RRT-Connect engagements and genuine give-ups
        # separately from the cheap lift-over re-routes already in `replans`
        # so a caller (or a test) can tell them apart without parsing HUD text.
        nonlocal rrt_routes, failed_routes
        if caption == "RRT-CONNECT ROUTE":
            rrt_routes += 1
        elif caption.startswith("NO ROUTE FOUND"):
            failed_routes += 1

    for xy, layer in cfg.requests:
        # reachability validator: workspace-radius guard, then a real IK solve
        reachable = np.hypot(xy[0], xy[1]) <= cfg.reach_max
        place = planner.ik(xy, place_grasp_z(cfg, layer), q) if reachable else None
        if not reachable or not place.success or place.position_error > 5e-3:
            rejected += 1
            for _ in range(26):
                frames.append([q, 0.0, -1, None,
                               f"SLOT REJECTED  unreachable  ({xy[0]:.2f}, {xy[1]:.2f})",
                               replans, rejected, zero_qd])
            continue

        above_bin = planner.ik(cfg.bin[part_i], cfg.z_safe, q).q
        at_bin = planner.ik_clear(cfg.bin[part_i], cfg.z_pick, above_bin)
        above_pal = planner.ik(xy, cfg.z_safe, place.q).q
        max_err = max(max_err, at_bin.position_error, place.position_error)

        # approach the bin (may cross the fixture coming back from the pallet)
        seq, qd, rep, cap = planner._move_impl(q, above_bin, -1)
        replans += rep
        tally(cap)
        emit(seq, qd, 0.0, -1, cap)
        seq, qd, rep, cap = planner._move_impl(above_bin, at_bin.q, -1)
        replans += rep
        tally(cap)
        emit(seq[1:], qd[1:], 0.0, -1, cap)
        for k in range(10):                                   # grasp
            frames.append([at_bin.q, k / 9, part_i, None, "", replans, rejected, zero_qd])
        seq, qd, rep, cap = planner._move_impl(at_bin.q, above_bin, part_i)
        replans += rep
        tally(cap)
        emit(seq[1:], qd[1:], 1.0, part_i, cap)

        # collision-aware traverse to the pallet, carrying the part
        seq, qd, rep, cap = planner._move_impl(above_bin, above_pal, part_i)
        replans += rep
        tally(cap)
        emit(seq[1:], qd[1:], 1.0, part_i, cap)

        at_pal = planner.ik_clear(xy, place_grasp_z(cfg, layer), above_pal)
        seq, qd, rep, cap = planner._move_impl(above_pal, at_pal.q, part_i)
        replans += rep
        tally(cap)
        emit(seq[1:], qd[1:], 1.0, part_i, cap)
        for k in range(10):                                   # release
            frames.append([at_pal.q, 1 - k / 9, part_i, None, "", replans, rejected, zero_qd])
        frames[-1][3] = (part_i, xy, layer)                   # snap part to its cell
        placed += 1
        seq, qd, rep, cap = planner._move_impl(at_pal.q, above_pal, -1)
        replans += rep
        tally(cap)
        emit(seq[1:], qd[1:], 0.0, -1, cap)
        q = above_pal
        part_i += 1

    seq, qd, rep, cap = planner._move_impl(q, home_q, -1)
    replans += rep
    tally(cap)
    emit(seq[1:], qd[1:], 0.0, -1, cap)
    return frames, dict(replans=replans, rejected=rejected, placed=placed,
                        rrt_routes=rrt_routes, failed_routes=failed_routes,
                        accuracy_mm=max_err * 1000, total=len(cfg.requests))


def draw_hud(frame, placed, total_slots, sim_t, replans, rejected, accuracy_mm, caption,
             n_parts):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        title_f = ImageFont.truetype("arial.ttf", 26)
        f = ImageFont.truetype("consola.ttf", 20)
        cap_f = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        title_f = f = cap_f = ImageFont.load_default()
    cycle = sim_t / placed if placed else 0.0
    rate = placed / (sim_t / 60) if sim_t > 0 else 0.0
    dr.rectangle([18, 18, 366, 244], fill=(12, 16, 24, 195))
    dr.text((32, 28), "UR5e Palletizing Cell", font=title_f, fill=(120, 200, 255, 255))
    rows = [
        f"Parts placed   {placed}/{n_parts}",
        f"Cycle time     {cycle:4.1f} s/part",
        f"Throughput     {rate:4.1f} parts/min",
        f"Re-plans       {replans}  (collision avoided)",
        f"Rejected       {rejected}  (unreachable)",
        f"Placement err  <={accuracy_mm:.2f} mm",
        f"Sim time       {sim_t:5.1f} s",
    ]
    for j, line in enumerate(rows):
        dr.text((32, 66 + j * 25), line, font=f, fill=(220, 228, 236, 255))
    if caption:
        # Red: rejected slot or a genuine give-up (collision NOT avoided).
        # Cyan: RRT-Connect had to engage (the lift-over heuristic failed).
        # Amber: the cheap lift-over heuristic handled it.
        if "REJECT" in caption or caption.startswith("NO ROUTE FOUND"):
            colour = (240, 90, 80, 255)
        elif caption.startswith("RRT-CONNECT"):
            colour = (90, 210, 250, 255)
        else:
            colour = (250, 200, 70, 255)
        w = img.width
        dr.rectangle([w / 2 - 230, img.height - 60, w / 2 + 230, img.height - 22], fill=(10, 12, 18, 190))
        dr.text((w / 2 - 215, img.height - 54), caption, font=cap_f, fill=colour)
    return np.asarray(img)


def _weld_relpose(mujoco, data, body1, body2):
    """Relative pose of body2 in body1's frame, in the (pos, quat) layout
    MuJoCo's compiled weld eq_data expects at data[3:6]/data[6:10] (see
    build_cell's weld setup and the mjEQ_WELD data layout this mirrors:
    anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1))."""
    p1, q1 = data.xpos[body1], data.xquat[body1]
    p2, q2 = data.xpos[body2], data.xquat[body2]
    q1_inv = np.zeros(4)
    mujoco.mju_negQuat(q1_inv, q1)
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, p2 - p1, q1_inv)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, q1_inv, q2)
    return relpos, relquat


def render(mujoco, model, planner, frames, meta, height, width, render_stride=4,
          physics=False, track_err_out=None, grasp_gap_out=None, fixture_contact_out=None,
          data=None):
    """Replay a build_plan() frame list into rendered HUD images.

    physics=False (default): the original kinematic teleport path -- every
    frame's `q` is written straight to data.qpos (no dynamics), and a carried
    part's pose is a scripted snap to grasp_centre() / the target pallet cell.
    Unchanged behaviour.

    physics=True: each frame's `(q, qd)` is tracked by actually stepping
    physics (PDGravityCompController over build_cell(physics=True)'s torque
    actuators, zero-order-held for cfg.dt / model.opt.timestep substeps, same
    pattern as apps/physics_control.track_trajectory) instead of teleporting.
    A carried part is pinned to the gripper by a real weld equality
    constraint (armed with the part's current relative pose to `g_base_mount`
    the instant the gripper reports fully closed, released the instant it
    starts opening again -- see the should_hold comment below for why the
    threshold is "fully closed", not "past half-way") instead of a scripted
    position snap, so it's actually carried by contact/constraint forces
    under real dynamics, not just visually glued to a computed point every
    frame. Requires a model
    built with build_cell(physics=True) (torque actuators + per-part weld
    equalities); using a physics=False model here will silently mistreat
    written torques as position setpoints (see build_cell's docstring).

    track_err_out, if given a list, gets one |q_achieved - q_ref| array
    (shape (6,)) appended per frame in physics mode -- lets a caller verify
    tracking error stayed bounded without re-deriving it from the images.

    grasp_gap_out, if given a list, gets the distance between the held
    part's centre and grasp_centre() appended every physics-mode frame a
    part is actually welded (held >= 0) -- if the weld is doing its job this
    stays essentially constant (rigid attachment), not just "close"; a
    growing or erratic gap would mean the part is slipping or the constraint
    isn't actually engaged.

    fixture_contact_out, if given a list, gets one (frame_idx, geom1_name,
    geom2_name) tuple appended per physics-mode substep in which the
    *actually simulated* arm/gripper/carried-part geometry genuinely touches
    (contact.dist < 0, real penetration -- not just inside
    cfg.physics_clearance_margin's contact-detection buffer, see
    build_cell(physics=True)) one of planner.fixture's geoms. This is the
    safety property cfg.physics_clearance_margin exists for: unlike
    track_err_out, which only shows tracking error stayed bounded, this
    proves the real simulated geometry never actually clipped the fixture.
    Empty in physics=False mode (nothing is simulated to clip anything).

    data, if given an MjData, is stepped/written in place instead of a fresh
    one created internally -- lets a caller inspect the final simulated
    state (e.g. where a part actually ended up) after render() returns.
    """
    cfg = planner.cfg
    data = mujoco.MjData(model) if data is None else data
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    grip_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in cfg.grip_joints]
    grip_dof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in cfg.grip_joints]
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    part_adr = [free_joint_qadr(mujoco, model, f"part{k}") for k in range(cfg.n_parts)]
    part_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"part{k}")
                for k in range(cfg.n_parts)]

    def grasp_centre():
        return 0.5 * (data.xpos[lp] + data.xpos[rp])

    open_g, closed_g = gripper_open_closed(mujoco, model, frames[0][0], grip_adr, grip_act)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.34, 0.02, cfg.table_h + 0.06]
    cam.distance = 1.8
    cam.azimuth = 128
    cam.elevation = -22

    r = mujoco.Renderer(model, height, width)
    part_state = [np.array([x, y, cfg.table_h + cfg.part, 1, 0, 0, 0.0]) for (x, y) in cfg.bin]

    controller = weld_ids = gripper_body = n_sub = None
    held = -1                          # part index currently welded, -1 = none
    if physics:
        from physics_control import PDGravityCompController
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        ctrl_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
        controller = PDGravityCompController(mujoco, model, data, joint_ids,
                                             PHYSICS_KP, PHYSICS_KD, ctrl_ids)
        weld_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_part{k}")
                    for k in range(cfg.n_parts)]
        gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_base_mount")
        n_sub = max(1, round(cfg.dt / model.opt.timestep))
        data.qpos[:6] = frames[0][0]
        for k in range(cfg.n_parts):
            data.qpos[part_adr[k]:part_adr[k] + 7] = part_state[k]
        mujoco.mj_forward(model, data)

    placed, imgs = 0, []
    for idx, (qk, gf, carry, snap, caption, replans, rejected, qdk) in enumerate(frames):
        if physics:
            for _ in range(n_sub):
                # gripper fingers stay kinematically driven (not physically
                # actuated) in physics mode too -- only the arm's own
                # dynamics and the part's grasp are real here; see render()'s
                # docstring and this task's write-up for the scope call.
                data.qpos[grip_adr] = open_g + (closed_g - open_g) * gf
                data.qvel[grip_dof] = 0.0
                controller.step(qk, qdk)
                if fixture_contact_out is not None:
                    for i in range(data.ncon):
                        c = data.contact[i]
                        f1, f2 = c.geom1 in planner.fixture, c.geom2 in planner.fixture
                        if not (f1 or f2) or c.dist >= 0:
                            continue    # c.dist >= 0: still inside the detection margin, not a real touch
                        other = c.geom2 if f1 else c.geom1
                        if model.geom_bodyid[other] in planner.mover_bodies:
                            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                            fixture_contact_out.append((idx, n1, n2))
            if track_err_out is not None:
                track_err_out.append(np.abs(controller.q - np.asarray(qk)))

            # Engage only once the gripper is fully closed (gf==1.0), not
            # partway through closing: the weld captures the part's *actual*
            # current pose relative to g_base_mount, not a recomputed
            # grasp_centre() (unlike the kinematic path's per-frame snap), so
            # capturing mid-close -- while the pads (and so grasp_centre())
            # are still moving -- bakes in whatever off-centre offset exists
            # at that instant and carries it through to placement unchanged.
            # Symmetrically, release the instant the fingers start opening
            # (gf just below 1.0) rather than waiting for them to be half
            # open, since nothing about a mid-open gf value is more
            # physically meaningful for "let go" than any other.
            should_hold = carry if (carry >= 0 and gf >= 1.0 - 1e-9) else -1
            if should_hold != held:
                if held >= 0:
                    data.eq_active[weld_ids[held]] = False
                if should_hold >= 0:
                    relpos, relquat = _weld_relpose(mujoco, data, gripper_body,
                                                    part_body[should_hold])
                    eqid = weld_ids[should_hold]
                    model.eq_data[eqid, 3:6] = relpos
                    model.eq_data[eqid, 6:10] = relquat
                    data.eq_active[eqid] = True
                held = should_hold
            if grasp_gap_out is not None and held >= 0:
                mujoco.mj_forward(model, data)
                grasp_gap_out.append(float(np.linalg.norm(data.xpos[part_body[held]] - grasp_centre())))
        else:
            data.qpos[:6] = qk
            data.qpos[grip_adr] = open_g + (closed_g - open_g) * gf
            if carry >= 0 and gf > 0.5:
                mujoco.mj_forward(model, data)
                c = grasp_centre()
                part_state[carry] = np.array([c[0], c[1], c[2], 1, 0, 0, 0.0])

        if snap is not None:
            placed += 1
            if not physics:
                # kinematic mode: nothing physically holds/drops the part, so
                # place it by scripted snap, same as the grasp above. Physics
                # mode instead lets the weld release and gravity/contact
                # settle the part for real (see the should_hold handling).
                pi, xy, layer = snap
                part_state[pi] = np.array([xy[0], xy[1], cfg.table_h + rest_centre_z(cfg, layer),
                                           1, 0, 0, 0.0])
        if idx % render_stride and idx != len(frames) - 1:
            continue                       # advance state every frame, render every Nth
        if not physics:
            for k in range(cfg.n_parts):
                data.qpos[part_adr[k]:part_adr[k] + 7] = part_state[k]
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        imgs.append(draw_hud(r.render(), placed, meta["total"], idx * cfg.dt,
                             replans, rejected, meta["accuracy_mm"], caption,
                             n_parts=cfg.n_parts))
    return imgs


def save_gif(frames, path, stride=1, fps=20, scale=0.38, colors=72):
    """Write the README hero GIF.

    GIF is the only format GitHub plays inline in a README, and it is a poor
    fit for this: no interframe motion compensation, so cost scales almost
    linearly with frame count. That forces a three-way trade between smoothness
    (frames), size, and resolution -- the MP4 (save_mp4) has none of these
    limits and is the better artefact wherever a real player is available.

    stride=1 keeps every rendered frame, since sparse sampling is what made an
    earlier version strobe; the resolution is spent down instead (scale 0.38)
    to pay for those frames. disposal=1 leaves each frame in place rather than
    restoring the background first, which lets the encoder emit only the
    changed region -- worth ~18% here, measured, because the camera is fixed
    and much of the scene never moves.
    """
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    picked = frames[::stride]
    w = int(frames[0].shape[1] * scale)
    h = int(frames[0].shape[0] * scale)
    pil = [Image.fromarray(f).resize((w, h), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=colors) for f in picked]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=True, disposal=1)
    print(f"saved {path}  ({len(pil)} frames, {w}x{h}, {path.stat().st_size / 1e6:.1f} MB)")


def save_mp4(frames, path, fps=30):
    """Write a web-ready H.264 MP4 (yuv420p for browser compatibility)."""
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rate-control by CRF, not imageio's `quality` (which maps to -qscale and
    # ignores how compressible the content is). The camera is fixed and most
    # of every frame is static, so constant-quality encoding gets the same
    # picture in a fraction of the bits: the default plan came out at 22 MB
    # under quality=8 (8.5 Mbit/s) versus 8.7 MB at CRF 18, which is visually
    # transparent for this content. +faststart moves the index to the front so
    # it starts playing before the whole file has downloaded.
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=8,
                     output_params=["-crf", "18", "-preset", "slow",
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    print(f"saved {path}  ({len(frames)} frames @ {fps} fps, {path.stat().st_size / 1e6:.1f} MB)")


def _limit_to_n_parts(frames, meta, n_parts):
    """Debug slicer for --parts: cut the frame list right after the n_parts-th
    part is placed (a snap event, frame[3] is not None) instead of playing
    out the full plan. Physics mode steps real dynamics every substep, so the
    full 8-part plan is much slower than kinematic playback; this lets a
    physics run demonstrate a complete pick-and-place cycle (tracking error,
    grasp, release) without paying for the rest."""
    cut = None
    seen = 0
    for i, f in enumerate(frames):
        if f[3] is not None:
            seen += 1
            if seen == n_parts:
                cut = i
                break
    if cut is None:
        return frames, meta         # fewer than n_parts ever get placed; nothing to cut
    meta = dict(meta, placed=n_parts, total=n_parts)
    return frames[:cut + 1], meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write docs/palletizing_cell.gif")
    parser.add_argument("--mp4", action="store_true", help="write docs/palletizing_cell.mp4 (web-ready)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--physics", action="store_true",
                        help="physics-tracked execution (PD + gravity-comp torque control over "
                             "build_cell(physics=True)'s torque actuators, real MuJoCo dynamics "
                             "stepped every substep, real weld-constraint grasp) instead of the "
                             "default kinematic teleport playback")
    parser.add_argument("--parts", type=int, default=None,
                        help="debug limiter: stop after this many parts are placed instead of "
                             "the full plan (physics mode is much slower than kinematic playback "
                             "since it steps real dynamics every substep)")
    args = parser.parse_args()

    import mujoco

    model = build_cell(mujoco, physics=args.physics)
    arm = SerialArm.ur5e()

    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*DEFAULT_CONFIG.bin[0], 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    planner = Planner(mujoco, model, arm, gripper_len)
    frames, meta = build_plan(planner)
    if args.parts is not None:
        frames, meta = _limit_to_n_parts(frames, meta, args.parts)
    print(f"plan: {len(frames)} frames | placed {meta['placed']}/{DEFAULT_CONFIG.n_parts} | "
          f"re-plans {meta['replans']} (rrt {meta['rrt_routes']}, failed {meta['failed_routes']}) | "
          f"rejected {meta['rejected']} | max IK err {meta['accuracy_mm']:.3f} mm"
          + ("  [physics]" if args.physics else ""))
    # Displayed-frame sampling. The motion between two DISPLAYED frames is
    # v_max * stride * dt, so the re-timing limits -- not the frame count --
    # set how smooth the output can possibly look. At cfg.v_max=2.6 rad/s and
    # dt=0.04, stride 8 (the GIF path: render_stride 4 x save_gif stride 2)
    # allows 0.83 rad = 48 deg of joint travel per displayed frame, which
    # strobes badly on the fast segments. Measured on the default plan:
    # end-effector travel per displayed frame is a median 64 mm / max 407 mm
    # at stride 8, versus 16 mm / 127 mm at stride 2 (joint travel likewise
    # 8.8 deg median / 47.7 deg max, versus 2.4 / 11.9). The motion is bursty
    # (67% of moving frames run below 30% of v_max, 6% saturate it), so a
    # sparse fixed stride samples the bursts far too coarsely.
    # MP4 can afford the frames that fixes this; GIF cannot (stride 2 over the
    # full plan is ~1230 frames, ~50 MB), so the GIF stays sparse by necessity.
    render_stride = 1 if args.mp4 else 4
    track_err = [] if args.physics else None
    imgs = render(mujoco, model, planner, frames, meta, args.height, args.width, render_stride,
                 physics=args.physics, track_err_out=track_err)
    if track_err:
        track_err = np.array(track_err)
        print(f"physics tracking error (rad): mean {track_err.mean():.4f}  "
              f"max {track_err.max():.4f}  final {track_err[-1].max():.4f}")

    if args.mp4:
        # 60 fps over render_stride=2 replays the 98 s cycle at 4.8x in ~21 s.
        # Playback speed is stride * dt * fps; smoothness is set by stride
        # alone, so the speed-up here costs nothing in strobing.
        save_mp4(imgs, ROOT / "docs" / "palletizing_cell.mp4", fps=60)
    if args.save:
        save_gif(imgs, ROOT / "docs" / "palletizing_cell.gif")
    if not (args.mp4 or args.save):
        from PIL import Image
        out = ROOT / "docs" / "palletizing_cell_still.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(imgs[len(imgs) // 2]).save(out)
        print(f"wrote a still to {out}; pass --save to write the GIF")


if __name__ == "__main__":
    main()

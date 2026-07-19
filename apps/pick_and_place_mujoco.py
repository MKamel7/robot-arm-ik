"""Photoreal pick-and-place demo: the same armik inverse kinematics that drives
the matplotlib demo, rendered in the MuJoCo physics engine with a real UR5e and
a Robotiq 2F-85 parallel-jaw gripper grasping a box on a work table.

The kinematics is entirely armik's: `SerialArm.ur5e()` forward kinematics and the
damped-least-squares IK solve each waypoint pose, and `joint_trajectory` builds
the timed motion between them. MuJoCo only renders the result and animates the
gripper -- the joint angles it displays are the ones this library computed. The
UR5e FK is cross-validated against the MuJoCo Menagerie model (identity joint
mapping, tool position agrees to ~1 mm), which is why the arm reaches the box.

The gripper linkage's open and closed joint configurations are found once by
settling the physics, then replayed kinematically; the box follows the grasp
centre while the gripper holds it.

Run (requires the optional `sim` extras: mujoco, imageio):
    python apps/pick_and_place_mujoco.py             show/save nothing but a still
    python apps/pick_and_place_mujoco.py --save      write docs/pick_and_place_mujoco.gif

Assets: UR5e and Robotiq 2F-85 models from the MuJoCo Menagerie, vendored under
`assets/` with their licenses (see assets/ATTRIBUTION.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import SerialArm, solve_ik, joint_trajectory        # noqa: E402
from armik.rotations import matrix_from_rotvec                 # noqa: E402

UR5E_SCENE = ROOT / "assets" / "ur5e" / "scene.xml"
GRIPPER = ROOT / "assets" / "robotiq_2f85" / "2f85.xml"

TABLE_H = 0.50                       # arm mounts on a table of this height
PICK = np.array([0.40, -0.25])       # pick (x, y) in the arm base frame
PLACE = np.array([0.35, 0.30])       # place (x, y)
Z_CONTACT = 0.045                    # grasp-centre height when grasping the box
Z_APPROACH = 0.22                    # grasp-centre height for the approach
GRIP_JOINTS = [
    "g_right_driver_joint", "g_right_coupler_joint", "g_right_spring_link_joint",
    "g_right_follower_joint", "g_left_driver_joint", "g_left_coupler_joint",
    "g_left_spring_link_joint", "g_left_follower_joint",
]


def tool_down_pose(x, y, z):
    """Target pose at (x, y, z) with the tool pointing straight down."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_rotvec(np.array([0.0, np.pi, 0.0]))
    T[:3, 3] = [x, y, z]
    return T


def build_scene(mujoco):
    """Compose the UR5e + gripper + table + box + place pad into one MjModel."""
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER))
    site = next(s for s in spec.sites if s.name == "attachment_site")
    site.attach_body(gripper.body("base_mount"), "g_", "")

    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
    except Exception:
        pass

    spec.body("base").pos = [0, 0, TABLE_H]     # mount the arm on the table

    wb = spec.worldbody
    top, leg = [0.42, 0.44, 0.48, 1.0], [0.20, 0.21, 0.23, 1.0]
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.30, 0.0, TABLE_H - 0.02],
                size=[0.55, 0.55, 0.02], rgba=top)
    for sx in (-0.22, 0.80):
        for sy in (-0.50, 0.50):
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[sx, sy, (TABLE_H - 0.04) / 2],
                        size=[0.03, 0.03, (TABLE_H - 0.04) / 2], rgba=leg)

    box = wb.add_body(name="box", pos=[PICK[0], PICK[1], TABLE_H + 0.022])
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.022, 0.022, 0.022],
                 rgba=[0.88, 0.28, 0.20, 1.0], mass=0.05)
    pad = wb.add_body(name="place_pad", pos=[PLACE[0], PLACE[1], TABLE_H + 0.001])
    pad.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.045, 0.0015, 0.0],
                 rgba=[0.25, 0.65, 0.38, 0.85], contype=0, conaffinity=0)

    wb.add_light(pos=[0.8, -0.8, TABLE_H + 1.2], dir=[-0.5, 0.5, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.4, 0.4, 0.4])
    wb.add_light(pos=[-0.4, 0.6, TABLE_H + 1.0], dir=[0.3, -0.4, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.25, 0.25, 0.3])

    return spec.compile()


def gripper_open_closed(mujoco, model, arm_q, grip_adr, grip_act):
    """Find the gripper's open and near-closed joint vectors by settling physics."""
    g0 = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    data.qpos[:6] = arm_q
    data.ctrl[:6] = arm_q
    data.ctrl[grip_act] = 0
    for _ in range(600):
        mujoco.mj_step(model, data)
    open_g = data.qpos[grip_adr].copy()
    data.ctrl[grip_act] = 200               # near-closed, not crushing
    for _ in range(800):
        mujoco.mj_step(model, data)
    closed_g = data.qpos[grip_adr].copy()
    model.opt.gravity[:] = g0
    return open_g, closed_g


def build_plan(arm, gripper_len):
    """Waypoint plan -> (q per frame, gripper close-fraction per frame, holding)."""
    def ik(xy, z_centre, seed):
        return solve_ik(arm, tool_down_pose(xy[0], xy[1], z_centre + gripper_len), seed)

    q_home = np.zeros(6)
    legs = [
        ("above pick", PICK, Z_APPROACH, False),
        ("reach down", PICK, Z_CONTACT, False),
        ("lift", PICK, Z_APPROACH, True),
        ("carry", PLACE, Z_APPROACH, True),
        ("lower", PLACE, Z_CONTACT, True),
        ("lift away", PLACE, Z_APPROACH, False),
        ("home", None, None, False),
    ]
    q, wpts, holds, names = q_home, [q_home], [], []
    for name, xy, z, hold in legs:
        res = None if xy is None else ik(xy, z, q)
        if res is not None and not res.success:
            print(f"warning: IK did not fully converge for '{name}' "
                  f"(pos err {res.position_error:.2e} m)")
        qn = q_home if xy is None else res.q
        wpts.append(qn)
        holds.append(hold)
        names.append(name)
        q = qn

    fq, fg, fh = [], [], []

    def dwell(qc, n, a, b, hold):
        for k in range(n):
            fq.append(qc)
            fg.append(a + (b - a) * k / max(1, n - 1))
            fh.append(hold)

    close = 0.0
    for i in range(len(wpts) - 1):
        _, seg, _ = joint_trajectory(wpts[i], wpts[i + 1], v_max=1.5, a_max=3.0, dt=0.04)
        if i > 0:
            seg = seg[1:]
        for qs in seg:
            fq.append(qs)
            fg.append(close)
            fh.append(holds[i])
        if names[i] == "reach down":
            dwell(wpts[i + 1], 12, 0.0, 1.0, False)   # close on the object
            close = 1.0
        if names[i] == "lower":
            dwell(wpts[i + 1], 12, 1.0, 0.0, True)     # open to release
            close = 0.0
    return fq, fg, fh


def render(mujoco, model, fq, fg, fh, height, width):
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    # address the box's free joint via its body (add_freejoint leaves it unnamed)
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_adr = model.jnt_qposadr[model.body_jntadr[box_bid]]
    grip_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in GRIP_JOINTS]
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")

    def grasp_centre():
        return 0.5 * (data.xpos[lp] + data.xpos[rp])

    open_g, closed_g = gripper_open_closed(mujoco, model, fq[0], grip_adr, grip_act)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.32, 0.0, TABLE_H + 0.08]
    cam.distance = 1.7
    cam.azimuth = 133
    cam.elevation = -20

    r = mujoco.Renderer(model, height, width)
    frames = []
    box_q = np.array([PICK[0], PICK[1], TABLE_H + 0.022, 1, 0, 0, 0.0])
    for qk, gf, hold in zip(fq, fg, fh):
        data.qpos[:6] = qk
        data.qpos[grip_adr] = open_g + (closed_g - open_g) * gf
        if hold:
            mujoco.mj_forward(model, data)
            c = grasp_centre()
            box_q = np.array([c[0], c[1], c[2], 1, 0, 0, 0.0])
        data.qpos[box_adr:box_adr + 7] = box_q
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        frames.append(r.render())
    return frames


def save_gif(frames, path, stride=3, fps=16, scale=0.6, colors=128):
    """Write a size-optimised GIF: subsample frames, downscale, quantise palette."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    picked = frames[::stride]
    w = int(frames[0].shape[1] * scale)
    h = int(frames[0].shape[0] * scale)
    pil = [Image.fromarray(f).resize((w, h), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=colors) for f in picked]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print(f"saved {path}  ({len(pil)} frames, {w}x{h}, {path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write docs/pick_and_place_mujoco.gif")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import mujoco

    model = build_scene(mujoco)
    arm = SerialArm.ur5e()

    # gripper length: flange to grasp centre at a downward pose
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*PICK, 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    fq, fg, fh = build_plan(arm, gripper_len)
    print(f"planned {len(fq)} frames; UR5e IK drives every one")
    frames = render(mujoco, model, fq, fg, fh, args.height, args.width)

    if args.save:
        save_gif(frames, ROOT / "docs" / "pick_and_place_mujoco.gif")
    else:
        from PIL import Image
        out = ROOT / "docs" / "pick_and_place_mujoco_still.png"
        Image.fromarray(frames[len(frames) // 2]).save(out)
        print(f"wrote a still to {out}; pass --save to write the GIF")


if __name__ == "__main__":
    main()

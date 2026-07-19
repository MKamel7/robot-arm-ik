"""Palletizing cell: a UR5e transfers parts from a supply bin into a multi-layer
pallet grid, driven entirely by this library's inverse kinematics, rendered in
MuJoCo. This is the industrial version of the pick-and-place demo.

It goes beyond a happy-path animation and shows three things a real automation
cell has to do:

* Collision-aware routing. A machine fixture stands between the bin and the
  pallet. Before each transfer the planner checks the direct joint-space path
  for collisions (MuJoCo contact queries); when it is blocked it re-routes with a
  lift-traverse-lower Cartesian path over the obstacle. The heads-up display
  counts how many moves were re-planned.
* Multi-layer palletizing. Parts are stacked into a 2x2x2 pallet, so the place
  height changes per layer.
* Failure handling. One requested pallet slot is deliberately outside the arm's
  reachable workspace. A reachability check catches it, the slot is rejected on
  screen, and the cell carries on with the reachable ones instead of faking it.

Every joint angle comes from armik: `SerialArm.ur5e()` forward kinematics, the
damped-least-squares IK (`solve_ik`) for each waypoint, `joint_trajectory` for
timed joint moves, and `cartesian_line` for the straight-line detour segments.
The reported placement accuracy is the IK position residual. MuJoCo renders the
cell and animates the Robotiq 2F-85 gripper.

Maps to real German factory-automation work (end-of-line palletizing, machine
tending, intralogistics): Krones, Siemens, KUKA, BMW plant logistics, and
warehouse-robotics companies.

Run (needs the optional `sim` extras: mujoco, imageio, pillow):
    python apps/palletizing_cell.py             write a still
    python apps/palletizing_cell.py --save      write docs/palletizing_cell.gif
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import SerialArm, solve_ik, joint_trajectory, cartesian_line   # noqa: E402
from armik.rotations import matrix_from_rotvec                            # noqa: E402

UR5E_SCENE = ROOT / "assets" / "ur5e" / "scene.xml"
GRIPPER = ROOT / "assets" / "robotiq_2f85" / "2f85.xml"

TABLE_H = 0.50
PART = 0.02                          # part half-size (0.04 m cube)
Z_PICK = 0.038                       # grasp-centre height at a part on the table
Z_SAFE = 0.26                        # local lift/lower height
DT = 0.04
LAYER_DZ = 2 * PART                  # vertical pitch between pallet layers
GRIP_JOINTS = [
    "g_right_driver_joint", "g_right_coupler_joint", "g_right_spring_link_joint",
    "g_right_follower_joint", "g_left_driver_joint", "g_left_coupler_joint",
    "g_left_spring_link_joint", "g_left_follower_joint",
]
UR_LINKS = {"base", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link"}
FIXTURE_XY = (0.41, 0.0)             # machine fixture, between bin and pallet


def grid(cx, cy, nx, ny, sx, sy):
    return [(cx + (i - (nx - 1) / 2) * sx, cy + (j - (ny - 1) / 2) * sy)
            for j in range(ny) for i in range(nx)]


BIN = grid(0.46, -0.30, 4, 2, 0.058, 0.075)            # 8 supply cells
PALLET_XY = grid(0.37, 0.30, 2, 2, 0.075, 0.075)       # 2x2 pallet footprint
# requested pallet slots: base layer, one UNREACHABLE slot, then the top layer
UNREACHABLE_XY = (0.90, 0.30)                           # outside the arm's workspace
REACH_MAX = 0.82                                        # UR5e workspace radius for a downward grasp
REQUESTS = ([(xy, 0) for xy in PALLET_XY]
            + [(UNREACHABLE_XY, 0)]
            + [(xy, 1) for xy in PALLET_XY])
N_PARTS = len(BIN)


def tool_down_pose(x, y, z):
    T = np.eye(4)
    T[:3, :3] = matrix_from_rotvec(np.array([0.0, np.pi, 0.0]))
    T[:3, 3] = [x, y, z]
    return T


def place_grasp_z(layer):
    return 0.05 + layer * LAYER_DZ


def rest_centre_z(layer):
    return 0.032 + layer * LAYER_DZ


def free_joint_qadr(mujoco, model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return model.jnt_qposadr[model.body_jntadr[bid]]


def build_cell(mujoco):
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER))
    site = next(s for s in spec.sites if s.name == "attachment_site")
    site.attach_body(gripper.body("base_mount"), "g_", "")
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
    except Exception:
        pass
    spec.body("base").pos = [0, 0, TABLE_H]

    wb = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER

    wb.add_geom(type=box, pos=[0.30, 0.0, TABLE_H - 0.02], size=[0.6, 0.6, 0.02],
                rgba=[0.42, 0.44, 0.48, 1])
    for sx in (-0.25, 0.85):
        for sy in (-0.55, 0.55):
            wb.add_geom(type=box, pos=[sx, sy, (TABLE_H - 0.04) / 2],
                        size=[0.03, 0.03, (TABLE_H - 0.04) / 2], rgba=[0.2, 0.21, 0.23, 1])

    # supply bin
    bx, by, bw, bl, bh, th = 0.46, -0.30, 0.16, 0.13, 0.045, 0.006
    for dx, dy, sxu, syu in [(bw, 0, th, bl), (-bw, 0, th, bl), (0, bl, bw, th), (0, -bl, bw, th)]:
        wb.add_geom(type=box, pos=[bx + dx, by + dy, TABLE_H + bh], size=[sxu, syu, bh],
                    rgba=[0.85, 0.65, 0.15, 1])
    # pallet
    wb.add_geom(type=box, pos=[0.37, 0.30, TABLE_H + 0.006], size=[0.13, 0.13, 0.006],
                rgba=[0.55, 0.40, 0.24, 1])

    # machine fixture between bin and pallet (the obstacle to route around)
    fx, fy = FIXTURE_XY
    wb.add_geom(type=box, name="fixture_col", pos=[fx, fy, TABLE_H + 0.15],
                size=[0.05, 0.06, 0.15], rgba=[0.30, 0.33, 0.40, 1])
    wb.add_geom(type=box, name="fixture_cap", pos=[fx, fy, TABLE_H + 0.31],
                size=[0.08, 0.09, 0.02], rgba=[0.20, 0.22, 0.28, 1])
    wb.add_geom(type=cyl, name="fixture_lamp", pos=[fx, fy, TABLE_H + 0.35],
                size=[0.02, 0.02, 0], rgba=[0.15, 0.7, 0.9, 1])

    # a marker at the unreachable requested slot (stays unfilled -> rejected)
    ux, uy = REQUESTS[4][0]
    wb.add_geom(type=cyl, pos=[ux, uy, TABLE_H + 0.002], size=[0.035, 0.002, 0],
                rgba=[0.85, 0.15, 0.12, 0.6], contype=0, conaffinity=0)

    for k, (x, y) in enumerate(BIN):
        b = wb.add_body(name=f"part{k}", pos=[x, y, TABLE_H + PART])
        b.add_freejoint()
        b.add_geom(type=box, size=[PART, PART, PART], rgba=[0.72, 0.76, 0.80, 1], mass=0.05)

    # industrial dressing: floor hazard border + controller cabinet + stack light
    hz = [0.90, 0.75, 0.10, 1]
    wb.add_geom(type=box, pos=[0.30, -0.72, 0.006], size=[0.78, 0.03, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[0.30, 0.72, 0.006], size=[0.78, 0.03, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[-0.50, 0.0, 0.006], size=[0.03, 0.72, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[1.10, 0.0, 0.006], size=[0.03, 0.72, 0.006], rgba=hz)
    wb.add_geom(type=box, pos=[-0.15, -0.9, 0.32], size=[0.14, 0.11, 0.32], rgba=[0.23, 0.26, 0.32, 1])
    for i, col in enumerate([[0.9, 0.15, 0.1, 1], [0.95, 0.8, 0.1, 1], [0.15, 0.8, 0.25, 1]]):
        wb.add_geom(type=cyl, pos=[-0.15, -0.9, 0.72 + i * 0.045], size=[0.028, 0.022, 0], rgba=col)

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

    def __init__(self, mujoco, model, arm, gripper_len):
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.arm = arm
        self.gl = gripper_len
        self.lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
        self.rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
        self.part_adr = [free_joint_qadr(mujoco, model, f"part{k}") for k in range(N_PARTS)]
        self.fixture = [i for i in range(model.ngeom)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("fixture")]
        self.mover_bodies = set()
        for i in range(model.nbody):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if n in UR_LINKS or n.startswith("g_") or n.startswith("part"):
                self.mover_bodies.add(i)

    def ik(self, xy, z, seed):
        return solve_ik(self.arm, tool_down_pose(xy[0], xy[1], z + self.gl), seed)

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

    def _joint_seg(self, qa, qb):
        _, seg, _ = joint_trajectory(qa, qb, v_max=2.6, a_max=6.5, dt=DT)
        return list(seg)

    def move(self, qa, qb, carried):
        """Collision-aware move: direct if clear, else lift-traverse-lower over
        the fixture. Returns (path, replanned)."""
        seg = self._joint_seg(qa, qb)
        if not self._path_collides(seg, carried):
            return seg, False
        pa, pb = self.arm.fk(qa)[:3, 3], self.arm.fk(qb)[:3, 3]
        for zc in (0.48, 0.54, 0.60, 0.66):
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
            if not self._path_collides(path, carried):
                return path, True
        return seg, True    # last resort


def build_plan(planner):
    frames = []            # [q, grip_frac, carried_idx, snap(layer,xy) or None, caption]
    replans = rejected = placed = 0
    max_err = 0.0
    q = np.zeros(6)
    part_i = 0

    def emit(seq, gf, carried, caption):
        for qs in seq:
            frames.append([qs, gf, carried, None, caption, replans, rejected])

    for xy, layer in REQUESTS:
        # reachability validator: workspace-radius guard, then a real IK solve
        reachable = np.hypot(xy[0], xy[1]) <= REACH_MAX
        place = planner.ik(xy, place_grasp_z(layer), q) if reachable else None
        if not reachable or not place.success or place.position_error > 5e-3:
            rejected += 1
            for _ in range(26):
                frames.append([q, 0.0, -1, None,
                               f"SLOT REJECTED  unreachable  ({xy[0]:.2f}, {xy[1]:.2f})",
                               replans, rejected])
            continue

        above_bin = planner.ik(BIN[part_i], Z_SAFE, q).q
        at_bin = planner.ik(BIN[part_i], Z_PICK, above_bin)
        above_pal = planner.ik(xy, Z_SAFE, place.q).q
        max_err = max(max_err, at_bin.position_error, place.position_error)

        # approach the bin (may cross the fixture coming back from the pallet)
        seq, rep = planner.move(q, above_bin, -1)
        replans += rep
        emit(seq, 0.0, -1, "COLLISION AVOIDED  re-routing" if rep else "")
        emit(planner._joint_seg(above_bin, at_bin.q)[1:], 0.0, -1, "")
        for k in range(10):                                   # grasp
            frames.append([at_bin.q, k / 9, part_i, None, "", replans, rejected])
        emit(planner._joint_seg(at_bin.q, above_bin)[1:], 1.0, part_i, "")

        # collision-aware traverse to the pallet, carrying the part
        seq, rep = planner.move(above_bin, above_pal, part_i)
        replans += rep
        emit(seq[1:], 1.0, part_i, "COLLISION AVOIDED  re-routing" if rep else "")

        at_pal = planner.ik(xy, place_grasp_z(layer), above_pal)
        emit(planner._joint_seg(above_pal, at_pal.q)[1:], 1.0, part_i, "")
        for k in range(10):                                   # release
            frames.append([at_pal.q, 1 - k / 9, part_i, None, "", replans, rejected])
        frames[-1][3] = (part_i, xy, layer)                   # snap part to its cell
        placed += 1
        emit(planner._joint_seg(at_pal.q, above_pal)[1:], 0.0, -1, "")
        q = above_pal
        part_i += 1

    seq, rep = planner.move(q, np.zeros(6), -1)
    replans += rep
    emit(seq[1:], 0.0, -1, "")
    return frames, dict(replans=replans, rejected=rejected, placed=placed,
                        accuracy_mm=max_err * 1000, total=len(REQUESTS))


def draw_hud(frame, placed, total_slots, sim_t, replans, rejected, accuracy_mm, caption):
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
        f"Parts placed   {placed}/{N_PARTS}",
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
        colour = (240, 90, 80, 255) if "REJECT" in caption else (250, 200, 70, 255)
        w = img.width
        dr.rectangle([w / 2 - 230, img.height - 60, w / 2 + 230, img.height - 22], fill=(10, 12, 18, 190))
        dr.text((w / 2 - 215, img.height - 54), caption, font=cap_f, fill=colour)
    return np.asarray(img)


def render(mujoco, model, planner, frames, meta, height, width, render_stride=4):
    data = mujoco.MjData(model)
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    grip_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in GRIP_JOINTS]
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    part_adr = [free_joint_qadr(mujoco, model, f"part{k}") for k in range(N_PARTS)]

    def grasp_centre():
        return 0.5 * (data.xpos[lp] + data.xpos[rp])

    open_g, closed_g = gripper_open_closed(mujoco, model, frames[0][0], grip_adr, grip_act)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.34, 0.02, TABLE_H + 0.06]
    cam.distance = 1.8
    cam.azimuth = 128
    cam.elevation = -22

    r = mujoco.Renderer(model, height, width)
    part_state = [np.array([x, y, TABLE_H + PART, 1, 0, 0, 0.0]) for (x, y) in BIN]
    placed, imgs = 0, []
    for idx, (qk, gf, carry, snap, caption, replans, rejected) in enumerate(frames):
        data.qpos[:6] = qk
        data.qpos[grip_adr] = open_g + (closed_g - open_g) * gf
        if carry >= 0 and gf > 0.5:
            mujoco.mj_forward(model, data)
            c = grasp_centre()
            part_state[carry] = np.array([c[0], c[1], c[2], 1, 0, 0, 0.0])
        if snap is not None:
            pi, xy, layer = snap
            part_state[pi] = np.array([xy[0], xy[1], TABLE_H + rest_centre_z(layer), 1, 0, 0, 0.0])
            placed += 1
        if idx % render_stride and idx != len(frames) - 1:
            continue                       # advance state every frame, render every Nth
        for k in range(N_PARTS):
            data.qpos[part_adr[k]:part_adr[k] + 7] = part_state[k]
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        imgs.append(draw_hud(r.render(), placed, meta["total"], idx * DT,
                             replans, rejected, meta["accuracy_mm"], caption))
    return imgs


def save_gif(frames, path, stride=2, fps=15, scale=0.46, colors=84):
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


def save_mp4(frames, path, fps=30):
    """Write a web-ready H.264 MP4 (yuv420p for browser compatibility)."""
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8,
                     macro_block_size=8, output_params=["-pix_fmt", "yuv420p"])
    print(f"saved {path}  ({len(frames)} frames @ {fps} fps, {path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write docs/palletizing_cell.gif")
    parser.add_argument("--mp4", action="store_true", help="write docs/palletizing_cell.mp4 (web-ready)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    import mujoco

    model = build_cell(mujoco)
    arm = SerialArm.ur5e()

    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*BIN[0], 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    planner = Planner(mujoco, model, arm, gripper_len)
    frames, meta = build_plan(planner)
    print(f"plan: {len(frames)} frames | placed {meta['placed']}/{N_PARTS} | "
          f"re-plans {meta['replans']} | rejected {meta['rejected']} | "
          f"max IK err {meta['accuracy_mm']:.3f} mm")
    # a smoother render for video; sparser for the GIF
    render_stride = 3 if args.mp4 else 4
    imgs = render(mujoco, model, planner, frames, meta, args.height, args.width, render_stride)

    if args.mp4:
        save_mp4(imgs, ROOT / "docs" / "palletizing_cell.mp4")
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

"""Pick-and-place demo: drive the arm through a sequence of Cartesian poses and
animate the motion in 3D.

The arm starts at home, reaches down to a pick location, closes the gripper,
carries the object to a place location, releases it, and returns home. Position
targets are solved with inverse kinematics; the motion between them is a
synchronised trapezoidal trajectory.

Run:
    python apps/pick_and_place.py             show the animation window
    python apps/pick_and_place.py --save      write docs/pick_and_place.gif
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from armik import SerialArm, solve_ik, joint_trajectory          # noqa: E402
from armik.rotations import matrix_from_rotvec                    # noqa: E402


def tool_down_pose(x, y, z):
    """Target pose at (x, y, z) with the tool pointing straight down."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_rotvec(np.array([0.0, np.pi, 0.0]))
    T[:3, 3] = [x, y, z]
    return T


def build_plan(arm):
    """Return (q_traj, holding, phase) sampled over the whole motion.

    holding[k] is True while the gripper carries the object; phase[k] is a short
    label for the current segment (used as the animation caption).
    """
    q_home = np.zeros(6)
    pick = (0.40, -0.25, 0.15)
    place = (0.35, 0.30, 0.15)
    approach_h = 0.35

    # (name, target pose, gripper-holding-during-this-segment)
    segments = [
        ("move to pick",   tool_down_pose(pick[0],  pick[1],  approach_h), False),
        ("reach down",     tool_down_pose(*pick),                          False),
        ("grasp + lift",   tool_down_pose(pick[0],  pick[1],  approach_h), True),
        ("carry to place", tool_down_pose(place[0], place[1], approach_h), True),
        ("lower object",   tool_down_pose(*place),                         True),
        ("release + lift", tool_down_pose(place[0], place[1], approach_h), False),
        ("return home",    None,                                           False),
    ]

    q = q_home
    q_wpts, hold_flags, names = [q_home], [], []
    for name, T, holding in segments:
        if T is None:
            q_next = q_home
        else:
            res = solve_ik(arm, T, q)
            if not res.success:
                print(f"warning: IK did not fully converge for '{name}' "
                      f"(pos err {res.position_error:.2e} m)")
            q_next = res.q
        q_wpts.append(q_next)
        hold_flags.append(holding)
        names.append(name)
        q = q_next

    q_all, hold_all, phase_all = [], [], []
    for i in range(len(q_wpts) - 1):
        _, q_seg, _ = joint_trajectory(q_wpts[i], q_wpts[i + 1],
                                       v_max=1.6, a_max=3.2, dt=0.04)
        if i > 0:
            q_seg = q_seg[1:]
        q_all.append(q_seg)
        hold_all.append(np.full(len(q_seg), hold_flags[i]))
        phase_all.append([names[i]] * len(q_seg))

    return np.vstack(q_all), np.concatenate(hold_all), sum(phase_all, [])


def arm_points(arm, q):
    """Base-to-tool joint origins as an (n+1, 3) array of 3D points."""
    return np.array([T[:3, 3] for T in arm.frames(q)])


def animate(arm, q_traj, holding, phase, save_path=None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    pick = np.array([0.40, -0.25, 0.15])
    place = np.array([0.35, 0.30, 0.15])

    def setup_axes():
        ax.clear()
        ax.set_xlim(-0.3, 0.7)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.1, 0.7)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=22, azim=-50)

    def draw(k):
        setup_axes()
        pts = arm_points(arm, q_traj[k])
        # links
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="#2a5db0",
                lw=3, ms=5, mfc="#0d1b2a")
        # base
        ax.scatter(*pts[0], color="black", s=60, marker="s")
        # pick / place pads
        ax.scatter(*pick, color="#8a8a8a", s=80, marker="s", alpha=0.6)
        ax.scatter(*place, color="#8a8a8a", s=80, marker="s", alpha=0.6)
        # tool tip, coloured by gripper state
        tip = pts[-1]
        if holding[k]:
            ax.scatter(*tip, color="#e07a1c", s=110, marker="o",
                       edgecolors="black", label="holding")
            ax.scatter(*tip, color="#e07a1c", s=25, marker="s")   # carried object
        else:
            ax.scatter(*tip, color="#2c9e5a", s=90, marker="o", edgecolors="black")
        ax.set_title(f"UR5 pick-and-place   |   {phase[k]}", fontsize=11)
        return ax,

    frames = range(0, len(q_traj))
    anim = FuncAnimation(fig, draw, frames=frames, interval=40, blit=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=25))
        print(f"saved {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write docs/pick_and_place.gif instead of showing a window")
    args = parser.parse_args()

    arm = SerialArm()
    q_traj, holding, phase = build_plan(arm)
    print(f"planned {len(q_traj)} trajectory samples across {len(set(phase))} phases")
    out = str(ROOT / "docs" / "pick_and_place.gif") if args.save else None
    animate(arm, q_traj, holding, phase, save_path=out)


if __name__ == "__main__":
    main()

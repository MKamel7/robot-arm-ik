"""Palletizing cell app: collision detection, the reachability guard, and plan
integrity (every reachable part placed exactly once).

Needs the optional `sim` extras (mujoco); skipped cleanly when mujoco isn't
installed (the fast pure-NumPy CI job never sees this file run).
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

import palletizing_cell   # noqa: E402
from palletizing_cell import (   # noqa: E402
    DEFAULT_CONFIG, Planner, build_cell, build_plan, place_grasp_z, rest_centre_z,
    solve_ik, tool_down_pose,
)
from armik import SerialArm   # noqa: E402


@pytest.fixture(scope="module")
def planner():
    """Build the cell and a Planner once; building + IK-seeding is the
    expensive part and every test below shares the same instance."""
    cfg = DEFAULT_CONFIG
    model = build_cell(mujoco, cfg)
    arm = SerialArm.ur5e()

    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*cfg.bin[0], 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    return Planner(mujoco, model, arm, gripper_len, cfg)


@pytest.fixture(scope="module")
def plan_result(planner):
    """Run build_plan once and reuse its (frames, meta) across the plan-
    integrity tests; this is the ~1-2 s part of the module."""
    return build_plan(planner)


def test_collides_at_fixture(planner):
    """A pose with the tool tip inside the machine fixture's column (world
    z in [table_h, table_h + 0.30], centred at cfg.fixture_xy) must be flagged
    as a collision by the planner's contact query."""
    cfg = planner.cfg
    q = planner.ik(cfg.fixture_xy, 0.15, cfg.ik_seed).q
    assert planner._collides(q, carried=-1) is True


def test_home_pose_is_clear(planner):
    """The ready pose above the supply bin is nowhere near the fixture and
    must not be reported as colliding."""
    cfg = planner.cfg
    q = planner.ik(cfg.home_xy, cfg.home_z, cfg.ik_seed).q
    assert planner._collides(q, carried=-1) is False


def test_reachability_guard_math():
    """Cheap non-MuJoCo sanity check of the workspace-radius guard used by
    build_plan before it even attempts an IK solve."""
    cfg = DEFAULT_CONFIG
    assert np.hypot(*cfg.unreachable_xy) > cfg.reach_max
    assert np.hypot(*cfg.pallet_xy[0]) <= cfg.reach_max


def test_unreachable_slot_is_rejected(plan_result):
    """The one deliberately-unreachable requested slot must be caught by the
    reachability validator and rejected, not silently dropped or faked."""
    _, meta = plan_result
    assert meta["rejected"] == 1


def test_all_reachable_parts_placed(plan_result):
    cfg = DEFAULT_CONFIG
    _, meta = plan_result
    assert meta["placed"] == cfg.n_parts == 8
    assert meta["rejected"] == 1


def test_each_part_placed_exactly_once(plan_result):
    """Walk the frame list for placement snaps (frame[3] = (part_index, xy,
    layer)) and confirm there are exactly meta['placed'] of them and no part
    index repeats (every part is placed exactly once)."""
    frames, meta = plan_result
    snaps = [frame[3] for frame in frames if frame[3] is not None]
    part_indices = [part_i for part_i, _xy, _layer in snaps]
    assert len(snaps) == meta["placed"]
    assert len(part_indices) == len(set(part_indices))


def test_default_plan_has_no_colliding_frames(planner, plan_result):
    """Every frame build_plan emits -- not just the two long traverse legs
    but the four short per-part bin/pallet legs (descend to grasp, lift after
    grasp, descend to place, lift after release) -- must be independently
    collision-free. Those four legs used to bypass Planner.move and call
    planner._joint_seg directly (no collision check at all); this is the
    regression test for that bug: re-run planner._collides on every frame of
    the default plan and require zero hits, the same check that first
    surfaced the bug (64 colliding frames, all from the four unrouted legs)."""
    frames, _meta = plan_result
    colliding = [i for i, f in enumerate(frames) if planner._collides(f[0], f[2])]
    assert colliding == []


def test_video_stride_keeps_displayed_motion_smooth(plan_result):
    """The MP4 path samples every `render_stride`-th plan frame, so the motion
    an actual viewer sees between two DISPLAYED frames is bounded by
    cfg.v_max * stride * dt -- the re-timing limits, not the frame count, set
    how smooth the output can be. Time-optimal re-timing made the plan bursty
    (most frames run well under v_max, a few saturate it), so a stride chosen
    without reference to v_max strobes on exactly those fast segments: at
    stride 8 the tool jumps a median 64 mm and up to 407 mm per displayed
    frame, which is what made the first Phase 1 render unwatchable.

    Guard both ends: the analytic worst case implied by the limits, and the
    empirical travel over the real default plan. Raising cfg.v_max or the MP4
    stride without re-checking playback should fail here rather than silently
    shipping a strobing video."""
    frames, _meta = plan_result
    cfg = DEFAULT_CONFIG
    stride = 1                                  # main()'s MP4 render_stride

    # analytic bound: what the velocity limit alone permits per displayed frame
    assert np.degrees(cfg.v_max * stride * cfg.dt) < 8.0

    # empirical: end-effector travel per displayed frame over the real plan
    arm = SerialArm.ur5e()
    ee = np.array([arm.fk(f[0])[:3, 3] for f in frames[::stride]])
    step_mm = np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1000
    assert np.percentile(step_mm, 50) < 25.0
    assert step_mm.max() < 80.0


def test_default_scene_never_needs_rrt(plan_result):
    """With DEFAULT_CONFIG's single fixture, the cheap lift-over heuristic
    already clears every blocked move (its 4 fixed candidate heights sit well
    above the fixture's default 0.30 m height) -- so RRT-Connect should never
    engage here, and nothing should ever be a genuine give-up either. This is
    the counterpart to test_rrt_connect_engages_when_lift_over_cannot_clear_
    obstacle below: RRT-Connect is wired in, but it is not dead code that
    fires unconditionally regardless of whether it's actually needed."""
    _, meta = plan_result
    assert meta["replans"] > 0                 # the fixture does force re-routes...
    assert meta["rrt_routes"] == 0              # ...but never needs RRT-Connect for them
    assert meta["failed_routes"] == 0


# --- Harder obstacle: the lift-over heuristic's fixed candidate heights ----
# (cfg.lift_heights, all <= 0.66 m local) provably cannot clear a fixture
# whose column reaches to local z = 2 * fixture_half_h >= 1.0 m. Only
# RRT-Connect's general joint-space search can route around a fixture that
# tall (it reconfigures the arm rather than trying to fly straight over the
# obstacle) -- this is the scenario that proves RRT-Connect has real
# substance in Planner.move, not just plumbing that never fires.

HARD_CONFIG = dataclasses.replace(DEFAULT_CONFIG, fixture_half_h=0.5, rrt_seed=7)


@pytest.fixture(scope="module")
def hard_planner():
    """Same cell as `planner`, but with a machine fixture tall enough
    (fixture_half_h=0.5 -> column top at local z=1.0) that every one of the
    lift-over heuristic's fixed candidate heights (max 0.66) still runs the
    tool straight into it. rrt_seed is fixed so the RRT-Connect call below is
    deterministic rather than occasionally slow or flaky."""
    cfg = HARD_CONFIG
    model = build_cell(mujoco, cfg)
    arm = SerialArm.ur5e()

    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*cfg.bin[0], 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    return Planner(mujoco, model, arm, gripper_len, cfg)


def _blocked_bin_approach(planner):
    """A specific bin-approach move in the hard scenario, reconstructed the
    same way build_plan chains its IK solves (home -> above bin[0] -> place
    at pallet[0] -> above bin[1]), without paying for a full 8-part plan.
    This particular leg (above the first pallet placement, back to the
    second bin cell) is directly blocked by the tall fixture."""
    cfg = planner.cfg
    home_q = planner.ik(cfg.home_xy, cfg.home_z, cfg.ik_seed).q
    place0 = planner.ik(cfg.pallet_xy[0], place_grasp_z(cfg, 0), home_q)
    above_pal0 = planner.ik(cfg.pallet_xy[0], cfg.z_safe, place0.q).q
    above_bin1 = planner.ik(cfg.bin[1], cfg.z_safe, above_pal0).q
    return above_pal0, above_bin1


def test_lift_over_cannot_clear_the_tall_fixture(hard_planner):
    """Sanity check on the scenario itself: the tall fixture's column top
    (local z = 2 * fixture_half_h) sits above every one of the lift-over
    heuristic's candidate heights, so none of its 4 tries can be a real
    detour -- confirms the harder scenario is actually harder, not just
    differently shaped."""
    cfg = hard_planner.cfg
    assert 2 * cfg.fixture_half_h > max(cfg.lift_heights)


def test_rrt_connect_engages_when_lift_over_cannot_clear_obstacle(hard_planner, monkeypatch):
    """The substance test: with the tall-fixture scenario, the direct path is
    blocked and the lift-over heuristic's 4 fixed heights all fail, yet
    Planner.move still returns a collision-free route -- because RRT-Connect
    engaged, not because it silently gave up. A spy on rrt_connect proves it
    was actually called and actually returned a path, rather than just
    trusting that the right caption string came out the other end."""
    calls = []
    real_rrt_connect = palletizing_cell.rrt_connect

    def spy(*args, **kwargs):
        result = real_rrt_connect(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(palletizing_cell, "rrt_connect", spy)

    qa, qb = _blocked_bin_approach(hard_planner)
    assert hard_planner._path_collides(hard_planner._joint_seg(qa, qb), -1)  # direct: blocked

    path, replanned, caption = hard_planner.move(qa, qb, -1)

    assert len(calls) == 1                  # rrt_connect was actually invoked...
    assert calls[0] is not None              # ...and it actually found a path
    assert replanned is True
    assert caption == "RRT-CONNECT ROUTE"    # not the lift-over caption, not a give-up
    assert path is not None and len(path) > 1
    assert not hard_planner._path_collides(path, -1)
    assert np.allclose(path[0], qa, atol=1e-2)
    assert np.allclose(path[-1], qb, atol=1e-2)


# --- Physics-mode execution: real dynamics, not just "it ran" ---------------
# build_cell(physics=True) swaps the 6 arm actuators for raw torque and adds
# a per-part weld equality; render(physics=True) tracks each frame's (q, qd)
# with physics_control.PDGravityCompController (real mj_step, not
# teleportation) and grasps by activating that weld instead of scripting the
# part's position. This is the integration test for that path: bounded PD +
# gravity-comp tracking error (not diverging, not just "finite"), and a
# grasp that actually holds the part rigidly to the gripper (a near-constant
# gap, not just "ends up somewhere plausible").

@pytest.fixture(scope="module")
def physics_planner():
    """Same construction as `planner`, but on a build_cell(physics=True)
    model (torque actuators + per-part welds instead of the default
    position-servo actuators)."""
    cfg = DEFAULT_CONFIG
    model = build_cell(mujoco, cfg, physics=True)
    arm = SerialArm.ur5e()

    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_left_pad")
    rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_right_pad")
    seed = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    res = solve_ik(arm, tool_down_pose(*cfg.bin[0], 0.20), seed)
    data.qpos[:6] = res.q
    mujoco.mj_forward(model, data)
    gripper_len = data.site_xpos[sid][2] - 0.5 * (data.xpos[lp][2] + data.xpos[rp][2])

    return model, Planner(mujoco, model, arm, gripper_len, cfg)


def test_physics_mode_tracks_and_grasps_one_part(physics_planner):
    """Run a single pick-and-place cycle under physics(=True) execution and
    check the two things that would be silently wrong if physics_control.py
    or the weld grasp weren't actually doing real work: tracking error stays
    small and bounded (PD + gravity-comp is actually converging, not just
    producing finite numbers), and the grasp gap (held part centre to
    grasp_centre()) stays essentially constant (a real rigid weld, not a
    part slipping/falling away while nominally "carried")."""
    model, planner = physics_planner
    cfg = planner.cfg
    frames, meta = build_plan(planner)
    frames, meta = palletizing_cell._limit_to_n_parts(frames, meta, 1)
    assert frames[-1][3] is not None            # cut right after part 0's placement snap event

    track_err, grasp_gap = [], []
    data = mujoco.MjData(model)
    palletizing_cell.render(mujoco, model, planner, frames, meta, height=180, width=240,
                            render_stride=8, physics=True,
                            track_err_out=track_err, grasp_gap_out=grasp_gap, data=data)
    track_err = np.array(track_err)

    assert np.all(np.isfinite(track_err))
    assert track_err.max() < 0.15, f"tracking error too large somewhere: {track_err.max()}"
    assert track_err[-1].max() < 0.05, f"did not converge by the end: {track_err[-1]}"

    assert len(grasp_gap) > 20                  # the carry leg spans many physics frames
    grasp_gap = np.array(grasp_gap)
    assert np.all(np.isfinite(grasp_gap))
    # a rigid weld holds the gap essentially constant; a part slipping or
    # merely resting near the gripper (not actually constrained) would drift
    assert grasp_gap.std() < 0.005, f"grasp gap not rigid, std={grasp_gap.std()}"

    # The part must have actually arrived near its pallet cell under real
    # physics (nothing scripts its position in physics mode, see render()) --
    # not floating away and not falling through the table. This is a looser
    # tolerance than the kinematic path's sub-mm placement: the weld captures
    # the part's *actual* pose the instant the gripper reports fully closed,
    # and closing kinematically-driven (not force-controlled) fingers onto a
    # freely-resting part is real contact physics -- it measurably nudges the
    # part by a few cm before the grip "sets", same as a sloppy real gripper
    # closing on a loose part would. The weld then faithfully carries that
    # exact pose to the pallet (proven rigid above), so a few cm of
    # placement offset is an honest artifact of that pre-grasp contact, not
    # a bug -- unlike the kinematic path, which recomputes/recentres the
    # part's pose from scratch every single frame and so never shows this.
    part0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part0")
    part_pos = data.xpos[part0]
    target_xy = cfg.pallet_xy[0]         # part0 is the first part picked, placed at slot 0 layer 0
    assert abs(part_pos[0] - target_xy[0]) < 0.06, f"part x off target: {part_pos}"
    assert abs(part_pos[1] - target_xy[1]) < 0.06, f"part y off target: {part_pos}"
    assert abs(part_pos[2] - (cfg.table_h + rest_centre_z(cfg, 0))) < 0.01, \
        f"part not resting at the pallet's layer-0 height: {part_pos}"


def test_physics_mode_never_clips_fixture_during_reroute(physics_planner):
    """Regression test for the physics-mode collision-avoidance gap: the
    kinematic reference path Planner.move plans around the fixture is
    provably collision-free (test_default_plan_has_no_colliding_frames), but
    physics=True execution doesn't just play that path back -- PD +
    gravity-comp tracking error can push the *actually simulated* arm off of
    it. Without cfg.physics_clearance_margin (build_cell(physics=True)), a
    transient ~0.17-0.26 rad wrist tracking-error spike during this exact
    two-part plan's fast collision-avoidance reroute leg let the real,
    simulated gripper clip fixture_lamp -- a genuine MuJoCo contact -- even
    though the reference path it was tracking never touched it. This is the
    test that would have caught that gap: run the same two-part physics-mode
    plan and demand zero real (penetrating, not just inside the margin
    buffer -- see render()'s fixture_contact_out) contacts between any mover
    body and the fixture, not merely that tracking error stayed small."""
    model, planner = physics_planner
    frames, meta = build_plan(planner)
    frames, meta = palletizing_cell._limit_to_n_parts(frames, meta, 2)
    assert frames[-1][3] is not None            # cut right after part 1's placement snap event

    track_err, fixture_contacts = [], []
    data = mujoco.MjData(model)
    palletizing_cell.render(mujoco, model, planner, frames, meta, height=180, width=240,
                            render_stride=8, physics=True,
                            track_err_out=track_err, fixture_contact_out=fixture_contacts,
                            data=data)

    # This plan is expected to still need collision-avoidance re-routing
    # (same fixture as the default scenario) -- confirms the test is
    # actually exercising the reroute leg that used to clip, not a
    # coincidentally-clear plan.
    assert any("re-routing" in f[4] or "RRT" in f[4] for f in frames)
    assert np.array(track_err).max() < 0.15    # bounded tracking, same threshold as the 1-part test

    assert fixture_contacts == [], \
        f"real (penetrating) fixture contact during physics execution: {fixture_contacts[:5]}"

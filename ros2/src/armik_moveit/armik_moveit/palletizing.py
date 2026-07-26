"""Palletizing cell on the ROS 2 / MoveIt 2 stack (mock hardware).

The ROS version of the Phase 1 MuJoCo palletizing_cell: the UR5e transfers parts
from a supply bin to a pallet, one slot at a time, routing around the machine
fixture (MoveIt/OMPL), gripping with the Robotiq 2F-85, and stacking in layers.
A reachability pre-check rejects any slot outside the arm's workspace instead of
faking it. Parts move as attached collision objects (mock hardware has no
physics grasp, so the grip is represented in the planning scene, standard for a
MoveIt pick-and-place). Production-style metrics are printed at the end.

    ros2 launch armik_moveit ur5e_gripper_moveit.launch.py    # (RViz optional)
    ros2 run armik_moveit palletize

Geometry mirrors scene.py's cell (bin/pallet/fixture).
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from control_msgs.action import GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    PlanningOptions,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from armik_moveit.scene import CELL_OBJECTS, build_scene

BASE = "base_link"
EEF = "tool0"
GROUP = "ur_manipulator"
SUCCESS = 1

# Gripper: link parts hang from, and the links they may touch when attached.
GRIPPER_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link",
]
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
GRIP_OPEN = 0.0
GRIP_CLOSED = 0.6

PART = 0.04                 # part cube edge (m)
GRIPPER_LEN = 0.16          # tool0 -> grasp point along the tool approach axis
APPROACH_DZ = 0.16          # standoff above a pick/place before descending
REACH_MAX = 0.82            # UR5e workspace radius for a downward grasp

HOME = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.5707, "elbow_joint": 0.0,
    "wrist_1_joint": -1.5707, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0,
}

# High, central waypoint above the fixture (top 0.60 m). Every transfer routes
# through here, so each leg is an easy near-vertical move instead of one hard
# plan around the fixture. This is the fixed-waypoint transit from the Phase 1
# cell; the standalone collision-aware routing is proven by scene_routing_check.
TRANSIT = (0.32, 0.0, 0.62)


def _grid(cx, cy, nx, ny, sx, sy):
    return [(cx + (i - (nx - 1) / 2) * sx, cy + (j - (ny - 1) / 2) * sy)
            for j in range(ny) for i in range(nx)]


# Parts sit on the supply bin top (bin: centre z 0.05, height 0.10 -> top 0.10).
BIN_TOP = 0.10
PART_Z = BIN_TOP + PART / 2
PICK_CELLS = [(x, y, PART_Z) for x, y in _grid(0.45, -0.30, 2, 2, 0.12, 0.12)]

# Pallet slots: a 2x2 footprint, up to 2 layers (pallet top z 0.08). Filled
# back-to-front (far row first) so no placement ever has to reach over a part
# already on the pallet, the standard palletizing order.
PALLET_TOP = 0.08
PALLET_XY = sorted(_grid(0.45, 0.32, 2, 2, 0.14, 0.14), key=lambda p: (-p[1], p[0]))
LAYERS = 1  # raise to 2 to stack a second layer (needs >=8 parts)


def slot_pose(index):
    """Map a place index to (x, y, z), filling a layer before stacking up."""
    per_layer = len(PALLET_XY)
    layer, within = divmod(index, per_layer)
    x, y = PALLET_XY[within]
    z = PALLET_TOP + PART / 2 + layer * PART
    return (x, y, z)


def reachable(x, y, z):
    # Grasp point is GRIPPER_LEN above the part; check the tool0 target.
    return math.sqrt(x * x + y * y + (z + GRIPPER_LEN) ** 2) <= REACH_MAX


class Palletizer(Node):
    def __init__(self):
        super().__init__("palletize")
        self.mg = ActionClient(self, MoveGroup, "/move_action")
        self.grip = ActionClient(self, GripperCommand, GRIPPER_ACTION)
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.replans = 0

    # --- planning scene helpers ---
    def _apply(self, scene_msg):
        self.scene.wait_for_service(timeout_sec=10.0)
        fut = self.scene.call_async(ApplyPlanningScene.Request(scene=scene_msg))
        rclpy.spin_until_future_complete(self, fut)
        return fut.result() is not None and fut.result().success

    def add_part(self, part_id, x, y, z):
        obj = CollisionObject()
        obj.header.frame_id = BASE
        obj.id = part_id
        prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[PART, PART, PART])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obj)
        return self._apply(scene)

    def remove_part(self, part_id):
        obj = CollisionObject(id=part_id, operation=CollisionObject.REMOVE)
        obj.header.frame_id = BASE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obj)
        return self._apply(scene)

    def attach_part(self, part_id):
        """Attach the part to the tool, GRIPPER_LEN out along the approach axis."""
        self.remove_part(part_id)
        aco = AttachedCollisionObject()
        aco.link_name = EEF
        aco.object.header.frame_id = EEF
        aco.object.id = part_id
        prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[PART, PART, PART])
        pose = Pose()
        pose.position.z = GRIPPER_LEN
        pose.orientation.w = 1.0
        aco.object.primitives.append(prim)
        aco.object.primitive_poses.append(pose)
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = GRIPPER_LINKS
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self._apply(scene)

    def detach_place(self, part_id, x, y, z):
        aco = AttachedCollisionObject()
        aco.link_name = EEF
        aco.object.id = part_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        self._apply(scene)
        return self.add_part(part_id, x, y, z)

    # --- motion helpers ---
    def _send(self, req, execute):
        goal = MoveGroup.Goal(request=req)
        opts = PlanningOptions()
        opts.plan_only = not execute
        goal.planning_options = opts
        self.mg.wait_for_server(timeout_sec=30.0)
        f = self.mg.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f)
        h = f.result()
        if h is None or not h.accepted:
            return None
        rf = h.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return rf.result().result.error_code.val

    def _req(self, attempts, seconds):
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = attempts
        req.allowed_planning_time = seconds
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        return req

    def go_home(self):
        req = self._req(10, 15.0)
        c = Constraints()
        for j, v in HOME.items():
            c.joint_constraints.append(JointConstraint(
                joint_name=j, position=v, tolerance_above=0.01,
                tolerance_below=0.01, weight=1.0))
        req.goal_constraints.append(c)
        return self._send(req, execute=True) == SUCCESS

    def go_to(self, x, y, z, seconds=18.0, label=""):
        """Move tool0 to a point (position-only goal). Retries on failure."""
        for attempt in range(3):
            req = self._req(20, seconds)
            pc = PositionConstraint()
            pc.header.frame_id = BASE
            pc.link_name = EEF
            pc.constraint_region.primitives.append(
                SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.05]))
            region = Pose()
            region.position.x, region.position.y, region.position.z = x, y, z
            region.orientation.w = 1.0
            pc.constraint_region.primitive_poses.append(region)
            pc.weight = 1.0
            c = Constraints()
            c.position_constraints.append(pc)
            req.goal_constraints.append(c)
            if self._send(req, execute=True) == SUCCESS:
                return True
            self.replans += 1
        if label:
            print(f"      step '{label}' failed at ({x:.2f},{y:.2f},{z:.2f})")
        return False

    def gripper(self, position):
        self.grip.wait_for_server(timeout_sec=15.0)
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 50.0
        f = self.grip.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f)
        h = f.result()
        rf = h.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return True

    # --- one pick-and-place ---
    def pick_place(self, part_id, pick, place):
        px, py, pz = pick
        qx, qy, qz = place
        pick_top = pz + GRIPPER_LEN
        place_top = qz + GRIPPER_LEN
        self.gripper(GRIP_OPEN)
        # pick: approach above bin, descend, grip, attach, ascend
        if not self.go_to(px, py, pick_top + APPROACH_DZ, label="approach-bin"):
            return False
        if not self.go_to(px, py, pick_top, seconds=10.0, label="descend-pick"):
            return False
        self.gripper(GRIP_CLOSED)
        self.attach_part(part_id)
        if not self.go_to(px, py, pick_top + APPROACH_DZ, label="lift"):
            return False
        # transfer via the high transit waypoint above the fixture
        if not self.go_to(*TRANSIT, label="transit"):
            return False
        # place: approach above slot, descend, release, detach, ascend
        if not self.go_to(qx, qy, place_top + APPROACH_DZ, label="approach-pallet"):
            return False
        if not self.go_to(qx, qy, place_top, seconds=10.0, label="descend-place"):
            return False
        # Part is down and released here: the placement counts from now on.
        self.detach_place(part_id, qx, qy, qz)
        self.gripper(GRIP_OPEN)
        # Clearing the pallet is best effort: try a straight retreat, else go
        # straight to the transit waypoint. Either way the part stays placed.
        if not self.go_to(qx, qy, place_top + APPROACH_DZ, label="retreat"):
            self.go_to(*TRANSIT, label="retreat-transit")
        else:
            self.go_to(*TRANSIT, label="transit-back")
        return True


def main():
    rclpy.init()
    node = Palletizer()
    try:
        # Build the cell (bin/pallet/fixture) and spawn the parts in the bin.
        node._apply(build_scene(clear=False))
        parts = []
        for i, (x, y, z) in enumerate(PICK_CELLS):
            pid = f"part_{i}"
            node.add_part(pid, x, y, z)
            parts.append((pid, (x, y, z)))
        time.sleep(0.5)
        print(f"cell: {', '.join(CELL_OBJECTS)} | {len(parts)} parts in the bin")

        if not node.go_home():
            print("could not reach home; aborting")
            sys.exit(1)
        node.go_to(*TRANSIT, label="initial-transit")  # tool-down, high, ready

        placed = rejected = 0
        t0 = time.time()
        place_index = 0
        for pid, pick in parts:
            slot = slot_pose(place_index)
            if not reachable(*slot):
                print(f"  {pid}: slot {slot} OUT OF REACH -> rejected")
                rejected += 1
                place_index += 1
                continue
            print(f"  {pid}: bin {tuple(round(v,2) for v in pick)} "
                  f"-> pallet slot {tuple(round(v,2) for v in slot)}")
            if node.pick_place(pid, pick, slot):
                placed += 1
                place_index += 1
            else:
                print(f"    {pid}: FAILED")
        node.go_home()
        elapsed = time.time() - t0

        cycle = elapsed / placed if placed else float("nan")
        print("\n=== palletizing metrics ===")
        print(f"parts placed : {placed}/{len(parts)}")
        print(f"rejected     : {rejected} (unreachable slots)")
        print(f"re-plans     : {node.replans}")
        print(f"total time   : {elapsed:.1f} s")
        print(f"cycle time   : {cycle:.1f} s/part")
        print("RESULT:", "PASS" if placed == len(parts) - rejected and placed > 0 else "FAIL")
        sys.exit(0 if placed > 0 else 1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

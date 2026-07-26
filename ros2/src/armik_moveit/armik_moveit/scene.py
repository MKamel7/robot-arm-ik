"""Populate the MoveIt planning scene for the palletizing cell.

Adds three collision objects in the arm's base_link frame, mirroring the Phase 1
MuJoCo cell: a supply bin (-y side), a pallet (+y side), and a machine fixture
standing between them. The fixture is tall enough that a straight bin-to-pallet
path collides with it, so OMPL has to route around, the ROS version of the
collision-aware routing from the standalone cell.

    ros2 run armik_moveit populate_scene        # add the cell (default)
    ros2 run armik_moveit populate_scene --clear # remove it again

Geometry lives in scene_config so the palletizing node can share it.
"""
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

BASE_FRAME = "base_link"

# Box collision objects: id -> (size_xyz, center_xyz), all in BASE_FRAME (metres).
CELL_OBJECTS = {
    # Supply bin the arm picks from (front-right of the base).
    "supply_bin": ((0.24, 0.20, 0.10), (0.45, -0.30, 0.05)),
    # Pallet the arm stacks onto (front-left of the base).
    "pallet": ((0.30, 0.30, 0.08), (0.45, 0.32, 0.04)),
    # Machine fixture standing between bin and pallet: a straight bin->pallet
    # transfer at working height would clip this, so the planner must go around.
    "fixture": ((0.10, 0.12, 0.60), (0.45, 0.0, 0.30)),
}


def _box(object_id: str, size, center) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = BASE_FRAME
    obj.id = object_id
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(s) for s in size]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(c) for c in center)
    pose.orientation.w = 1.0
    obj.primitives.append(prim)
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD
    return obj


def build_scene(clear: bool) -> PlanningScene:
    scene = PlanningScene()
    scene.is_diff = True
    for object_id, (size, center) in CELL_OBJECTS.items():
        obj = _box(object_id, size, center)
        obj.operation = CollisionObject.REMOVE if clear else CollisionObject.ADD
        scene.world.collision_objects.append(obj)
    return scene


def main() -> None:
    clear = "--clear" in sys.argv
    rclpy.init()
    node = Node("populate_scene")
    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=15.0):
        node.get_logger().error("/apply_planning_scene not available (is move_group up?)")
        rclpy.shutdown()
        sys.exit(1)

    req = ApplyPlanningScene.Request()
    req.scene = build_scene(clear)
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    ok = future.result() is not None and future.result().success
    action = "cleared" if clear else "added"
    print(f"planning scene {action}: {'OK' if ok else 'FAILED'} "
          f"({', '.join(CELL_OBJECTS)})")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

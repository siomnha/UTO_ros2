"""Strict path-only IFDS ROS node for the IFDS -> UTO control chain."""

import json
from pathlib import Path as FilePath

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray
import yaml

from .ifds_core import (
    PathStatus,
    SphereObstacle,
    multiply_quaternions,
    plan_ifds_path,
    transform_position,
)


class IFDSPlannerNode(Node):
    """Consume mean pose/goal/obstacles and publish paths; never flight setpoints."""

    def __init__(self) -> None:
        super().__init__("ifds_planner")
        defaults = {
            "use_sim_time": True,
            "planner_only": True,
            "planning_frame": "map",
            "odometry_topic": "/Odometry",
            "goal_topic": "/ifds/goal",
            "mission_goal_topic": "/ifds/mission_goal",
            "path_topic": "/ifds/local_path",
            "path_status_topic": "/ifds/path_status",
            "obstacle_topic": "/ifds/obstacles",
            "obstacle_yaml": "",
            "obstacle_frame": "map",
            "planning_rate": 2.0,
            "path_validity_duration": 0.8,
            "target_threshold": 0.05,
            "obstacle_clearance": 0.25,
            "tf_timeout": 0.1,
            "gnss_denied": True,
            "dynamic_obstacles": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not self.get_parameter("planner_only").value:
            raise RuntimeError("this integration requires planner_only=true")
        self.frame = str(self.get_parameter("planning_frame").value)
        self.position = None
        self.goal = None
        self.goal_orientation = None
        self.goal_generation = 0
        self.path_generation = 0
        self.replan_requested = False
        self.obstacles = []
        self.obstacles_valid = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter("mission_goal_topic").value, goal_qos
        )
        self.path_pub = self.create_publisher(Path, self.get_parameter("path_topic").value, 10)
        self.status_pub = self.create_publisher(
            String, self.get_parameter("path_status_topic").value, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("odometry_topic").value, self._on_odometry, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value, self._on_goal, 10
        )
        if self.get_parameter("dynamic_obstacles").value:
            self.create_subscription(
                MarkerArray,
                self.get_parameter("obstacle_topic").value,
                self._on_obstacles,
                10,
            )
        self._load_obstacles()
        rate = float(self.get_parameter("planning_rate").value)
        self.timer = self.create_timer(1.0 / rate, self._plan_if_requested)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _transform(self, position, orientation, source_frame, stamp):
        if not source_frame:
            raise ValueError("EMPTY_INPUT_FRAME")
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("NONFINITE_INPUT_POSITION")
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
            raise ValueError("NONFINITE_INPUT_QUATERNION")
        if source_frame == self.frame:
            return position, orientation
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame,
                source_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=float(self.get_parameter("tf_timeout").value)),
            )
        except TransformException as exception:
            raise ValueError("TF_UNAVAILABLE") from exception
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_q = [rotation.x, rotation.y, rotation.z, rotation.w]
        output_position = transform_position(
            position, [translation.x, translation.y, translation.z], transform_q
        )
        output_orientation = multiply_quaternions(transform_q, orientation)
        return output_position, output_orientation

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        try:
            position, _ = self._transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                message.header.frame_id,
                message.header.stamp,
            )
        except ValueError as exception:
            self.position = None
            self._invalidate(str(exception))
            return
        self.position = position
        if self.goal is not None:
            self.replan_requested = True

    def _on_goal(self, message: PoseStamped) -> None:
        pose = message.pose
        try:
            position, orientation = self._transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                message.header.frame_id,
                message.header.stamp,
            )
            if np.linalg.norm(orientation) < 1e-12:
                raise ValueError("INVALID_GOAL_QUATERNION")
        except ValueError as exception:
            self.goal = None
            self.goal_orientation = None
            self._invalidate(str(exception))
            return
        self.goal = position
        self.goal_orientation = orientation / np.linalg.norm(orientation)
        self.goal_generation += 1
        self.replan_requested = True
        checked = PoseStamped()
        checked.header.frame_id = self.frame
        checked.header.stamp = self.get_clock().now().to_msg()
        if checked.header.stamp.sec == 0 and checked.header.stamp.nanosec == 0:
            checked.header.stamp.nanosec = 1
        checked.pose.position.x, checked.pose.position.y, checked.pose.position.z = position
        (
            checked.pose.orientation.x,
            checked.pose.orientation.y,
            checked.pose.orientation.z,
            checked.pose.orientation.w,
        ) = self.goal_orientation
        self.goal_pub.publish(checked)

    def _load_obstacles(self) -> None:
        path = str(self.get_parameter("obstacle_yaml").value)
        if not path:
            return
        try:
            data = yaml.safe_load(FilePath(path).read_text()) or {}
            if data.get("frame_id") != self.frame:
                raise ValueError("OBSTACLE_FRAME_MISMATCH")
            self.obstacles = [
                SphereObstacle(item["center"], float(item["radius"]))
                for item in data.get("obstacles", [])
            ]
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exception:
            self.obstacles = []
            self.obstacles_valid = False
            self._invalidate(f"INVALID_OBSTACLE_YAML:{exception}")

    def _on_obstacles(self, message: MarkerArray) -> None:
        updated = []
        try:
            for marker in message.markers:
                if marker.header.frame_id != self.frame:
                    raise ValueError("OBSTACLE_FRAME_MISMATCH")
                if marker.action == marker.DELETEALL:
                    updated = []
                    continue
                if marker.action == marker.DELETE:
                    continue
                radius = max(marker.scale.x, marker.scale.y, marker.scale.z) / 2.0
                updated.append(
                    SphereObstacle(
                        [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                        radius,
                    )
                )
        except ValueError as exception:
            self.obstacles_valid = False
            self._invalidate(str(exception))
            return
        self.obstacles = updated
        self.obstacles_valid = True
        self.replan_requested = True

    def _plan_if_requested(self) -> None:
        if not self.replan_requested:
            return
        self.replan_requested = False
        if self.position is None or self.goal is None or not self.obstacles_valid:
            self._invalidate("MISSING_POSITION_OR_GOAL")
            return
        now = self.get_clock().now()
        try:
            points = plan_ifds_path(
                self.position,
                self.goal,
                self.obstacles,
                float(self.get_parameter("obstacle_clearance").value),
                float(self.get_parameter("target_threshold").value),
            )
        except ValueError as exception:
            self._invalidate(str(exception))
            return
        path = Path()
        path.header.frame_id = self.frame
        path.header.stamp = now.to_msg()
        if path.header.stamp.sec == 0 and path.header.stamp.nanosec == 0:
            path.header.stamp.nanosec = 1
        for point in points:
            waypoint = PoseStamped()
            waypoint.header = path.header
            waypoint.pose.position.x, waypoint.pose.position.y, waypoint.pose.position.z = point
            waypoint.pose.orientation.w = 1.0
            path.poses.append(waypoint)
        self.path_generation += 1
        self.path_pub.publish(path)
        stamp_ns = path.header.stamp.sec * 1_000_000_000 + path.header.stamp.nanosec
        planned_at = stamp_ns * 1e-9
        self._publish_status(
            PathStatus(
                True,
                stamp_ns,
                self.path_generation,
                self.goal_generation,
                planned_at,
                planned_at + float(self.get_parameter("path_validity_duration").value),
                "PATH_REACHES_MISSION_GOAL",
            )
        )

    def _invalidate(self, reason: str) -> None:
        self.replan_requested = False
        now = self._now()
        self._publish_status(
            PathStatus(
                False,
                0,
                self.path_generation,
                self.goal_generation,
                now,
                now,
                reason or "NO_VALID_IFDS_PATH",
            )
        )

    def _publish_status(self, status: PathStatus) -> None:
        self.status_pub.publish(String(data=status.to_json()))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IFDSPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

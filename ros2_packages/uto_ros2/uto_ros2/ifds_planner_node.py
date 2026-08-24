"""Path-only ROS wrapper around the original modulation-based IFDS planner."""

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
import yaml

from .ifds_contract import (
    PathStatus,
    SemanticPathGeneration,
    multiply_quaternions,
    selected_odometry_topic,
    transform_position,
    validate_obstacle_world,
)
from .ifds_core import IFDSConfig, IFDSPlanner, obstacle_from_mapping
from .ifds_path_adapter import (
    already_at_mission_goal,
    append_exact_ifds_goal,
    static_ifds_replan_requested,
)


class IFDSPlannerNode(Node):
    """Publish IFDS paths and contracts; this node never publishes flight setpoints."""

    def __init__(self) -> None:
        super().__init__("ifds_planner")
        defaults = {
            "use_sim_time": True, "planner_only": True, "frame_id": "map",
            "gnss_denied": True, "gnss_odom_topic": "/x500/gnss/odometry",
            "fast_lio_odom_topic": "/Odometry", "goal_topic": "/ifds/goal",
            "mission_goal_topic": "/ifds/mission_goal", "path_topic": "/ifds/local_path",
            "path_status_topic": "/ifds/path_status", "obstacle_updates_topic": "/ifds/obstacles",
            "obstacles_yaml": "", "planning_rate_hz": 2.0, "path_validity_duration": 0.8,
            "plan_once_static": False,
            "tf_timeout": 0.1, "allow_empty_obstacles": False,
            "validate_world_consistency": False, "world_sdf": "",
            "path_geometry_change_threshold": 0.05, "path_resample_spacing": 0.10,
            "rho0": 2.5, "sigma0": 0.01, "cruise_speed": 2.0, "dt": 0.1,
            "max_iterations": 1000, "target_threshold": 0.05, "delta_g": 2.0,
            "alpha_deg": 0.0, "shape_following": False, "min_gamma": 1.02,
            "dynamic_obstacles": False, "velocity_mode": "normal", "optimizer_mode": 0,
            "local_optimizer_period_steps": 5, "wall_modulation_gain": 1.5,
            "wall_influence_distance": 1.0,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)
        if not self.get_parameter("planner_only").value:
            raise RuntimeError("uto_ros2 requires IFDS planner_only=true")
        self.frame = str(self.get_parameter("frame_id").value)
        self.position = None
        self.goal = None
        self.goal_orientation = None
        self.goal_generation = 0
        self.obstacle_generation = 0
        self.replan_requested = False
        self.static_path_planned = False
        self.obstacles = []
        self.map_valid = False
        self.semantic_generation = SemanticPathGeneration(
            self.get_parameter("path_geometry_change_threshold").value,
            self.get_parameter("path_resample_spacing").value,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.create_publisher(PoseStamped, self._p("mission_goal_topic"), goal_qos)
        self.path_pub = self.create_publisher(Path, self._p("path_topic"), 10)
        self.status_pub = self.create_publisher(String, self._p("path_status_topic"), 10)
        odom_topic = selected_odometry_topic(
            bool(self._p("gnss_denied")), str(self._p("gnss_odom_topic")),
            str(self._p("fast_lio_odom_topic")),
        )
        self.odometry_subscription = self.create_subscription(Odometry, odom_topic, self._on_odometry, 10)
        self.create_subscription(PoseStamped, self._p("goal_topic"), self._on_goal, 10)
        self.create_subscription(String, self._p("obstacle_updates_topic"), self._on_obstacles, 10)
        self._load_obstacles()
        self.create_timer(1.0 / max(float(self._p("planning_rate_hz")), 0.1), self._plan)
        self.get_logger().info(f"path-only IFDS ready: odometry={odom_topic}, frame={self.frame}")

    def _p(self, name):
        return self.get_parameter(name).value

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _transform(self, position, orientation, source_frame, stamp):
        position = np.asarray(position, float)
        orientation = np.asarray(orientation, float)
        if not source_frame or position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("INVALID_INPUT_FRAME_OR_POSITION")
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)) or np.linalg.norm(orientation) < 1e-12:
            raise ValueError("INVALID_INPUT_QUATERNION")
        if source_frame == self.frame:
            return position, orientation / np.linalg.norm(orientation)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, source_frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=float(self._p("tf_timeout"))),
            )
        except TransformException as exception:
            raise ValueError("TF_UNAVAILABLE") from exception
        t, q = tf.transform.translation, tf.transform.rotation
        quaternion = [q.x, q.y, q.z, q.w]
        return transform_position(position, [t.x, t.y, t.z], quaternion), multiply_quaternions(quaternion, orientation)

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        try:
            self.position, _ = self._transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                message.header.frame_id, message.header.stamp,
            )
            if static_ifds_replan_requested(
                self.goal is not None,
                bool(self._p("plan_once_static")),
                self.static_path_planned,
            ):
                self.replan_requested = True
        except ValueError as exception:
            self.position = None
            self._invalidate(str(exception))

    def _on_goal(self, message: PoseStamped) -> None:
        pose = message.pose
        try:
            position, orientation = self._transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                message.header.frame_id, message.header.stamp,
            )
        except ValueError as exception:
            self._invalidate(str(exception))
            return
        self.goal, self.goal_orientation = position, orientation
        self.goal_generation += 1
        self.static_path_planned = False
        self.replan_requested = True
        self._invalidate("NEW_GOAL_PENDING")
        checked = PoseStamped()
        checked.header.frame_id = self.frame
        checked.header.stamp = self.get_clock().now().to_msg()
        if checked.header.stamp.sec == 0 and checked.header.stamp.nanosec == 0:
            checked.header.stamp.nanosec = 1
        checked.pose.position.x, checked.pose.position.y, checked.pose.position.z = position
        (checked.pose.orientation.x, checked.pose.orientation.y,
         checked.pose.orientation.z, checked.pose.orientation.w) = orientation
        self.goal_pub.publish(checked)

    def _parse_obstacles(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("header"), dict):
            raise ValueError("OBSTACLE_HEADER_REQUIRED")
        if data["header"].get("frame_id") != self.frame:
            raise ValueError("OBSTACLE_FRAME_MISMATCH")
        items = data.get("obstacles", [])
        if not items and not bool(self._p("allow_empty_obstacles")):
            raise ValueError("EMPTY_OBSTACLE_MAP_NOT_ALLOWED")
        return [obstacle_from_mapping(item) for item in items]

    def _load_obstacles(self) -> None:
        path = str(self._p("obstacles_yaml"))
        try:
            if not path:
                raise ValueError("OBSTACLE_YAML_REQUIRED")
            data = yaml.safe_load(FilePath(path).read_text()) or {}
            obstacles = self._parse_obstacles(data)
            if bool(self._p("validate_world_consistency")):
                world = str(self._p("world_sdf"))
                valid, reasons = validate_obstacle_world(path, world, self.frame)
                if not valid:
                    raise ValueError("WORLD_MISMATCH:" + ";".join(reasons))
            self.obstacles, self.map_valid = obstacles, True
            self.obstacle_generation += 1
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exception:
            self.obstacles, self.map_valid = [], False
            self._invalidate(f"INVALID_OBSTACLE_MAP:{exception}")

    def _on_obstacles(self, message: String) -> None:
        try:
            data = yaml.safe_load(message.data) or {}
            obstacles = self._parse_obstacles(data)
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as exception:
            self.map_valid = False
            self._invalidate(f"INVALID_OBSTACLE_UPDATE:{exception}")
            return
        self.obstacles, self.map_valid = obstacles, True
        self.obstacle_generation += 1
        self.static_path_planned = False
        self.replan_requested = self.goal is not None and self.position is not None

    def _config(self) -> IFDSConfig:
        names = IFDSConfig.__dataclass_fields__
        return IFDSConfig(**{name: self._p(name) for name in names})

    def _plan(self) -> None:
        if not self.replan_requested or self.position is None or self.goal is None or not self.map_valid:
            return
        self.replan_requested = False
        now = self._now()
        if already_at_mission_goal(self.position, self.goal):
            self._invalidate("ALREADY_AT_MISSION_GOAL", terminal=True)
            return
        planner = IFDSPlanner(self._config(), list(self.obstacles), plan_time_s=now)
        found, waypoints, reason = planner.plan(self.position.copy(), self.goal.copy())
        if found:
            found, waypoints, reason = append_exact_ifds_goal(planner, waypoints, self.goal)
        if not found or len(waypoints) < 2 or not np.all(np.isfinite(waypoints)):
            self._invalidate(reason or "NO_VALID_IFDS_PATH")
            if self._p("plan_once_static"):
                self.replan_requested = True
            return
        stamp = self.get_clock().now().to_msg()
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp.nanosec = 1
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        generation = self.semantic_generation.update(
            waypoints, self.position, self.goal_generation, self.obstacle_generation
        )
        path = Path()
        path.header.frame_id, path.header.stamp = self.frame, stamp
        for point in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        status = PathStatus(True, stamp_ns, generation, self.goal_generation,
                            self.obstacle_generation, now,
                            now + float(self._p("path_validity_duration")), reason)
        self.path_pub.publish(path)
        self.status_pub.publish(String(data=status.to_json()))
        self.static_path_planned = bool(self._p("plan_once_static"))

    def _invalidate(self, reason: str, terminal: bool = False) -> None:
        status = PathStatus(False, 0, self.semantic_generation.generation,
                            self.goal_generation, self.obstacle_generation,
                            max(self._now(), 0.0), max(self._now(), 0.0), str(reason), terminal)
        self.status_pub.publish(String(data=status.to_json()))
        if terminal or reason == "NEW_GOAL_PENDING":
            self.get_logger().info(f"IFDS path transition: {reason}")
        else:
            self.get_logger().error(f"IFDS path invalid: {reason}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IFDSPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

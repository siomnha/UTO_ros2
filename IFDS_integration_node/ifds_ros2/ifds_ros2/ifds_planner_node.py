"""ROS 2 node wrapping the IFDS local planner for PX4/Gazebo simulations."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import threading
from typing import Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .ifds_core import IFDSConfig, IFDSPlanner, ObstacleGeometry, obstacle_from_mapping
from .path_tracker import CarrotPathTracker


class IFDSPlannerNode(Node):
    """Plan local IFDS paths from GNSS or FAST-LIO2 odometry to a goal."""

    def __init__(self) -> None:
        super().__init__('ifds_planner')
        self.declare_parameter('obstacles_yaml', '')
        self.declare_parameter('gnss_denied', False)
        self.declare_parameter('gnss_odom_topic', '/x500/gnss/odometry')
        self.declare_parameter('fast_lio_odom_topic', '/Odometry')
        self.declare_parameter('goal_topic', '/ifds/goal')
        self.declare_parameter('path_topic', '/ifds/local_path')
        self.declare_parameter('setpoint_topic', '/mavros/setpoint_position/local')
        self.declare_parameter('status_topic', '/ifds/status')
        self.declare_parameter('obstacle_updates_topic', '/ifds/obstacles')
        self.declare_parameter('planning_rate_hz', 2.0)
        self.declare_parameter('setpoint_rate_hz', 30.0)
        self.declare_parameter('lookahead_distance', 2.5)
        self.declare_parameter('yaw_rate_limit', 1.0)
        self.declare_parameter('max_setpoint_speed', 3.0)
        self.declare_parameter('rho0', 2.5)
        self.declare_parameter('sigma0', 0.01)
        self.declare_parameter('cruise_speed', 2.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('max_iterations', 1000)
        self.declare_parameter('target_threshold', 1.0)
        self.declare_parameter('delta_g', 2.0)
        self.declare_parameter('alpha_deg', 0.0)
        self.declare_parameter('shape_following', False)
        self.declare_parameter('min_gamma', 1.02)
        self.declare_parameter('dynamic_obstacles', False)
        self.declare_parameter('velocity_mode', 'normal')
        self.declare_parameter('optimizer_mode', 0)
        self.declare_parameter('local_optimizer_period_steps', 5)
        self.declare_parameter('wall_modulation_gain', 1.5)
        self.declare_parameter('wall_influence_distance', 1.0)
        self.declare_parameter('hold_on_failure', True)
        self.declare_parameter('frame_id', 'map')

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.current_pose: Optional[PoseStamped] = None
        self.goal: Optional[np.ndarray] = None
        self.last_setpoint_position: Optional[np.ndarray] = None
        self.hold_position: Optional[np.ndarray] = None
        self.last_yaw: Optional[float] = None
        self.goal_generation = 0
        self.tracker_lock = threading.Lock()
        self.tracker = CarrotPathTracker(
            lookahead_distance=float(self.get_parameter('lookahead_distance').value),
        )
        self.obstacles = self._load_obstacles()
        self.path_pub = self.create_publisher(NavPath, str(self.get_parameter('path_topic').value), 10)
        self.setpoint_pub = self.create_publisher(PoseStamped, str(self.get_parameter('setpoint_topic').value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter('status_topic').value), 10)

        self.sensor_callbacks = ReentrantCallbackGroup()
        self.planning_callbacks = MutuallyExclusiveCallbackGroup()
        self.setpoint_callbacks = MutuallyExclusiveCallbackGroup()
        odom_topic = self._selected_odom_topic()
        self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            20,
            callback_group=self.sensor_callbacks,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter('goal_topic').value),
            self._goal_cb,
            10,
            callback_group=self.sensor_callbacks,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('obstacle_updates_topic').value),
            self._obstacle_update_cb,
            10,
            callback_group=self.sensor_callbacks,
        )
        planning_rate = float(self.get_parameter('planning_rate_hz').value)
        setpoint_rate = float(self.get_parameter('setpoint_rate_hz').value)
        self.setpoint_period = 1.0 / max(setpoint_rate, 1.0)
        self.create_timer(
            1.0 / max(planning_rate, 0.1),
            self._plan_timer_cb,
            callback_group=self.planning_callbacks,
        )
        self.create_timer(
            self.setpoint_period,
            self._setpoint_timer_cb,
            callback_group=self.setpoint_callbacks,
        )
        self.get_logger().info(
            f'IFDS planner ready: odom={odom_topic}, obstacles={len(self.obstacles)}, '
            f'frame={self.frame_id}, velocity_mode={self.get_parameter("velocity_mode").value}'
        )

    def _selected_odom_topic(self) -> str:
        if bool(self.get_parameter('gnss_denied').value):
            return str(self.get_parameter('fast_lio_odom_topic').value)
        return str(self.get_parameter('gnss_odom_topic').value)

    def _load_obstacles(self) -> list[ObstacleGeometry]:
        yaml_path = str(self.get_parameter('obstacles_yaml').value)
        if not yaml_path:
            return []
        path = Path(yaml_path).expanduser()
        if not path.exists():
            self.get_logger().warning(f'obstacles_yaml does not exist: {path}')
            return []
        with path.open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream) or {}
        obstacles, _ = self._parse_obstacle_payload(data)
        return obstacles

    def _parse_obstacle_payload(self, data: object) -> tuple[list[ObstacleGeometry], dict]:
        if isinstance(data, dict):
            obstacle_items = data.get('obstacles', [])
            header = data.get('header', {}) or {}
        elif isinstance(data, list):
            obstacle_items = data
            header = {}
        else:
            obstacle_items = []
            header = {}
        return [obstacle_from_mapping(item) for item in obstacle_items], header

    def _obstacle_update_cb(self, msg: String) -> None:
        try:
            data = yaml.safe_load(msg.data) or {}
            obstacles, header = self._parse_obstacle_payload(data)
            self.obstacles = obstacles
        except (TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
            self.get_logger().warning(f'failed to parse obstacle update: {exc}')
            return
        frame = header.get('frame_id', self.frame_id) if isinstance(header, dict) else self.frame_id
        self.get_logger().info(f'updated IFDS obstacles: {len(self.obstacles)} frame={frame}')

    def _odom_cb(self, msg: Odometry) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.current_pose = pose
        if self.last_yaw is None:
            quaternion = msg.pose.pose.orientation
            sin_yaw = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
            cos_yaw = 1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2)
            self.last_yaw = math.atan2(sin_yaw, cos_yaw)

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.goal_generation += 1
        with self.tracker_lock:
            self.tracker.clear()
            self.last_setpoint_position = None
            self.hold_position = self._current_position() if self.current_pose is not None else None
        self.get_logger().info(f'new IFDS goal: {self.goal.tolist()}')

    def _plan_timer_cb(self) -> None:
        if self.current_pose is None or self.goal is None:
            return
        start = self._current_position()
        goal = self.goal.copy()
        goal_generation = self.goal_generation
        planner = IFDSPlanner(
            self._config_from_params(),
            self.obstacles,
            plan_time_s=self._now_seconds(),
        )
        found, waypoints, reason = planner.plan(start, goal)
        if goal_generation != self.goal_generation:
            return
        if found and float(np.linalg.norm(start - goal)) <= float(self.get_parameter('target_threshold').value):
            with self.tracker_lock:
                self.tracker.clear()
                self.last_setpoint_position = None
                self.hold_position = goal.copy()
            self._publish_status(f'GOAL_REACHED: {reason}')
            self.setpoint_pub.publish(self._hold_setpoint())
            return
        if found and len(waypoints) > 1:
            with self.tracker_lock:
                accepted, deviation = self.tracker.replace_path(waypoints, start)
                if accepted:
                    self.hold_position = None
                    if self.last_setpoint_position is None:
                        self.last_setpoint_position = start.copy()
            if accepted:
                self.path_pub.publish(self._to_path(waypoints))
                self._publish_status(
                    f'PLAN_OK_REPLACED {len(waypoints)} waypoints: {reason}; carrot_jump={deviation:.2f}m'
                )
            return

        with self.tracker_lock:
            self.tracker.clear()
            self.last_setpoint_position = None
            self.hold_position = start.copy()
        self._publish_status(f'PLAN_FAILED_HOLDING: {reason}')
        self.setpoint_pub.publish(self._hold_setpoint())

    def _config_from_params(self) -> IFDSConfig:
        return IFDSConfig(
            rho0=float(self.get_parameter('rho0').value),
            sigma0=float(self.get_parameter('sigma0').value),
            cruise_speed=float(self.get_parameter('cruise_speed').value),
            dt=float(self.get_parameter('dt').value),
            max_iterations=int(self.get_parameter('max_iterations').value),
            target_threshold=float(self.get_parameter('target_threshold').value),
            delta_g=float(self.get_parameter('delta_g').value),
            alpha_deg=float(self.get_parameter('alpha_deg').value),
            shape_following=bool(self.get_parameter('shape_following').value),
            min_gamma=float(self.get_parameter('min_gamma').value),
            dynamic_obstacles=bool(self.get_parameter('dynamic_obstacles').value),
            velocity_mode=str(self.get_parameter('velocity_mode').value),
            optimizer_mode=int(self.get_parameter('optimizer_mode').value),
            local_optimizer_period_steps=int(self.get_parameter('local_optimizer_period_steps').value),
            wall_modulation_gain=float(self.get_parameter('wall_modulation_gain').value),
            wall_influence_distance=float(self.get_parameter('wall_influence_distance').value),
        )

    def _to_path(self, waypoints: np.ndarray) -> NavPath:
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        for point in waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _setpoint_timer_cb(self) -> None:
        if self.current_pose is None:
            return
        with self.tracker_lock:
            if not self.tracker.active:
                if self.hold_position is not None:
                    self.setpoint_pub.publish(self._hold_setpoint())
                return
            position = self._current_position()
            carrot, tangent, _ = self.tracker.carrot(position)
            self.setpoint_pub.publish(self._to_setpoint(carrot, tangent))

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _current_position(self) -> np.ndarray:
        return np.array(
            [
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z,
            ],
            dtype=float,
        )

    def _to_setpoint(self, carrot: np.ndarray, tangent: np.ndarray) -> PoseStamped:
        setpoint = self._hold_setpoint()
        target = self._limit_setpoint_motion(carrot)
        setpoint.pose.position.x = float(target[0])
        setpoint.pose.position.y = float(target[1])
        setpoint.pose.position.z = float(target[2])
        desired_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        yaw = self._rate_limited_yaw(desired_yaw)
        setpoint.pose.orientation.x = 0.0
        setpoint.pose.orientation.y = 0.0
        setpoint.pose.orientation.z = math.sin(yaw / 2.0)
        setpoint.pose.orientation.w = math.cos(yaw / 2.0)
        return setpoint

    def _limit_setpoint_motion(self, target: np.ndarray) -> np.ndarray:
        if self.last_setpoint_position is None:
            self.last_setpoint_position = target.copy()
            return target
        max_step = max(float(self.get_parameter('max_setpoint_speed').value), 0.0) * self.setpoint_period
        delta = target - self.last_setpoint_position
        distance = float(np.linalg.norm(delta))
        if max_step > 0.0 and distance > max_step:
            target = self.last_setpoint_position + delta * (max_step / distance)
        self.last_setpoint_position = target.copy()
        return target

    def _rate_limited_yaw(self, desired_yaw: float) -> float:
        if self.last_yaw is None:
            self.last_yaw = desired_yaw
            return desired_yaw
        error = math.atan2(math.sin(desired_yaw - self.last_yaw), math.cos(desired_yaw - self.last_yaw))
        max_delta = max(float(self.get_parameter('yaw_rate_limit').value), 0.0) * self.setpoint_period
        self.last_yaw += float(np.clip(error, -max_delta, max_delta))
        return self.last_yaw

    def _hold_setpoint(self) -> PoseStamped:
        hold = PoseStamped()
        hold.header.stamp = self.get_clock().now().to_msg()
        hold.header.frame_id = self.frame_id
        if self.current_pose is not None:
            hold.pose = deepcopy(self.current_pose.pose)
            if self.hold_position is not None:
                hold.pose.position.x = float(self.hold_position[0])
                hold.pose.position.y = float(self.hold_position[1])
                hold.pose.position.z = float(self.hold_position[2])
        else:
            hold.pose.orientation.w = 1.0
        return hold

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = IFDSPlannerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Ground-truth-only ROS validation recorder; never publishes flight commands."""

from collections import deque
import json
import math
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .math_utils import quat_to_rot
from .trajectory import Trajectory
from .validation import TrialRecord, ValidationStore


class ValidationMetricsNode(Node):
    """Observe ground truth, planner data, and context and atomically persist trials."""

    def __init__(self):
        super().__init__("uto_validation_metrics")
        defaults = {
            "ground_truth_topic": "/model/x500/odometry",
            "belief_topic": "/Odometry",
            "trajectory_topic": "/uto/trajectory",
            "diagnostics_topic": "/uto/diagnostics",
            "px4_status_topic": "/uto/px4_status",
            "mission_goal_topic": "/ifds/mission_goal",
            "trial_context_topic": "/validation/trial_context",
            "trial_written_topic": "/validation/trial_written",
            "validation_output_directory": "~/uto_validation_results",
            "terminal_average_window": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.store = ValidationStore(self.get_parameter("validation_output_directory").value)
        self.context = None
        self.samples = deque()
        self.trajectory = None
        self.diagnostics = {}
        self.goal = None
        self.pending_finish = None
        transient = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        topic = lambda name: self.get_parameter(name).value
        self.create_subscription(Odometry, topic("ground_truth_topic"), self._ground_truth, 50)
        self.create_subscription(Odometry, topic("belief_topic"), lambda message: None, 10)
        self.create_subscription(String, topic("trajectory_topic"), self._trajectory, 10)
        self.create_subscription(String, topic("diagnostics_topic"), self._diagnostics, 10)
        self.create_subscription(String, topic("px4_status_topic"), lambda message: None, 10)
        self.create_subscription(PoseStamped, topic("mission_goal_topic"), self._goal, transient)
        self.create_subscription(String, topic("trial_context_topic"), self._context, transient)
        self.written = self.create_publisher(String, topic("trial_written_topic"), transient)
        self.flush_timer = self.create_timer(0.05, self._flush_pending)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _ground_truth(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        rotation = quat_to_rot([q.x, q.y, q.z, q.w])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.asin(np.clip(-rotation[2, 0], -1.0, 1.0))
        rates = message.twist.twist.angular
        self.samples.append((self._now(), np.array([p.x, p.y, p.z]), roll, pitch, np.array([rates.x, rates.y, rates.z])))

    def _trajectory(self, message):
        try:
            self.trajectory = Trajectory.from_json(message.data)
        except ValueError as exception:
            self.get_logger().warning(str(exception))

    def _diagnostics(self, message):
        try:
            self.diagnostics = json.loads(message.data)
        except ValueError:
            pass

    def _goal(self, message):
        p = message.pose.position
        self.goal = np.array([p.x, p.y, p.z])

    def _context(self, message):
        try:
            context = json.loads(message.data)
        except ValueError:
            return
        if context.get("event") == "start":
            self.context = context
            self.samples.clear()
        elif context.get("event") in ("finish", "abort") and self.context:
            if context["event"] == "abort":
                self._finish(context, self._now())
            else:
                self.pending_finish = (context, self._now())

    def _flush_pending(self):
        if self.pending_finish is None:
            return
        event, finish_time = self.pending_finish
        if self._now() - finish_time >= self.get_parameter("terminal_average_window").value:
            self.pending_finish = None
            self._finish(event, finish_time)

    def _finish(self, event, finish_time):
        now = self._now()
        window = self.get_parameter("terminal_average_window").value
        recent = [sample for sample in self.samples if finish_time <= sample[0] <= finish_time + window]
        positions = np.asarray([sample[1] for sample in recent])
        goal = np.asarray(self.context["goal"], dtype=float)
        final = positions.mean(axis=0) if len(positions) else np.full(3, np.nan)
        error = final - goal
        path_errors = []
        if self.trajectory is not None:
            for _, position, *_ in self.samples:
                path_errors.append(float(np.min(np.linalg.norm(self.trajectory.states[:, :3] - position, axis=1))))
        rates = np.asarray([sample[4] for sample in self.samples])
        initial = self.context["initial_error"]
        record = TrialRecord(
            self.context["trial_id"], self.context["planner_mode"], int(self.context["seed"]),
            int(self.context["sample_index"]), event["event"] == "finish", event.get("failure_reason", ""),
            *goal, *initial, *final, *error, float(np.linalg.norm(error)),
            float(np.sqrt(np.mean(np.square(path_errors)))) if path_errors else np.nan,
            max(path_errors, default=np.nan),
            max((abs(sample[2]) for sample in self.samples), default=np.nan),
            max((abs(sample[3]) for sample in self.samples), default=np.nan),
            float(np.sqrt(np.mean(rates**2))) if rates.size else np.nan,
            self.trajectory.times[-1] if self.trajectory is not None else np.nan,
            self.diagnostics.get("cold_build_time", np.nan), self.diagnostics.get("parameter_update_time", np.nan),
            self.diagnostics.get("solve_time", np.nan), self.diagnostics.get("extraction_time", np.nan),
            self.diagnostics.get("worker_total_time", np.nan),
            self.diagnostics.get("predicted_terminal_position_covariance_trace")
            if self.diagnostics.get("predicted_terminal_position_covariance_trace") is not None
            else np.nan,
        )
        self.store.append(record)
        self.written.publish(String(data=json.dumps({"trial_id": record.trial_id, "written": True})))
        self.context = None


def main(args=None):
    rclpy.init(args=args)
    node = ValidationMetricsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

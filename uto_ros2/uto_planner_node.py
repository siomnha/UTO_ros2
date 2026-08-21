"""ROS wiring for belief/path snapshots, background solve, admission, and commit."""

from dataclasses import dataclass
import json
import threading
from typing import Optional
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from .belief_adapter import Belief
from .ifds_path_adapter import Polyline, path_generation
from .math_utils import align_enu_velocity, ned_to_enu
from .planner_runtime import (
    LatestWinsWorker,
    PLANNER_PARAMETER_DEFAULTS,
    PlannerState,
    PlanningRequest,
    build_runtime_components,
    can_resume,
    commit_continuity_errors,
    commit_due_status,
    update_goal_dwell,
)


@dataclass(frozen=True)
class PlannerSnapshot:
    now: float
    belief: Optional[Belief]
    belief_stable: bool
    path: Optional[dict]
    px4: dict
    velocity_fresh: bool


class UTOPlannerNode(Node):
    """Receive inputs -> request solve -> validate -> commit -> publish."""

    def __init__(self) -> None:
        super().__init__("uto_planner")
        self._declare_parameters()
        self.lock = threading.RLock()
        self.state = PlannerState.WAIT_PX4
        self.belief = None
        self.belief_stable = False
        self.path = None
        self.velocity = None
        self.velocity_stamp = 0.0
        self.velocity_clock_offset = None
        self.px4 = {"connected": False, "hold_ready": False, "failsafe": False}
        self.request_generation = 0
        self.last_path_generation = ""
        self.first_request_submitted = False
        self.solve_in_progress = False
        self.last_gate = None
        self.last_commit_lateness = 0.0
        self.last_commit_errors = None
        self.goal_since = None
        self._build_components()
        self._create_interfaces()

    def _declare_parameters(self) -> None:
        for name, value in PLANNER_PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, value)

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _build_components(self) -> None:
        components = build_runtime_components(self._parameter)
        self.nlp = components.nlp
        self.adapter = components.adapter
        self.delay = components.delay
        self.buffer = components.buffer
        self.manager = components.manager
        self.gate = components.gate
        self.worker = LatestWinsWorker(self._solve_request)

    def _create_interfaces(self) -> None:
        self.trajectory_pub = self.create_publisher(String, self._parameter("trajectory_topic"), 10)
        self.execution_command_pub = self.create_publisher(
            String, self._parameter("execution_command_topic"), 10
        )
        self.diagnostics_pub = self.create_publisher(
            String, self._parameter("diagnostics_topic"), 10
        )
        self.create_subscription(Odometry, self._parameter("belief_topic"), self._on_odometry, 10)
        self.create_subscription(Path, self._parameter("path_topic"), self._on_path, 10)
        self.create_subscription(
            String, self._parameter("px4_state_topic"), self._on_px4_status, 10
        )
        if self._parameter("velocity_source") == "separate_velocity_topic":
            self.create_subscription(
                TwistStamped, self._parameter("velocity_topic"), self._on_velocity, 10
            )
        if self._parameter("velocity_source") == "px4_vehicle_odometry":
            self.create_subscription(
                VehicleOdometry,
                self._parameter("px4_velocity_topic"),
                self._on_px4_velocity,
                rclpy.qos.qos_profile_sensor_data,
            )
        self.create_service(Trigger, self._parameter("resume_service"), self._on_resume)
        self.planning_timer = self.create_timer(
            1.0 / self._parameter("replan_rate"), self._on_planning_timer
        )
        self.commit_timer = self.create_timer(
            1.0 / self._parameter("commit_check_rate"), self._on_commit_timer
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_velocity(self, message: TwistStamped) -> None:
        self.velocity = np.array(
            [message.twist.linear.x, message.twist.linear.y, message.twist.linear.z]
        )
        self.velocity_stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _on_px4_velocity(self, message: VehicleOdometry) -> None:
        now = self._now()
        source_us = getattr(message, "timestamp_sample", 0) or message.timestamp
        source_time = float(source_us) * 1e-6
        mode = self._parameter("px4_velocity_time_mode")
        observed_offset = now - source_time
        if mode == "ros":
            self.velocity_clock_offset = 0.0
            if abs(observed_offset) > self._parameter("source_clock_tolerance"):
                self.velocity = None
                self.belief_stable = False
                self.adapter.detector.invalidate()
                self._publish_diagnostics("PX4 velocity ROS timestamp is not synchronized", "ERROR")
                return
        elif mode == "offset":
            if self.velocity_clock_offset is None:
                self.velocity_clock_offset = observed_offset
            elif abs(observed_offset - self.velocity_clock_offset) > self._parameter(
                "source_clock_tolerance"
            ):
                self.velocity = None
                self.belief_stable = False
                self.adapter.detector.invalidate()
                self._publish_diagnostics("PX4 velocity clock domain changed", "ERROR")
                return
        else:
            self.velocity = None
            self._publish_diagnostics("unsupported PX4 velocity time mode", "ERROR")
            return
        enu = ned_to_enu(message.velocity)
        try:
            self.velocity = align_enu_velocity(
                enu,
                self._parameter("velocity_frame_alignment_mode"),
                self._parameter("velocity_frame_yaw_offset"),
            )
            self.velocity_stamp = source_time + self.velocity_clock_offset
        except ValueError as exception:
            self.velocity = None
            self.belief_stable = False
            self.adapter.detector.invalidate()
            self._publish_diagnostics(str(exception), "ERROR")

    def _velocity_for_odometry(self, message: Odometry, now: float):
        if self._parameter("velocity_source") == "patched_odometry_twist":
            velocity = np.array(
                [
                    message.twist.twist.linear.x,
                    message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                ]
            )
            return velocity, np.all(np.isfinite(velocity))
        velocity_age = now - self.velocity_stamp
        fresh = self.velocity is not None and 0.0 <= velocity_age <= self._parameter(
            "velocity_timeout"
        )
        return self.velocity, fresh

    def _on_odometry(self, message: Odometry) -> None:
        now = self._now()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        velocity, velocity_ok = self._velocity_for_odometry(message, now)
        if self._parameter("velocity_source") == "patched_odometry_twist" and velocity_ok:
            self.velocity = velocity.copy()
            self.velocity_stamp = stamp
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        try:
            belief, stable = self.adapter.convert(
                stamp,
                message.header.frame_id,
                [position.x, position.y, position.z],
                [orientation.x, orientation.y, orientation.z, orientation.w],
                message.pose.covariance,
                velocity,
                now,
                message.header.frame_id == self._parameter("planning_frame"),
                velocity_ok,
            )
        except ValueError as exception:
            with self.lock:
                self.belief_stable = False
                self.adapter.detector.invalidate()
            self._publish_diagnostics(str(exception), "ERROR")
            return
        with self.lock:
            self.belief = belief
            self.belief_stable = stable

    def _on_path(self, message: Path) -> None:
        if message.header.frame_id != self._parameter("planning_frame"):
            self._publish_diagnostics("path frame mismatch", "ERROR")
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            self._publish_diagnostics("IFDS path header stamp is zero", "ERROR")
            return
        points = [
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in message.poses
        ]
        try:
            path = {
                "stamp": stamp,
                "received": self._now(),
                "polyline": Polyline(points),
                "generation": path_generation(stamp, points),
            }
        except ValueError as exception:
            self._publish_diagnostics(str(exception), "ERROR")
            return
        with self.lock:
            self.path = path

    def _on_px4_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except ValueError:
            self._publish_diagnostics("invalid PX4 status", "ERROR")
            return
        with self.lock:
            self.px4.update(status)

    def _snapshot(self) -> PlannerSnapshot:
        now = self._now()
        with self.lock:
            velocity_fresh = self._parameter("velocity_source") == "patched_odometry_twist"
            if not velocity_fresh:
                velocity_fresh = (
                    self.velocity is not None
                    and 0.0 <= now - self.velocity_stamp <= self._parameter("velocity_timeout")
                )
            return PlannerSnapshot(
                now, self.belief, self.belief_stable, self.path, dict(self.px4), velocity_fresh
            )

    def _on_planning_timer(self) -> None:
        """Low-rate planning scheduler; never performs commit or safety work."""
        snapshot = self._snapshot()
        if not self._should_request_plan(snapshot):
            return
        try:
            first = self.state in (PlannerState.BUILDING_NLP, PlannerState.FIRST_SOLVE)
            request = self._make_request(snapshot, first)
            if self.worker.submit(request, replace_pending=not first):
                self.solve_in_progress = True
                if first:
                    self.first_request_submitted = True
                    self.state = PlannerState.FIRST_SOLVE
                else:
                    self.state = PlannerState.REPLANNING
        except ValueError as exception:
            self._enter_safe_hold(str(exception), snapshot)

    def _on_commit_timer(self) -> None:
        """High-rate state, completion, commit, goal, freshness, and diagnostics loop."""
        for event in self.worker.drain_completions():
            self._on_solve_complete(event.request, event.result, event.stale)
        snapshot = self._snapshot()
        self._update_goal(snapshot)
        self._update_mission_state(snapshot)
        self._commit_candidate_if_due(snapshot)
        self._publish_diagnostics()

    def _update_mission_state(self, snapshot: PlannerSnapshot) -> None:
        if snapshot.px4.get("mode") == "FAULT":
            self.state = PlannerState.FAULT
            return
        if snapshot.px4.get("failsafe"):
            self.state = PlannerState.SAFE_HOLD
            return
        if not snapshot.px4.get("connected"):
            self.state = PlannerState.WAIT_PX4
            return
        if self.state == PlannerState.WAIT_PX4:
            self.state = PlannerState.TAKEOFF
        elif self.state == PlannerState.TAKEOFF:
            self.state = PlannerState.HOLD
        elif self.state == PlannerState.HOLD and snapshot.px4.get("hold_ready"):
            self.state = PlannerState.WAIT_BELIEF_STABLE
        elif self.state == PlannerState.WAIT_BELIEF_STABLE and snapshot.belief_stable:
            self.state = PlannerState.WAIT_IFDS_INITIAL_PATH
        elif self.state == PlannerState.WAIT_IFDS_INITIAL_PATH and self._path_fresh(snapshot):
            self.state = PlannerState.BUILDING_NLP
        if self.state in (PlannerState.EXECUTING, PlannerState.REPLANNING):
            if not self._data_fresh(snapshot):
                self._enter_safe_hold("runtime data expired", snapshot)
            elif self.buffer.remaining(snapshot.now) <= 0 and not self._goal_conditions(snapshot):
                self._enter_safe_hold("trajectory expired before goal", snapshot)

    def _path_fresh(self, snapshot: PlannerSnapshot) -> bool:
        if snapshot.path is None:
            return False
        timeout = self._parameter("path_timeout")
        receive_age = snapshot.now - snapshot.path["received"]
        stamp_age = snapshot.now - snapshot.path["stamp"]
        return 0.0 <= receive_age <= timeout and 0.0 <= stamp_age <= timeout

    def _data_fresh(self, snapshot: PlannerSnapshot) -> bool:
        return (
            snapshot.belief is not None
            and 0.0 <= snapshot.now - snapshot.belief.stamp <= self._parameter("belief_timeout")
            and snapshot.velocity_fresh
            and self._path_fresh(snapshot)
        )

    def _control_at(self, absolute_time: float) -> np.ndarray:
        active = self.buffer.active
        if active and active.commit_time <= absolute_time <= active.end_time:
            return active.sample(absolute_time)[1]
        return np.array([9.81, 0.0, 0.0, 0.0])

    def _make_request(self, snapshot: PlannerSnapshot, first: bool) -> PlanningRequest:
        if not self._data_fresh(snapshot) or snapshot.belief is None or snapshot.path is None:
            raise ValueError("cannot plan from stale snapshot")
        delay = self.delay.estimate()
        if first:
            delay = max(delay, self._parameter("cold_start_delay"))
        commit_time = snapshot.now + delay
        sigma, mean, rotation, covariance = self.delay.propagate(
            snapshot.belief.sigma_states,
            snapshot.belief.stamp,
            commit_time,
            self._control_at,
            self._parameter("process_noise_diagonal"),
        )
        references = snapshot.path["polyline"].lookahead(
            mean[:3],
            self._parameter("lookahead_count"),
            self._parameter("lookahead_spacing"),
        )
        with self.lock:
            self.request_generation += 1
            generation = self.request_generation
        return PlanningRequest(
            generation,
            snapshot.path["generation"],
            snapshot.belief.generation,
            snapshot.now,
            commit_time,
            sigma,
            mean,
            rotation,
            covariance,
            references,
            snapshot.path["polyline"],
            first,
        )

    def _should_request_plan(self, snapshot: PlannerSnapshot) -> bool:
        if self.state in (PlannerState.BUILDING_NLP, PlannerState.FIRST_SOLVE):
            return not self.first_request_submitted and not self.solve_in_progress
        if self.state not in (PlannerState.EXECUTING, PlannerState.REPLANNING):
            return False
        if self._parameter("mode") == "global":
            unchanged = snapshot.path and snapshot.path["generation"] == self.last_path_generation
            if unchanged and self.buffer.remaining(snapshot.now) > self._parameter("horizon") * 0.4:
                return False
        return True

    def _solve_request(self, request: PlanningRequest) -> dict:
        self.nlp.build()
        terminal_tolerance = self._parameter("terminal_velocity_tolerance")
        final_mode = int(
            np.linalg.norm(request.references[-1] - request.references[0])
            < self._parameter("lookahead_spacing")
        )
        velocity_reference = np.zeros(3)
        if not final_mode:
            velocity_reference = (request.references[-1] - request.references[-2]) / max(
                self._parameter("lookahead_spacing"), 0.1
            )
        bound = terminal_tolerance if final_mode else self._parameter("velocity_max")
        self.nlp.set_parameters(
            request.sigma_states,
            request.references,
            self._parameter("horizon"),
            velocity_reference,
            [-bound] * 3,
            [bound] * 3,
            final_mode,
            self._parameter("weights"),
            self._parameter("terminal_position_tolerance"),
        )
        result = self.nlp.solve()
        result["terminal_velocity_lower"] = np.full(3, -bound)
        result["terminal_velocity_upper"] = np.full(3, bound)
        result["max_lgr_dynamics_residual"] = self.nlp.compute_residual(result)
        return result

    def _on_solve_complete(self, request: PlanningRequest, result, stale: bool) -> None:
        self.solve_in_progress = False
        now = self._now()
        with self.lock:
            current_path_generation = self.path["generation"] if self.path else ""
        if (
            stale
            or request.request_generation != self.request_generation
            or request.path_generation != current_path_generation
        ):
            self.manager.stale_discards += 1
            if request.first:
                self.first_request_submitted = False
                self.state = PlannerState.BUILDING_NLP
            return
        if isinstance(result, Exception):
            if request.first:
                self.first_request_submitted = False
                self.state = PlannerState.BUILDING_NLP
            elif self.buffer.remaining(now) <= 0:
                self.state = PlannerState.SAFE_HOLD
            self._publish_diagnostics(str(result), "ERROR")
            return
        self.delay.record_solve_time(self.nlp.solve_time)
        gate = self.gate.check(result, request, self.request_generation)
        self.last_gate = gate
        accepted, reason = self.manager.admit(
            request,
            result,
            now,
            self.request_generation,
            gate,
            self._parameter("planning_frame"),
        )
        if accepted:
            self.state = PlannerState.TRAJECTORY_READY if request.first else PlannerState.REPLANNING
        elif request.first:
            self.first_request_submitted = False
            self.state = PlannerState.BUILDING_NLP
        elif reason not in ("stale", "late") and self.buffer.remaining(now) <= 0:
            self.state = PlannerState.SAFE_HOLD
        self._publish_diagnostics(reason)

    def _commit_candidate_if_due(self, snapshot: PlannerSnapshot) -> None:
        if not self.buffer.candidate_due(snapshot.now) or snapshot.belief is None:
            return
        candidate = self.buffer.candidate
        commit_status, lateness = commit_due_status(
            snapshot.now, candidate, self._parameter("allowed_commit_lateness")
        )
        self.last_commit_lateness = lateness
        if commit_status == "late":
            self.buffer.discard_candidate()
            self.manager.deadline_misses += 1
            self.first_request_submitted = False
            if self.buffer.remaining(snapshot.now) <= 0:
                self._enter_safe_hold("candidate commit was too late", snapshot)
            return
        errors = commit_continuity_errors(candidate, snapshot.belief)
        self.last_commit_errors = errors
        tolerances = (
            self._parameter("commit_position_tolerance"),
            self._parameter("commit_velocity_tolerance"),
            self._parameter("commit_attitude_tolerance"),
        )
        if all(error <= tolerance for error, tolerance in zip(errors, tolerances)):
            executable = self.buffer.commit_candidate()
            self.last_path_generation = executable.path_generation
            self.first_request_submitted = False
            self.trajectory_pub.publish(String(data=executable.to_json()))
            self.state = PlannerState.EXECUTING
            return
        self.buffer.discard_candidate()
        self.first_request_submitted = False
        if self.buffer.remaining(snapshot.now) > 0:
            self.state = PlannerState.EXECUTING
        else:
            self.state = PlannerState.SAFE_HOLD
        self._publish_diagnostics("commit belief continuity rejected", "WARN")

    def _goal_conditions(self, snapshot: PlannerSnapshot) -> bool:
        if snapshot.belief is None or snapshot.path is None:
            return False
        goal = snapshot.path["polyline"].p[-1]
        return np.linalg.norm(snapshot.belief.position - goal) <= self._parameter(
            "goal_position_tolerance"
        ) and np.linalg.norm(snapshot.belief.velocity) <= self._parameter("goal_velocity_tolerance")

    def _update_goal(self, snapshot: PlannerSnapshot) -> None:
        if self.state not in (PlannerState.EXECUTING, PlannerState.REPLANNING):
            self.goal_since = None
            return
        self.goal_since, reached = update_goal_dwell(
            self.goal_since,
            snapshot.now,
            self._goal_conditions(snapshot),
            self.buffer.remaining(snapshot.now),
            self._parameter("goal_dwell_time"),
        )
        if reached:
            self.state = PlannerState.GOAL_REACHED

    def _enter_safe_hold(self, reason: str, snapshot: PlannerSnapshot) -> None:
        self.state = PlannerState.SAFE_HOLD
        self.execution_command_pub.publish(String(data=json.dumps({"command": "SAFE_HOLD"})))
        self._publish_diagnostics(reason, "ERROR")

    def _on_resume(self, request, response):
        snapshot = self._snapshot()
        allowed = can_resume(
            self.state,
            snapshot.px4,
            snapshot.belief_stable,
            snapshot.velocity_fresh,
            self._path_fresh(snapshot),
        )
        response.success = allowed
        if allowed:
            self.first_request_submitted = False
            self.state = PlannerState.WAIT_IFDS_INITIAL_PATH
            response.message = "resume accepted; belief and path will be revalidated"
        else:
            response.message = "resume rejected: FAULT or readiness conditions not met"
        return response

    def _publish_diagnostics(self, reason: str = "", level: str = "OK") -> None:
        gate = self.last_gate
        payload = {
            "state": self.state.name,
            "level": level,
            "reason": reason,
            "cold_build_time": self.nlp.build_time,
            "parameter_update_time": self.nlp.parameter_update_time,
            "solve_time": self.nlp.solve_time,
            "solve_p90": self.delay.percentile90(),
            "extraction_time": self.nlp.extraction_time,
            "predicted_commit_delay": self.delay.estimate(),
            "deadline_miss_count": self.manager.deadline_misses,
            "stale_candidate_discard_count": self.manager.stale_discards,
            "nlp_build_count": self.nlp.build_count,
            "active_trajectory_remaining": self.buffer.remaining(self._now()),
            "commit_lateness": self.last_commit_lateness,
            "actual_commit_error": self.last_commit_errors,
            "belief_age": self._now() - self.belief.stamp if self.belief else None,
            "path_age": self._now() - self.path["stamp"] if self.path else None,
            "velocity_age": (
                self._now() - self.velocity_stamp if self.velocity is not None else None
            ),
            "velocity_clock_offset": self.velocity_clock_offset,
            "worker_pending": self.worker.pending_count(),
            "worker_solving": self.worker.solve_in_progress,
            "active_path_generation": self.last_path_generation,
            "candidate_path_generation": (
                self.buffer.candidate.path_generation if self.buffer.candidate else None
            ),
            "max_lgr_dynamics_residual": gate.max_lgr_dynamics_residual if gate else None,
            "gate_time": gate.elapsed if gate else 0.0,
            "gate_accepted": gate.accepted if gate else None,
            "gate_reasons": gate.reasons if gate else [],
            "max_dense_mean_path_error": gate.max_dense_mean_path_error if gate else None,
            "max_dense_sigma_path_error": gate.max_dense_sigma_path_error if gate else None,
            "max_dense_velocity": gate.max_dense_velocity if gate else None,
            "max_dense_attitude": gate.max_dense_attitude if gate else None,
            "rollout_endpoint_error": gate.rollout_endpoint_error if gate else None,
            "first_request_submitted": self.first_request_submitted,
            "solve_in_progress": self.solve_in_progress,
        }
        self.diagnostics_pub.publish(String(data=json.dumps(payload)))

    def destroy_node(self):
        self.worker.shutdown()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UTOPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

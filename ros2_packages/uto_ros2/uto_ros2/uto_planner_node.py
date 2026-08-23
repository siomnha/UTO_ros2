"""ROS wiring for belief/path snapshots, background solve, admission, and commit."""

from dataclasses import dataclass
import json
import threading
import time
from typing import Optional
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from .belief_adapter import Belief
from .ifds_path_adapter import Polyline, path_generation
from .ifds_contract import (
    PathPairCache,
    PathStatus,
    path_pair_timeout_events,
    status_matches_path,
)
from .math_utils import align_enu_velocity, ned_to_enu
from .planner_runtime import (
    CommitOutcome,
    ExecutionDecision,
    LatestWinsWorker,
    PLANNER_PARAMETER_DEFAULTS,
    PlannerState,
    PlanningRequest,
    TimingBreakdown,
    WorkerResult,
    build_runtime_components,
    can_restart_mission,
    can_resume,
    commit_continuity_errors,
    commit_due_status,
    execution_decision,
    goal_generation_changed,
    mission_goal_input_valid,
    mission_goal_satisfied,
    mission_goal_yaw_and_generation,
    planning_data_fresh,
    terminal_goal_data_fresh,
    invalid_ifds_path_requires_hold,
    ifds_no_path_transition,
    update_goal_dwell,
)


@dataclass(frozen=True)
class MissionGoal:
    stamp: float
    position: np.ndarray
    yaw: Optional[float]
    generation: str


@dataclass(frozen=True)
class PlannerSnapshot:
    now: float
    belief: Optional[Belief]
    belief_stable: bool
    path: Optional[dict]
    mission_goal: Optional[MissionGoal]
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
        self.path_status = None
        self.path_pairs = PathPairCache()
        self.path_pair_timeouts = 0
        self.last_path_pair_timeout = None
        self.ifds_terminal_ready = False
        self.hold_current_requested = False
        self.mission_goal = None
        self.goal_restart_pending = False
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
        self.last_timing = None
        self.last_completion_queue_time = 0.0
        self.last_request_to_commit_time = 0.0
        self.candidate_request_time = None
        self.pending_safety_reason = ""
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
            String,
            self._parameter("path_status_topic"),
            self._on_path_status,
            10,
        )
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped,
            self._parameter("mission_goal_topic"),
            self._on_mission_goal,
            goal_qos,
        )
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
            stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            pending_path = {
                "stamp": stamp,
                "stamp_ns": stamp_ns,
                "received": self._now(),
                "polyline": Polyline(points),
                "generation": path_generation(
                    stamp, points, self._parameter("path_generation_resolution")
                ),
            }
        except ValueError as exception:
            self._publish_diagnostics(str(exception), "ERROR")
            return
        with self.lock:
            pair = self.path_pairs.add_path(stamp_ns, pending_path, self._now())
            if pair is not None:
                self._activate_path_pair(pair)

    def _on_path_status(self, message: String) -> None:
        try:
            status = PathStatus.from_json(message.data)
        except (ValueError, TypeError, json.JSONDecodeError) as exception:
            self._invalidate_ifds_path(f"INVALID_PATH_STATUS:{exception}")
            return
        if status.terminal:
            self._handle_no_path_transition(status, terminal=True)
            return
        if not status.valid:
            if status.reason == "NEW_GOAL_PENDING":
                self._handle_no_path_transition(status, terminal=False)
                return
            self._invalidate_ifds_path(status.reason)
            return
        with self.lock:
            pair = self.path_pairs.add_status(status, self._now())
            if pair is not None:
                self._activate_path_pair(pair)

    def _activate_path_pair(self, pair) -> None:
        """Activate one matched Path/status pair atomically while holding ``lock``."""
        path = pair.path
        path["generation"] = (
            f"ifds:{pair.status.goal_generation}:"
            f"{pair.status.obstacle_generation}:{pair.status.path_generation}"
        )
        path["status"] = pair.status
        self.path = path
        self.path_status = pair.status
        self.ifds_terminal_ready = False

    def _expire_path_pairs(self, now: float) -> None:
        with self.lock:
            events = path_pair_timeout_events(
                self.path_pairs, now, self._parameter("path_pair_timeout")
            )
            self.path_pair_timeouts += len(events)
            if events:
                self.last_path_pair_timeout = events[-1]
        for event in events:
            self._publish_diagnostics(
                f"{event.reason}:stamp={event.expired_stamp_ns}", "WARN"
            )

    def _handle_no_path_transition(self, status: PathStatus, terminal: bool) -> None:
        """Invalidate old planning work without deleting the bridge hold reference."""
        with self.lock:
            update = ifds_no_path_transition(
                self.state, terminal, self.request_generation
            )
            self.ifds_terminal_ready = update.terminal_ready
            self.path = None
            self.path_status = status
            self.path_pairs.clear()
            self.buffer.discard_candidate()
            self.request_generation = update.request_generation
            self.first_request_submitted = False
            self.last_path_generation = ""
            self.goal_since = None
            self.hold_current_requested = update.publish_hold_current
            if self.state not in (PlannerState.FAULT, PlannerState.SAFE_HOLD):
                self.state = update.next_state
        if update.publish_hold_current:
            self.execution_command_pub.publish(
                String(data=json.dumps({"command": "HOLD_CURRENT"}))
            )
        self._publish_diagnostics(status.reason, "INFO" if terminal else "WARN")

    def _invalidate_ifds_path(self, reason: str) -> None:
        now = self._now()
        with self.lock:
            previous = self.path_status
            self.path_status = PathStatus(
                False,
                0,
                previous.path_generation if previous else 0,
                previous.goal_generation if previous else 0,
                previous.obstacle_generation if previous else 0,
                now,
                now,
                reason or "NO_VALID_IFDS_PATH",
            )
            self.path_pairs.clear()
            self.path = None
            self.buffer.discard_candidate()
            self.request_generation += 1
            active_state = invalid_ifds_path_requires_hold(self.state)
            if not active_state and self.state not in (PlannerState.FAULT, PlannerState.SAFE_HOLD):
                self.state = PlannerState.WAIT_IFDS_INITIAL_PATH
        if active_state:
            self._enter_safe_hold(f"IFDS path invalid: {reason}", self._snapshot())
        else:
            self._publish_diagnostics(f"IFDS path invalid: {reason}", "WARN")

    def _on_mission_goal(self, message: PoseStamped) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        point = message.pose.position
        position = np.array([point.x, point.y, point.z], dtype=float)
        if not mission_goal_input_valid(
            stamp,
            self._now(),
            message.header.frame_id,
            self._parameter("planning_frame"),
            position,
            self._parameter("mission_goal_timeout"),
        ):
            self._publish_diagnostics(
                "invalid/stale mission goal timestamp/frame/position", "ERROR"
            )
            return
        orientation = message.pose.orientation
        try:
            yaw, generation = mission_goal_yaw_and_generation(
                position,
                [orientation.x, orientation.y, orientation.z, orientation.w],
                self._parameter("goal_yaw_enabled"),
            )
        except ValueError as exception:
            self._publish_diagnostics(str(exception), "ERROR")
            return
        if not np.all(np.isfinite(position)) or (yaw is not None and not np.isfinite(yaw)):
            self._publish_diagnostics("non-finite mission goal", "ERROR")
            return
        with self.lock:
            changed = goal_generation_changed(self.mission_goal, generation)
            self.mission_goal = MissionGoal(stamp, position, yaw, generation)
            if changed:
                self.goal_since = None
                self.goal_restart_pending = self.state == PlannerState.GOAL_REACHED
        if changed:
            self._invalidate_planning_for_mission_goal()

    def _invalidate_planning_for_mission_goal(self) -> None:
        """Make mission-goal/status cross-topic ordering safe and idempotent."""
        with self.lock:
            was_executing = self.state in (
                PlannerState.TRAJECTORY_READY,
                PlannerState.EXECUTING,
                PlannerState.REPLANNING,
            )
            planning_started = was_executing or self.state in (
                PlannerState.BUILDING_NLP,
                PlannerState.FIRST_SOLVE,
            )
            self.path = None
            self.path_pairs.clear()
            self.buffer.discard_candidate()
            self.request_generation += 1
            self.first_request_submitted = False
            self.last_path_generation = ""
            if was_executing:
                self.hold_current_requested = True
            if planning_started and self.state not in (
                PlannerState.FAULT,
                PlannerState.SAFE_HOLD,
                PlannerState.GOAL_REACHED,
            ):
                self.state = (
                    PlannerState.HOLD
                    if self.ifds_terminal_ready
                    else PlannerState.WAIT_IFDS_INITIAL_PATH
                )
        if was_executing:
            self.execution_command_pub.publish(
                String(data=json.dumps({"command": "HOLD_CURRENT"}))
            )

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
                now,
                self.belief,
                self.belief_stable,
                self.path,
                self.mission_goal,
                dict(self.px4),
                velocity_fresh,
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
        """Make exactly one commit/continue/hold/goal execution decision."""
        for event in self.worker.drain_completions():
            self._on_solve_complete(event.request, event.result, event.stale)
        snapshot = self._snapshot()
        self._expire_path_pairs(snapshot.now)
        self._update_mission_state(snapshot)
        if self.state in (PlannerState.FAULT, PlannerState.SAFE_HOLD):
            self._publish_diagnostics()
            return
        outcome = self._commit_candidate_if_due(snapshot)
        if outcome != CommitOutcome.COMMITTED:
            self._update_goal(snapshot)
        goal_dwell_pending = self._goal_conditions(snapshot)
        decision = execution_decision(
            outcome,
            self._planning_data_fresh(snapshot),
            self.buffer.remaining(snapshot.now),
            self.state == PlannerState.GOAL_REACHED,
            goal_dwell_pending,
            self._terminal_goal_data_fresh(snapshot),
        )
        if decision == ExecutionDecision.SAFE_HOLD and self.state in (
            PlannerState.EXECUTING,
            PlannerState.REPLANNING,
            PlannerState.TRAJECTORY_READY,
        ):
            self._enter_safe_hold(
                self.pending_safety_reason
                or "no committable candidate and active trajectory expired",
                snapshot,
            )
        if decision != ExecutionDecision.SAFE_HOLD:
            self.pending_safety_reason = ""
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

    def _path_fresh(self, snapshot: PlannerSnapshot) -> bool:
        if snapshot.path is None:
            return False
        status = snapshot.path.get("status") if snapshot.path else None
        if (
            status is None
            or not status.valid
            or not status_matches_path(status, snapshot.path["stamp_ns"], snapshot.now)
        ):
            return False
        timeout = self._parameter("path_timeout")
        receive_age = snapshot.now - snapshot.path["received"]
        stamp_age = snapshot.now - snapshot.path["stamp"]
        return 0.0 <= receive_age <= timeout and 0.0 <= stamp_age <= timeout

    def _belief_age(self, snapshot: PlannerSnapshot) -> float:
        return np.inf if snapshot.belief is None else snapshot.now - snapshot.belief.stamp

    def _mission_goal_valid(self, snapshot: PlannerSnapshot) -> bool:
        goal = snapshot.mission_goal
        if goal is None:
            return False
        timeout = self._parameter("mission_goal_timeout")
        age = snapshot.now - goal.stamp
        return age >= 0.0 and (timeout <= 0.0 or age <= timeout)

    def _planning_data_fresh(self, snapshot: PlannerSnapshot) -> bool:
        return planning_data_fresh(
            self._belief_age(snapshot),
            self._parameter("belief_timeout"),
            snapshot.velocity_fresh,
            self._path_fresh(snapshot),
        )

    def _terminal_goal_data_fresh(self, snapshot: PlannerSnapshot) -> bool:
        return terminal_goal_data_fresh(
            self._belief_age(snapshot),
            self._parameter("belief_timeout"),
            snapshot.velocity_fresh,
            self._mission_goal_valid(snapshot),
        )

    def _control_at(self, absolute_time: float) -> np.ndarray:
        if self.hold_current_requested:
            return np.array([9.81, 0.0, 0.0, 0.0])
        active = self.buffer.active
        if active and active.commit_time <= absolute_time <= active.end_time:
            return active.sample(absolute_time)[1]
        return np.array([9.81, 0.0, 0.0, 0.0])

    def _make_request(self, snapshot: PlannerSnapshot, first: bool) -> PlanningRequest:
        if not self._planning_data_fresh(snapshot) or snapshot.belief is None or snapshot.path is None:
            raise ValueError("cannot plan from stale snapshot")
        delay = self.delay.estimate(first, self._parameter("cold_start_delay"))
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
            time.perf_counter(),
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

    def _solve_request(self, request: PlanningRequest) -> WorkerResult:
        worker_start = time.perf_counter()
        prepare_start = worker_start
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
        prepare_time = time.perf_counter() - prepare_start
        result = self.nlp.solve()
        result["terminal_velocity_lower"] = np.full(3, -bound)
        result["terminal_velocity_upper"] = np.full(3, bound)
        result["max_lgr_dynamics_residual"] = self.nlp.compute_residual(result)
        gate = self.gate.check(result, request, request.request_generation)
        completed = time.perf_counter()
        timing = TimingBreakdown(
            enqueue_time=request.enqueue_monotonic,
            worker_start_time=worker_start,
            nlp_prepare_time=prepare_time,
            ipopt_solve_time=self.nlp.solve_time,
            extraction_time=self.nlp.extraction_time,
            dense_gate_time=gate.elapsed,
            worker_completion_time=completed,
            worker_total_time=completed - worker_start,
            cold_build_time=self.nlp.build_time,
            nlp_build_count=self.nlp.build_count,
        )
        return WorkerResult(result, gate, timing)

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
                self.state = (
                    PlannerState.BUILDING_NLP
                    if self.path is not None
                    else PlannerState.WAIT_IFDS_INITIAL_PATH
                )
            return
        if isinstance(result, Exception):
            if request.first:
                self.first_request_submitted = False
                self.state = (
                    PlannerState.BUILDING_NLP
                    if self.path is not None
                    else PlannerState.WAIT_IFDS_INITIAL_PATH
                )
            elif self.buffer.remaining(now) <= 0:
                self.pending_safety_reason = "planner worker failed"
            self._publish_diagnostics(str(result), "ERROR")
            return
        if not isinstance(result, WorkerResult):
            raise TypeError("worker returned an invalid completion payload")
        consumed = time.perf_counter()
        self.last_timing = result.timing
        self.last_completion_queue_time = max(0.0, consumed - result.timing.worker_completion_time)
        admission_latency = max(0.0, consumed - result.timing.enqueue_time)
        self.delay.record_latency(admission_latency)
        gate = result.gate_result
        solve_result = result.solve_result
        self.last_gate = gate
        accepted, reason = self.manager.admit(
            request,
            solve_result,
            now,
            self.request_generation,
            gate,
            self._parameter("planning_frame"),
        )
        if accepted:
            self.candidate_request_time = request.request_time
            self.state = PlannerState.TRAJECTORY_READY if request.first else PlannerState.REPLANNING
        elif request.first:
            self.first_request_submitted = False
            self.state = (
                PlannerState.BUILDING_NLP
                if self.path is not None
                else PlannerState.WAIT_IFDS_INITIAL_PATH
            )
        elif reason not in ("stale", "late") and self.buffer.remaining(now) <= 0:
            self.pending_safety_reason = "candidate admission failed"
        self._publish_diagnostics(reason)

    def _commit_candidate_if_due(self, snapshot: PlannerSnapshot) -> CommitOutcome:
        candidate = self.buffer.candidate
        if candidate is None:
            return CommitOutcome.NONE
        status, lateness = commit_due_status(
            snapshot.now, candidate, self._parameter("allowed_commit_lateness")
        )
        self.last_commit_lateness = lateness
        if status == "waiting":
            return CommitOutcome.WAITING
        if status == "late":
            self.buffer.discard_candidate()
            self.manager.deadline_misses += 1
            self.first_request_submitted = False
            return CommitOutcome.LATE
        if snapshot.belief is None or not self._planning_data_fresh(snapshot):
            self.buffer.discard_candidate()
            self.first_request_submitted = False
            return CommitOutcome.REJECTED
        errors = commit_continuity_errors(candidate, snapshot.belief)
        self.last_commit_errors = errors
        tolerances = (
            self._parameter("commit_position_tolerance"),
            self._parameter("commit_velocity_tolerance"),
            self._parameter("commit_attitude_tolerance"),
        )
        if not all(error <= tolerance for error, tolerance in zip(errors, tolerances)):
            self.buffer.discard_candidate()
            self.first_request_submitted = False
            return CommitOutcome.REJECTED
        executable = self.buffer.commit_candidate()
        self.last_path_generation = executable.path_generation
        self.first_request_submitted = False
        self.trajectory_pub.publish(String(data=executable.to_json()))
        if self.candidate_request_time is not None:
            self.last_request_to_commit_time = max(0.0, snapshot.now - self.candidate_request_time)
        self.candidate_request_time = None
        self.state = PlannerState.EXECUTING
        self.hold_current_requested = False
        return CommitOutcome.COMMITTED

    def _goal_conditions(self, snapshot: PlannerSnapshot) -> bool:
        goal = snapshot.mission_goal
        if (
            snapshot.belief is None
            or goal is None
            or (self.buffer.active is None and not self.ifds_terminal_ready)
            or not self._terminal_goal_data_fresh(snapshot)
        ):
            return False
        timeout = self._parameter("mission_goal_timeout")
        if timeout > 0.0 and not 0.0 <= snapshot.now - goal.stamp <= timeout:
            return False
        return mission_goal_satisfied(
            snapshot.belief,
            goal,
            self._parameter("goal_position_tolerance"),
            self._parameter("goal_velocity_tolerance"),
            self._parameter("goal_yaw_enabled"),
            self._parameter("goal_yaw_tolerance"),
        )

    def _update_goal(self, snapshot: PlannerSnapshot) -> None:
        goal_states = (
            PlannerState.HOLD,
            PlannerState.WAIT_BELIEF_STABLE,
            PlannerState.WAIT_IFDS_INITIAL_PATH,
            PlannerState.EXECUTING,
            PlannerState.REPLANNING,
        )
        if self.state not in goal_states:
            self.goal_since = None
            return
        if not self._terminal_goal_data_fresh(snapshot) or snapshot.px4.get("failsafe"):
            self.goal_since = None
            return
        self.goal_since, reached = update_goal_dwell(
            self.goal_since,
            snapshot.now,
            self._goal_conditions(snapshot),
            0.0 if self.ifds_terminal_ready else self.buffer.remaining(snapshot.now),
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
        restart_goal = can_restart_mission(
            self.state,
            self.goal_restart_pending,
            snapshot.px4,
            self._mission_goal_valid(snapshot),
            snapshot.belief is not None,
            snapshot.belief_stable,
            snapshot.velocity_fresh,
            self._path_fresh(snapshot),
        )
        allowed = can_resume(
            self.state,
            snapshot.px4,
            snapshot.belief_stable,
            snapshot.velocity_fresh,
            self._path_fresh(snapshot),
        )
        allowed = allowed or restart_goal
        response.success = allowed
        if allowed:
            self.goal_restart_pending = False
            self.goal_since = None
            self.buffer.discard_candidate()
            self.first_request_submitted = False
            self.state = PlannerState.WAIT_BELIEF_STABLE
            response.message = "resume accepted; belief and path will be revalidated"
        else:
            response.message = "resume rejected: FAULT or readiness conditions not met"
        return response

    def _publish_diagnostics(self, reason: str = "", level: str = "OK") -> None:
        gate = self.last_gate
        first_delay = self.buffer.active is None and self.state in (
            PlannerState.BUILDING_NLP,
            PlannerState.FIRST_SOLVE,
            PlannerState.TRAJECTORY_READY,
        )
        projection = self.delay.last_projection
        payload = {
            "state": self.state.name,
            "level": level,
            "reason": reason,
            "cold_build_time": self.last_timing.cold_build_time if self.last_timing else 0.0,
            "parameter_update_time": self.last_timing.nlp_prepare_time if self.last_timing else 0.0,
            "solve_time": self.last_timing.ipopt_solve_time if self.last_timing else 0.0,
            "solve_p90": self.delay.percentile90(),  # deprecated compatibility field
            "admission_latency_p90": self.delay.percentile90(),
            "estimated_planning_delay": self.delay.estimate(
                first_delay, self._parameter("cold_start_delay")
            ),
            "delay_estimate_mode": self.delay.estimate_mode(first_delay),
            "latency_sample_count": len(self.delay.samples),
            "latency_window_size": self.delay.window,
            "latency_min_samples": self.delay.minimum_samples,
            "cold_start_delay": self._parameter("cold_start_delay"),
            "initial_delay": self._parameter("initial_delay"),
            "queue_time": (
                self.last_timing.worker_start_time - self.last_timing.enqueue_time
                if self.last_timing
                else 0.0
            ),
            "nlp_prepare_time": self.last_timing.nlp_prepare_time if self.last_timing else 0.0,
            "ipopt_solve_time": self.last_timing.ipopt_solve_time if self.last_timing else 0.0,
            "extraction_time": self.last_timing.extraction_time if self.last_timing else 0.0,
            "dense_gate_time": self.last_timing.dense_gate_time if self.last_timing else 0.0,
            "worker_total_time": self.last_timing.worker_total_time if self.last_timing else 0.0,
            "completion_queue_time": self.last_completion_queue_time,
            "request_to_commit_time": self.last_request_to_commit_time,
            "deadline_miss_count": self.manager.deadline_misses,
            "stale_candidate_discard_count": self.manager.stale_discards,
            "nlp_build_count": self.last_timing.nlp_build_count if self.last_timing else 0,
            "active_trajectory_remaining": self.buffer.remaining(self._now()),
            "commit_lateness": self.last_commit_lateness,
            "actual_commit_error": self.last_commit_errors,
            "belief_age": self._now() - self.belief.stamp if self.belief else None,
            "path_age": self._now() - self.path["stamp"] if self.path else None,
            "ifds_path_status_valid": self.path_status.valid if self.path_status else None,
            "ifds_path_status_reason": self.path_status.reason if self.path_status else None,
            "ifds_path_status_terminal": self.path_status.terminal if self.path_status else None,
            "hold_current_requested": self.hold_current_requested,
            "ifds_path_valid_until": self.path_status.valid_until if self.path_status else None,
            "ifds_obstacle_generation": (
                self.path_status.obstacle_generation if self.path_status else None
            ),
            "path_pair_timeout_count": self.path_pair_timeouts,
            "path_pair_timeout": (
                {
                    "reason": self.last_path_pair_timeout.reason,
                    "expired_stamp_ns": self.last_path_pair_timeout.expired_stamp_ns,
                    "pending_path_count": self.last_path_pair_timeout.pending_path_count,
                    "pending_status_count": self.last_path_pair_timeout.pending_status_count,
                }
                if self.last_path_pair_timeout
                else None
            ),
            "pending_path_count": len(self.path_pairs.paths),
            "pending_path_status_count": len(self.path_pairs.statuses),
            "velocity_age": (
                self._now() - self.velocity_stamp if self.velocity is not None else None
            ),
            "velocity_clock_offset": self.velocity_clock_offset,
            "worker_pending": self.worker.pending_count(),
            "worker_solving": self.worker.solve_in_progress,
            "active_path_generation": self.last_path_generation,
            "mission_goal_generation": (
                self.mission_goal.generation if self.mission_goal else None
            ),
            "mission_goal_yaw_enabled": self._parameter("goal_yaw_enabled"),
            "mission_goal_has_yaw": self.mission_goal is not None
            and self.mission_goal.yaw is not None,
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
            "max_dense_endpoint_position_error": (
                gate.max_dense_endpoint_position_error if gate else None
            ),
            "max_dense_endpoint_velocity_error": (
                gate.max_dense_endpoint_velocity_error if gate else None
            ),
            "max_dense_endpoint_attitude_error": (
                gate.max_dense_endpoint_attitude_error if gate else None
            ),
            "first_request_submitted": self.first_request_submitted,
            "solve_in_progress": self.solve_in_progress,
            "sigma_projection": (
                {
                    "relative_covariance_error": projection.relative_covariance_error,
                    "position_velocity_cross_error": projection.position_velocity_cross_error,
                    "attitude_velocity_cross_error": projection.attitude_velocity_cross_error,
                    "target_rank": projection.target_rank,
                    "reconstructed_rank": projection.reconstructed_rank,
                    "discarded_eigenvalue_energy": projection.discarded_eigenvalue_energy,
                }
                if projection
                else None
            ),
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

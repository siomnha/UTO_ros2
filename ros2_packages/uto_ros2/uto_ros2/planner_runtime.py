"""Planner state, delay prediction, feasibility admission, and worker utilities."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import threading
import queue
import time
from typing import Callable
import numpy as np
from .belief_adapter import joint_sigma_process_update
from .dynamics import rk4
from .lgr import interpolate_control
from .ifds_path_adapter import Polyline
from .math_utils import euler_to_rot, so3_log
from .trajectory import Trajectory, TrajectoryBuffer


class PlannerState(Enum):
    WAIT_PX4 = auto()
    TAKEOFF = auto()
    HOLD = auto()
    WAIT_BELIEF_STABLE = auto()
    WAIT_IFDS_INITIAL_PATH = auto()
    BUILDING_NLP = auto()
    FIRST_SOLVE = auto()
    TRAJECTORY_READY = auto()
    EXECUTING = auto()
    REPLANNING = auto()
    GOAL_REACHED = auto()
    SAFE_HOLD = auto()
    FAULT = auto()


class ExecutionDecision(Enum):
    COMMIT = auto()
    CONTINUE = auto()
    SAFE_HOLD = auto()
    GOAL_REACHED = auto()


class CommitOutcome(Enum):
    NONE = auto()
    WAITING = auto()
    COMMITTED = auto()
    REJECTED = auto()
    LATE = auto()


def invalid_ifds_path_requires_hold(state: PlannerState) -> bool:
    return state in (
        PlannerState.TRAJECTORY_READY,
        PlannerState.EXECUTING,
        PlannerState.REPLANNING,
    )


@dataclass(frozen=True)
class IfdsNoPathUpdate:
    next_state: PlannerState
    request_generation: int
    terminal_ready: bool
    publish_hold_current: bool


def ifds_no_path_transition(
    state: PlannerState, terminal: bool, request_generation: int = 0
) -> IfdsNoPathUpdate:
    """Describe the atomic invalidation required by terminal/new-goal statuses."""
    executing = state in (
        PlannerState.TRAJECTORY_READY,
        PlannerState.EXECUTING,
        PlannerState.REPLANNING,
    )
    readiness_states = (
        PlannerState.WAIT_PX4,
        PlannerState.TAKEOFF,
        PlannerState.HOLD,
        PlannerState.WAIT_BELIEF_STABLE,
    )
    if state in readiness_states or state in (PlannerState.FAULT, PlannerState.SAFE_HOLD):
        next_state = state
    elif state == PlannerState.GOAL_REACHED:
        next_state = state
    else:
        next_state = PlannerState.HOLD if terminal else PlannerState.WAIT_IFDS_INITIAL_PATH
    return IfdsNoPathUpdate(
        next_state, request_generation + 1, terminal, executing
    )


@dataclass(frozen=True)
class PlanningRequest:
    request_generation: int
    path_generation: str
    belief_generation: int
    request_time: float
    commit_time: float
    sigma_states: np.ndarray
    predicted_mean: np.ndarray
    predicted_rotation: np.ndarray
    predicted_covariance: np.ndarray
    references: np.ndarray
    path: Polyline
    first: bool = False
    enqueue_monotonic: float = 0.0
    terminal_segment: bool = False
    commit_time_finalized: bool = True


def terminal_segment_matches_goal(
    references, mission_goal_position, tolerance: float
) -> bool:
    """Identify a final segment from the explicit mission goal, never spacing heuristics."""
    if mission_goal_position is None or tolerance < 0.0:
        return False
    references = np.asarray(references, dtype=float)
    goal = np.asarray(mission_goal_position, dtype=float)
    return bool(
        references.ndim == 2
        and len(references) > 0
        and goal.shape == (3,)
        and np.all(np.isfinite(references[-1]))
        and np.all(np.isfinite(goal))
        and np.linalg.norm(references[-1] - goal) <= tolerance
    )


def global_preflight_ready(
    px4: dict,
    belief_stable: bool,
    path_fresh: bool,
    mission_goal_valid: bool,
) -> bool:
    """Require the aircraft to be connected and holding before a global solve."""
    return bool(
        px4.get("connected")
        and px4.get("hold_ready")
        and not px4.get("failsafe")
        and belief_stable
        and path_fresh
        and mission_goal_valid
    )


def select_planning_initial_state(belief, delay_enabled: bool, delayed_factory):
    """Freeze the current belief when delay compensation is disabled."""
    if delay_enabled:
        return delayed_factory()
    return (
        belief.sigma_states.copy(),
        belief.mean_state.copy(),
        belief.rotation.copy(),
        belief.covariance.copy(),
    )


def global_post_solve_commit_time(completion_time: float, lead_time: float) -> float:
    """Schedule a global trajectory only after solve/gate completion."""
    if not np.isfinite(completion_time) or not np.isfinite(lead_time) or lead_time < 0.0:
        raise ValueError("global commit time inputs must be finite and lead must be nonnegative")
    return float(completion_time + lead_time)


def global_one_shot_plan_allowed(trajectory_committed: bool) -> bool:
    """A new mission resets the flag; a committed mission cannot replan in flight."""
    return not trajectory_committed


def px4_status_payload(**values) -> dict:
    """Normalize bridge diagnostics to JSON-safe scalar values."""
    payload = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            value = value.item()
        payload[str(key)] = value
    return payload


@dataclass(frozen=True)
class TimingBreakdown:
    enqueue_time: float
    worker_start_time: float
    nlp_prepare_time: float
    ipopt_solve_time: float
    extraction_time: float
    dense_gate_time: float
    worker_completion_time: float
    worker_total_time: float
    cold_build_time: float = 0.0
    nlp_build_count: int = 0

    def __post_init__(self):
        values = (
            self.enqueue_time,
            self.worker_start_time,
            self.nlp_prepare_time,
            self.ipopt_solve_time,
            self.extraction_time,
            self.dense_gate_time,
            self.worker_completion_time,
            self.worker_total_time,
            self.cold_build_time,
            self.nlp_build_count,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("timing values must be finite and nonnegative")


@dataclass(frozen=True)
class WorkerResult:
    solve_result: dict
    gate_result: object
    timing: TimingBreakdown


@dataclass
class GateConfig:
    velocity_max: float = 4.0
    angle_max: float = 0.6
    control_min: tuple = (0.0, -0.48, -0.48, -1.2)
    control_max: tuple = (18.0, 0.48, 0.48, 1.2)
    terminal_position_tolerance: float = 0.3
    path_tube: float = 0.8
    sigma_path_tube: float = 1.0
    start_position_tolerance: float = 0.5
    start_velocity_tolerance: float = 0.7
    minimum_horizon: float = 0.2
    residual_limit: float = 1e-4
    dense_points_per_region: int = 15
    rollout_endpoint_position_tolerance: float = 0.3
    rollout_endpoint_velocity_tolerance: float = 0.5
    rollout_endpoint_attitude_tolerance: float = 0.2


@dataclass
class GateResult:
    accepted: bool
    reasons: list
    elapsed: float
    max_path_error: float
    max_sigma_path_error: float
    max_lgr_dynamics_residual: float
    max_dense_mean_path_error: float
    max_dense_sigma_path_error: float
    max_dense_velocity: float
    max_dense_attitude: float
    max_dense_endpoint_position_error: float
    max_dense_endpoint_velocity_error: float
    max_dense_endpoint_attitude_error: float


class FeasibilityGate:
    def __init__(self, config: GateConfig) -> None:
        self.config = config

    def check(self, result: dict, request: PlanningRequest, current_generation: int) -> GateResult:
        started = time.perf_counter()
        reasons = []
        states = np.asarray(result.get("states_physical"))
        controls = np.asarray(result.get("controls_physical"))
        sigma = np.asarray(result.get("sigma_states_physical"))
        times = np.asarray(result.get("times"))
        status = str(result.get("stats", {}).get("return_status", ""))
        success = bool(result.get("stats", {}).get("success", False))
        success = success or status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        if not success:
            reasons.append("solver status")
        if states.ndim != 2 or states.shape[1:] != (9,):
            reasons.append("state shape")
        if controls.ndim != 2 or controls.shape[1:] != (4,):
            reasons.append("control shape")
        if sigma.ndim != 3 or sigma.shape[1:] != (7, 9):
            reasons.append("sigma shape")
        if not all(np.all(np.isfinite(value)) for value in (states, controls, sigma, times)):
            reasons.append("non-finite")
        if len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0):
            reasons.append("time monotonicity")
        elif times[-1] < self.config.minimum_horizon:
            reasons.append("time coverage")
        if request.request_generation != current_generation:
            reasons.append("stale generation")
        residual = float(result.get("max_lgr_dynamics_residual", np.inf))
        if residual > self.config.residual_limit:
            reasons.append("LGR dynamics residual")
        if states.size:
            if np.max(np.abs(states[:, 3:6])) > self.config.velocity_max + 1e-5:
                reasons.append("velocity bound")
            if np.max(np.abs(states[:, 6:8])) > self.config.angle_max + 1e-5:
                reasons.append("attitude bound")
            if (
                np.linalg.norm(states[0, :3] - request.predicted_mean[:3])
                > self.config.start_position_tolerance
            ):
                reasons.append("start position")
            if (
                np.linalg.norm(states[0, 3:6] - request.predicted_mean[3:6])
                > self.config.start_velocity_tolerance
            ):
                reasons.append("start velocity")
            if (
                np.linalg.norm(states[-1, :3] - request.references[-1])
                > self.config.terminal_position_tolerance
            ):
                reasons.append("terminal position")
            velocity_lower = result.get("terminal_velocity_lower")
            velocity_upper = result.get("terminal_velocity_upper")
            if velocity_lower is not None and np.any(
                states[-1, 3:6] < np.asarray(velocity_lower) - 1e-5
            ):
                reasons.append("terminal velocity lower")
            if velocity_upper is not None and np.any(
                states[-1, 3:6] > np.asarray(velocity_upper) + 1e-5
            ):
                reasons.append("terminal velocity upper")
        if controls.size:
            if np.any(controls < np.asarray(self.config.control_min) - 1e-5):
                reasons.append("control lower bound")
            if np.any(controls > np.asarray(self.config.control_max) + 1e-5):
                reasons.append("control upper bound")
        mean_error = max(
            (request.path.project(point)[2] for point in states[:, :3]), default=np.inf
        )
        sigma_error = max(
            (request.path.project(point)[2] for point in sigma[:, :, :3].reshape(-1, 3)),
            default=np.inf,
        )
        if mean_error > self.config.path_tube:
            reasons.append("mean path tube")
        if sigma_error > self.config.sigma_path_tube:
            reasons.append("sigma path tube")
        dense = self._dense_rollout(result, request)
        if dense[0] > self.config.path_tube:
            reasons.append("dense mean path tube")
        if dense[1] > self.config.sigma_path_tube:
            reasons.append("dense sigma path tube")
        if dense[2] > self.config.velocity_max + 1e-5:
            reasons.append("dense velocity bound")
        if dense[3] > self.config.angle_max + 1e-5:
            reasons.append("dense attitude bound")
        if dense[4] > self.config.rollout_endpoint_position_tolerance:
            reasons.append("rollout endpoint position")
        if dense[5] > self.config.rollout_endpoint_velocity_tolerance:
            reasons.append("rollout endpoint velocity")
        if dense[6] > self.config.rollout_endpoint_attitude_tolerance:
            reasons.append("rollout endpoint attitude")
        if dense[7]:
            reasons.append("dense control bound")
        return GateResult(
            not reasons,
            reasons,
            time.perf_counter() - started,
            mean_error,
            sigma_error,
            residual,
            *dense[:7],
        )

    def _dense_rollout(self, result: dict, request: PlanningRequest) -> tuple:
        controls = result.get("physical_control_blocks")
        endpoints = result.get("region_endpoint_sigma_physical")
        nodes = result.get("lgr_nodes")
        regions = int(result.get("regions", 0))
        horizon = float(result.get("horizon", 0.0))
        if controls is None or endpoints is None or nodes is None or regions < 1 or horizon <= 0:
            return (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, True)
        states = np.asarray(request.sigma_states, dtype=float).T.copy()
        duration = horizon / regions
        points = self.config.dense_points_per_region
        metrics = [0.0] * 7
        control_violation = False
        for region in range(regions):
            step = duration / points
            queries = -1.0 + (np.arange(points) + 0.5) * 2.0 / points
            dense_controls = interpolate_control(nodes, controls[region], queries)
            for index in range(points):
                control = dense_controls[:, index]
                states = np.stack([rk4(state, control, step) for state in states])
                if not np.all(np.isfinite(states)) or not np.all(np.isfinite(control)):
                    return (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, True)
                control_violation |= bool(
                    np.any(control < np.asarray(self.config.control_min) - 1e-5)
                    or np.any(control > np.asarray(self.config.control_max) + 1e-5)
                )
                metrics[0] = max(metrics[0], request.path.project(states[:, :3].mean(axis=0))[2])
                metrics[1] = max(metrics[1], max(request.path.project(p)[2] for p in states[:, :3]))
                metrics[2] = max(metrics[2], float(np.max(np.abs(states[:, 3:6]))))
                metrics[3] = max(metrics[3], float(np.max(np.abs(states[:, 6:8]))))
            lgr_endpoint = np.asarray(endpoints[region])
            for rollout_state, endpoint_state in zip(states, lgr_endpoint):
                metrics[4] = max(
                    metrics[4], float(np.linalg.norm(rollout_state[:3] - endpoint_state[:3]))
                )
                metrics[5] = max(
                    metrics[5], float(np.linalg.norm(rollout_state[3:6] - endpoint_state[3:6]))
                )
                attitude_error = so3_log(
                    euler_to_rot(endpoint_state[6:9]).T @ euler_to_rot(rollout_state[6:9])
                )
                metrics[6] = max(metrics[6], float(np.linalg.norm(attitude_error)))
        return (*metrics, control_violation)


class DelayPredictor:
    """P90 commit-delay estimator and SO(3)-aware sigma propagation."""

    def __init__(
        self,
        initial: float,
        window: int,
        minimum_samples: int,
        minimum: float,
        maximum: float,
        validation: float,
        margin: float,
        scheduling_margin: float = 0.02,
        latency_clip_min: float = 0.0,
        latency_clip_max: float = 2.0,
    ) -> None:
        numeric = (
            initial,
            minimum,
            maximum,
            validation,
            margin,
            scheduling_margin,
            latency_clip_min,
            latency_clip_max,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("delay parameters must be finite and nonnegative")
        if minimum > maximum:
            raise ValueError("minimum delay exceeds maximum delay")
        if window < 1 or minimum_samples < 1 or minimum_samples > window:
            raise ValueError("delay minimum samples must be in [1, window]")
        if not 0.0 <= latency_clip_min <= latency_clip_max:
            raise ValueError("invalid latency clipping range")
        self.samples = deque(maxlen=window)
        self.initial = initial
        self.window = window
        self.minimum_samples = minimum_samples
        self.minimum = minimum
        self.maximum = maximum
        self.validation = validation
        self.margin = margin
        self.scheduling_margin = scheduling_margin
        self.latency_clip_min = latency_clip_min
        self.latency_clip_max = latency_clip_max
        self.last_projection = None

    def record_latency(self, elapsed: float) -> None:
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("latency must be finite and nonnegative")
        self.samples.append(float(np.clip(elapsed, self.latency_clip_min, self.latency_clip_max)))

    def record_solve_time(self, elapsed: float) -> None:
        """Backward-compatible alias; values now mean full admission latency."""
        self.record_latency(elapsed)

    def estimate_mode(self, first: bool = False) -> str:
        if first:
            return "first_cold"
        return "steady_p90" if len(self.samples) >= self.minimum_samples else "initial_history"

    def percentile90(self) -> float:
        if len(self.samples) < self.minimum_samples:
            return self.initial
        return float(np.percentile(self.samples, 90))

    def estimate(self, first: bool = False, cold_start_delay: float = 0.0) -> float:
        base = cold_start_delay if first else self.percentile90()
        return float(
            np.clip(
                base + self.validation + self.margin + self.scheduling_margin,
                self.minimum,
                self.maximum,
            )
        )

    def propagate(
        self,
        sigma_states: np.ndarray,
        belief_stamp: float,
        commit_time: float,
        control_at: Callable[[float], np.ndarray],
        process_noise_diagonal: np.ndarray,
    ):
        if not np.isfinite(belief_stamp) or not np.isfinite(commit_time):
            raise ValueError("belief and commit clocks must be finite")
        if commit_time < belief_stamp:
            raise ValueError("commit_time precedes belief timestamp")
        duration = commit_time - belief_stamp
        steps = max(1, int(np.ceil(duration / 0.02)))
        step = duration / steps
        propagated = np.asarray(sigma_states, dtype=float).copy()
        for index in range(steps):
            absolute_time = belief_stamp + index * step
            control = np.asarray(control_at(absolute_time), dtype=float)
            propagated = np.stack(
                [rk4(propagated[:, sigma], control, step) for sigma in range(propagated.shape[1])],
                axis=1,
            )
        process = np.asarray(process_noise_diagonal, dtype=float)
        if process.shape == (9,):
            process = process[[0, 1, 2, 6, 7, 8]]
        if process.shape != (6,):
            raise ValueError("delay process noise must have 6 or 9 diagonal entries")
        pose_process = np.diag(process) * duration
        projection = joint_sigma_process_update(propagated, pose_process)
        self.last_projection = projection.diagnostics
        return (
            projection.sigma_states,
            projection.mean_state,
            projection.mean_rotation,
            projection.covariance,
        )


@dataclass(frozen=True)
class CompletionEvent:
    request: object
    result: object
    stale: bool


class LatestWinsWorker:
    """Single solver-owner thread; ROS thread drains immutable completions."""

    def __init__(self, solve: Callable, complete: Callable = None) -> None:
        self.solve = solve
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False
        self.solve_in_progress = False
        self.completions = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, request, replace_pending: bool = True) -> bool:
        with self.condition:
            if self.stopping:
                return False
            if not replace_pending and (self.solve_in_progress or self.pending is not None):
                return False
            self.pending = request
            self.condition.notify()
            return True

    def _run(self) -> None:
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                request = self.pending
                self.pending = None
                self.solve_in_progress = True
            try:
                result = self.solve(request)
            except Exception as exception:
                result = exception
            with self.condition:
                self.solve_in_progress = False
                stale = self.pending is not None
            self.completions.put(CompletionEvent(request, result, stale))

    def drain_completions(self) -> list:
        events = []
        while True:
            try:
                events.append(self.completions.get_nowait())
            except queue.Empty:
                return events

    def pending_count(self) -> int:
        with self.condition:
            return int(self.pending is not None)

    def shutdown(self, timeout: float = 5.0) -> bool:
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive()


class CandidateManager:
    def __init__(self, buffer: TrajectoryBuffer, guard: float) -> None:
        self.buffer = buffer
        self.guard = guard
        self.deadline_misses = 0
        self.stale_discards = 0

    def admit(
        self,
        request: PlanningRequest,
        result: dict,
        now: float,
        current_generation: int,
        gate: GateResult,
        frame_id: str,
    ) -> tuple:
        if request.request_generation != current_generation:
            self.stale_discards += 1
            return False, "stale"
        if not request.commit_time_finalized:
            return False, "commit time not finalized"
        if now > request.commit_time - self.guard:
            self.deadline_misses += 1
            return False, "late"
        if not gate.accepted:
            return False, ",".join(gate.reasons)
        trajectory = Trajectory(
            result["times"],
            result["states_physical"],
            result["controls_physical"],
            request.request_generation,
            int(request.commit_time * 1e9),
            request.path_generation,
            frame_id,
            result["mean_covariances"],
        )
        return self.buffer.offer(trajectory), "accepted"


def execution_decision(
    outcome: CommitOutcome,
    data_fresh: bool,
    active_remaining: float,
    goal_reached: bool,
    goal_dwell_pending: bool = False,
    terminal_goal_data_fresh: bool = False,
) -> ExecutionDecision:
    """Resolve one mutually exclusive execution action for a commit tick."""
    if outcome == CommitOutcome.COMMITTED:
        return ExecutionDecision.COMMIT
    if goal_reached:
        return ExecutionDecision.GOAL_REACHED
    if outcome == CommitOutcome.WAITING:
        return ExecutionDecision.CONTINUE if data_fresh else ExecutionDecision.SAFE_HOLD
    # An expired terminal trajectory is still a valid terminal-hold semantic
    # while an explicitly supplied mission goal is accumulating dwell time.
    if goal_dwell_pending and terminal_goal_data_fresh:
        return ExecutionDecision.CONTINUE
    if not data_fresh or active_remaining <= 0.0:
        return ExecutionDecision.SAFE_HOLD
    return ExecutionDecision.CONTINUE


def mission_goal_input_valid(
    stamp: float,
    now: float,
    frame_id: str,
    planning_frame: str,
    position,
    timeout: float,
) -> bool:
    """Validate an explicit persistent or time-limited mission goal."""
    if stamp <= 0.0 or frame_id != planning_frame or not np.all(np.isfinite(position)):
        return False
    age = now - stamp
    return age >= 0.0 and (timeout <= 0.0 or age <= timeout)


def goal_generation_changed(previous, current: str) -> bool:
    return previous is None or previous.generation != current


def mission_goal_satisfied(
    belief,
    goal,
    position_tolerance: float,
    velocity_tolerance: float,
    yaw_enabled: bool,
    yaw_tolerance: float,
) -> bool:
    """Check only an explicit mission goal, never a local path endpoint."""
    if belief is None or goal is None:
        return False
    position_ok = np.linalg.norm(belief.position - goal.position) <= position_tolerance
    velocity_ok = np.linalg.norm(belief.velocity) <= velocity_tolerance
    if not yaw_enabled:
        return bool(position_ok and velocity_ok)
    if goal.yaw is None:
        return False
    yaw_error = abs(
        np.arctan2(
            np.sin(belief.mean_state[8] - goal.yaw),
            np.cos(belief.mean_state[8] - goal.yaw),
        )
    )
    return bool(position_ok and velocity_ok and yaw_error <= yaw_tolerance)


def mission_goal_yaw_and_generation(position, quaternion_xyzw, yaw_enabled: bool):
    """Parse an optional goal yaw and create a semantic content generation."""
    position = np.asarray(position, dtype=float)
    position_key = tuple(float(value) for value in position)
    if not yaw_enabled:
        return None, ",".join(f"{value:.6f}" for value in position_key)
    from .math_utils import quat_to_rot, rot_to_euler

    yaw = float(rot_to_euler(quat_to_rot(quaternion_xyzw))[2])
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    generation = ",".join(f"{value:.6f}" for value in (*position_key, yaw))
    return yaw, generation


def planning_data_fresh(belief_age: float, belief_timeout: float, velocity_fresh: bool, path_fresh: bool):
    return bool(0.0 <= belief_age <= belief_timeout and velocity_fresh and path_fresh)


def terminal_goal_data_fresh(
    belief_age: float,
    belief_timeout: float,
    velocity_fresh: bool,
    goal_valid: bool,
):
    return bool(0.0 <= belief_age <= belief_timeout and velocity_fresh and goal_valid)


def can_restart_mission(
    state: PlannerState,
    restart_pending: bool,
    px4: dict,
    goal_valid: bool,
    belief_present: bool,
    belief_stable: bool,
    velocity_fresh: bool,
    path_fresh: bool,
) -> bool:
    return bool(
        state == PlannerState.GOAL_REACHED
        and restart_pending
        and px4.get("connected")
        and px4.get("hold_ready")
        and not px4.get("failsafe")
        and goal_valid
        and belief_present
        and belief_stable
        and velocity_fresh
        and path_fresh
    )


def commit_due_status(now: float, candidate: Trajectory, allowed_lateness: float) -> tuple:
    """Return waiting/due/late and signed commit lateness."""
    lateness = now - candidate.commit_time
    if lateness < 0.0:
        return "waiting", lateness
    if lateness > allowed_lateness:
        return "late", lateness
    return "due", lateness


def update_goal_dwell(since, now: float, conditions: bool, active_remaining: float, dwell: float):
    """Pure dwell transition helper used by the high-rate state timer."""
    if active_remaining > 0.0 or not conditions:
        return None, False
    if since is None:
        return now, False
    return since, now - since >= dwell


def commit_continuity_errors(candidate: Trajectory, belief) -> tuple:
    initial = candidate.states[0]
    position = float(np.linalg.norm(initial[:3] - belief.mean_state[:3]))
    velocity = float(np.linalg.norm(initial[3:6] - belief.mean_state[3:6]))
    attitude = float(np.linalg.norm(so3_log(belief.rotation.T @ euler_to_rot(initial[6:9]))))
    return position, velocity, attitude


def commit_continuity_accepted(candidate: Trajectory, belief, tolerances) -> tuple:
    """Return continuity admission and its position/velocity/SO(3) errors."""
    errors = commit_continuity_errors(candidate, belief)
    tolerances = tuple(float(value) for value in tolerances)
    if len(tolerances) != 3:
        raise ValueError("commit continuity requires three tolerances")
    return all(error <= tolerance for error, tolerance in zip(errors, tolerances)), errors


def can_resume(
    state: PlannerState, px4: dict, belief_stable: bool, velocity_fresh: bool, path_fresh: bool
) -> bool:
    if state == PlannerState.FAULT:
        return False
    return (
        state == PlannerState.SAFE_HOLD
        and bool(px4.get("connected"))
        and not bool(px4.get("failsafe"))
        and bool(px4.get("hold_ready"))
        and belief_stable
        and velocity_fresh
        and path_fresh
    )


def offboard_control_flags(level: str) -> dict:
    """Return mutually exclusive PX4 OffboardControlMode fields."""
    if level not in ("position", "velocity", "acceleration"):
        raise ValueError("unsupported offboard control level")
    return {
        "position": level == "position",
        "velocity": level == "velocity",
        "acceleration": level == "acceleration",
        "attitude": False,
        "body_rate": False,
    }


class CommandState(Enum):
    WAIT_CONNECTION = auto()
    PRESTREAM = auto()
    REQUEST_OFFBOARD = auto()
    REQUEST_ARM = auto()
    TAKEOFF_HOLD = auto()
    READY = auto()
    FAULT = auto()


class Px4CommandSequencer:
    """Small ACK-aware offboard/arm retry sequencer."""

    def __init__(
        self,
        auto_arm: bool,
        offboard_command: int,
        arm_command: int,
        retry_interval: float,
        ack_timeout: float,
        max_retries: int,
    ) -> None:
        self.auto_arm = auto_arm
        self.offboard_command = offboard_command
        self.arm_command = arm_command
        self.retry_interval = retry_interval
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries
        self.state = CommandState.WAIT_CONNECTION
        self.pending_command = None
        self.request_time = 0.0
        self.last_send_time = -np.inf
        self.retries = 0
        self.acknowledged = False
        self.fault_reason = ""

    def acknowledge(self, command: int, result: int) -> None:
        if command != self.pending_command:
            return
        if result == 0:
            self.acknowledged = True
        else:
            self.acknowledged = False
            self.state = CommandState.FAULT
            self.fault_reason = f"PX4 rejected command {command} with result {result}"

    def tick(
        self,
        now: float,
        connected: bool,
        prestream_done: bool,
        offboard_ready: bool,
        armed: bool,
        hold_ready: bool,
    ):
        if self.state == CommandState.FAULT:
            return None
        if not connected:
            self.state = CommandState.WAIT_CONNECTION
            return None
        if self.state == CommandState.WAIT_CONNECTION:
            self.state = CommandState.PRESTREAM
        if self.state == CommandState.PRESTREAM:
            if not prestream_done:
                return None
            if not self.auto_arm:
                if offboard_ready and armed:
                    self.state = CommandState.TAKEOFF_HOLD
                return None
            self._begin(self.offboard_command, now)
            self.state = CommandState.REQUEST_OFFBOARD
        if self.state == CommandState.REQUEST_OFFBOARD:
            if offboard_ready or self.acknowledged:
                self._begin(self.arm_command, now)
                self.state = CommandState.REQUEST_ARM
            else:
                return self._retry(now)
        if self.state == CommandState.REQUEST_ARM:
            if armed or self.acknowledged:
                self.pending_command = None
                self.state = CommandState.TAKEOFF_HOLD
            else:
                return self._retry(now)
        if self.state == CommandState.TAKEOFF_HOLD and hold_ready:
            self.state = CommandState.READY
        return None

    def _begin(self, command: int, now: float) -> None:
        self.pending_command = command
        self.request_time = now
        self.last_send_time = -np.inf
        self.retries = 0
        self.acknowledged = False

    def _retry(self, now: float):
        if now - self.request_time > self.ack_timeout and self.retries >= self.max_retries:
            self.state = CommandState.FAULT
            self.fault_reason = "PX4 command ACK retry limit exceeded"
            return None
        if now - self.last_send_time < self.retry_interval:
            return None
        if self.retries >= self.max_retries:
            return None
        self.last_send_time = now
        self.retries += 1
        return self.pending_command


PLANNER_PARAMETER_DEFAULTS = {
    "belief_topic": "/Odometry",
    "path_topic": "/ifds/local_path",
    "path_status_topic": "/ifds/path_status",
    "path_pair_timeout": 0.20,
    "path_generation_resolution": 0.05,
    "mission_goal_topic": "/ifds/mission_goal",
    "mission_goal_timeout": 0.0,
    "velocity_topic": "/uto/velocity",
    "px4_velocity_topic": "/fmu/out/vehicle_odometry",
    "px4_state_topic": "/uto/px4_status",
    "trajectory_topic": "/uto/trajectory",
    "execution_command_topic": "/uto/execution_command",
    "diagnostics_topic": "/uto/diagnostics",
    "resume_service": "/uto/resume",
    "planning_frame": "map",
    "mode": "online",
    "global_one_shot": False,
    "global_replan_enabled": True,
    "delay_compensation_enabled": True,
    "global_commit_lead_time": 0.10,
    "global_preflight_retry_enabled": True,
    "global_preflight_max_attempts": 3,
    "global_preflight_retry_period": 0.5,
    "terminal_goal_match_tolerance": 0.10,
    "velocity_source": "patched_odometry_twist",
    "velocity_frame_alignment_mode": "identity",
    "velocity_frame_yaw_offset": 0.0,
    "horizon": 3.0,
    "lookahead_count": 10,
    "lookahead_spacing": 0.4,
    "replan_rate": 1.5,
    "commit_check_rate": 50.0,
    "allowed_commit_lateness": 0.04,
    "belief_timeout": 0.3,
    "path_timeout": 0.8,
    "velocity_timeout": 0.3,
    "source_clock_tolerance": 0.5,
    "px4_velocity_time_mode": "offset",
    "stable_samples": 5,
    "covariance_eigen_floor": 1e-9,
    "covariance_inflation": 1.2,
    "position_covariance_trace_max": 0.2,
    "attitude_covariance_trace_max": 0.05,
    "mean_delta_max": 0.3,
    "covariance_delta_max": 0.1,
    "regions": 2,
    "lgr_nodes_per_region": 5,
    "sigma_count": 7,
    "control_check_points_per_region": 31,
    "gate_dense_points_per_region": 15,
    "gate_rollout_endpoint_position_tolerance": 0.3,
    "gate_rollout_endpoint_velocity_tolerance": 0.5,
    "gate_rollout_endpoint_attitude_tolerance": 0.2,
    "path_tube_radius": 0.8,
    "sigma_path_tube_radius": 1.0,
    "initial_delay": 0.5,
    "cold_start_delay": 1.2,
    "delay_p90_window": 20,
    "delay_p90_min_samples": 5,
    "minimum_delay": 0.2,
    "maximum_delay": 1.5,
    "validation_time": 0.02,
    "commit_margin": 0.08,
    "commit_scheduling_margin": 0.02,
    "latency_clip_min": 0.05,
    "latency_clip_max": 1.5,
    "commit_guard": 0.05,
    "commit_position_tolerance": 0.5,
    "commit_velocity_tolerance": 0.7,
    "commit_attitude_tolerance": 0.25,
    "goal_position_tolerance": 0.2,
    "goal_velocity_tolerance": 0.2,
    "goal_yaw_enabled": False,
    "goal_yaw_tolerance": 0.2,
    "goal_dwell_time": 1.0,
    "terminal_position_tolerance": 0.3,
    "terminal_velocity_tolerance": 0.05,
    "process_noise_diagonal": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.001, 0.001, 0.001],
    "state_scale": [3.0, 1.0, 1.2, 4.0, 4.0, 4.0, 0.6, 0.6, 0.6],
    "control_scale": [9.81, 0.48, 0.48, 1.2],
    "control_min": [0.0, -0.48, -0.48, -1.2],
    "control_max": [18.0, 0.48, 0.48, 1.2],
    "velocity_max": 4.0,
    "angle_max": 0.6,
    "solver_tolerance": 2e-5,
    "solver_max_iterations": 900,
    "weights": [1.0, 10.0, 10.0, 0.007, 1e-6, 1e-6],
}


@dataclass
class RuntimeComponents:
    nlp: object
    adapter: object
    delay: DelayPredictor
    buffer: TrajectoryBuffer
    manager: CandidateManager
    gate: FeasibilityGate


def build_runtime_components(parameter: Callable[[str], object]) -> RuntimeComponents:
    """Construct startup-only math/runtime components from ROS parameters."""
    from .belief_adapter import BeliefAdapter, StabilityConfig
    from .uto_nlp import UTOConfig, UTONLP

    nlp = UTONLP(
        UTOConfig(
            regions=parameter("regions"),
            nodes=parameter("lgr_nodes_per_region"),
            sigma=parameter("sigma_count"),
            references=parameter("lookahead_count"),
            state_scale=tuple(parameter("state_scale")),
            control_scale=tuple(parameter("control_scale")),
            control_min=tuple(parameter("control_min")),
            control_max=tuple(parameter("control_max")),
            velocity_max=parameter("velocity_max"),
            angle_max=parameter("angle_max"),
            terminal_position_tolerance=parameter("terminal_position_tolerance"),
            control_check_points=parameter("control_check_points_per_region"),
            max_iter=parameter("solver_max_iterations"),
            tolerance=parameter("solver_tolerance"),
        )
    )
    stability = StabilityConfig(
        parameter("stable_samples"),
        parameter("belief_timeout"),
        parameter("position_covariance_trace_max"),
        parameter("attitude_covariance_trace_max"),
        parameter("mean_delta_max"),
        parameter("covariance_delta_max"),
    )
    adapter = BeliefAdapter(
        parameter("covariance_eigen_floor"), parameter("covariance_inflation"), stability
    )
    delay = DelayPredictor(
        parameter("initial_delay"),
        parameter("delay_p90_window"),
        parameter("delay_p90_min_samples"),
        parameter("minimum_delay"),
        parameter("maximum_delay"),
        parameter("validation_time"),
        parameter("commit_margin"),
        parameter("commit_scheduling_margin"),
        parameter("latency_clip_min"),
        parameter("latency_clip_max"),
    )
    buffer = TrajectoryBuffer()
    manager = CandidateManager(buffer, parameter("commit_guard"))
    gate = FeasibilityGate(
        GateConfig(
            velocity_max=parameter("velocity_max"),
            angle_max=parameter("angle_max"),
            control_min=tuple(parameter("control_min")),
            control_max=tuple(parameter("control_max")),
            terminal_position_tolerance=parameter("terminal_position_tolerance"),
            path_tube=parameter("path_tube_radius"),
            sigma_path_tube=parameter("sigma_path_tube_radius"),
            dense_points_per_region=parameter("gate_dense_points_per_region"),
            rollout_endpoint_position_tolerance=parameter(
                "gate_rollout_endpoint_position_tolerance"
            ),
            rollout_endpoint_velocity_tolerance=parameter(
                "gate_rollout_endpoint_velocity_tolerance"
            ),
            rollout_endpoint_attitude_tolerance=parameter(
                "gate_rollout_endpoint_attitude_tolerance"
            ),
        )
    )
    return RuntimeComponents(nlp, adapter, delay, buffer, manager, gate)

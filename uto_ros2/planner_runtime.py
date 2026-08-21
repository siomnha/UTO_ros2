"""Planner state, delay prediction, feasibility admission, and worker utilities."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import threading
import time
from typing import Callable
import numpy as np
from .belief_adapter import reconstruct_belief_from_sigma, resample_sigma_states
from .dynamics import rk4
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


@dataclass
class GateResult:
    accepted: bool
    reasons: list
    elapsed: float
    max_path_error: float
    max_sigma_path_error: float
    max_lgr_dynamics_residual: float


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
        return GateResult(
            not reasons,
            reasons,
            time.perf_counter() - started,
            mean_error,
            sigma_error,
            residual,
        )


class DelayPredictor:
    """P90 commit-delay estimator and SO(3)-aware sigma propagation."""

    def __init__(
        self,
        default: float,
        window: int,
        minimum: float,
        maximum: float,
        validation: float,
        margin: float,
    ) -> None:
        self.samples = deque(maxlen=window)
        self.default = default
        self.minimum = minimum
        self.maximum = maximum
        self.validation = validation
        self.margin = margin

    def record_solve_time(self, elapsed: float) -> None:
        self.samples.append(float(elapsed))

    def percentile90(self) -> float:
        if len(self.samples) < 3:
            return self.default
        return float(np.percentile(self.samples, 90))

    def estimate(self) -> float:
        return float(
            np.clip(
                self.percentile90() + self.validation + self.margin,
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
        mean, rotation, covariance = reconstruct_belief_from_sigma(propagated)
        process = np.asarray(process_noise_diagonal, dtype=float)
        if process.shape == (9,):
            process = process[[0, 1, 2, 6, 7, 8]]
        if process.shape != (6,):
            raise ValueError("delay process noise must have 6 or 9 diagonal entries")
        covariance = covariance + np.diag(process) * duration
        resampled = resample_sigma_states(mean, rotation, covariance)
        return resampled, mean, rotation, covariance


class LatestWinsWorker:
    """One solver-owner thread with at most one pending latest request."""

    def __init__(self, solve: Callable, complete: Callable) -> None:
        self.solve = solve
        self.complete = complete
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False
        self.solve_in_progress = False
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
            self.complete(request, result, stale)

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


def commit_continuity_errors(candidate: Trajectory, belief) -> tuple:
    initial = candidate.states[0]
    position = float(np.linalg.norm(initial[:3] - belief.mean_state[:3]))
    velocity = float(np.linalg.norm(initial[3:6] - belief.mean_state[3:6]))
    attitude = float(np.linalg.norm(so3_log(belief.rotation.T @ euler_to_rot(initial[6:9]))))
    return position, velocity, attitude


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
    "velocity_topic": "/uto/velocity",
    "px4_velocity_topic": "/fmu/out/vehicle_odometry",
    "px4_state_topic": "/uto/px4_status",
    "trajectory_topic": "/uto/trajectory",
    "execution_command_topic": "/uto/execution_command",
    "diagnostics_topic": "/uto/diagnostics",
    "resume_service": "/uto/resume",
    "planning_frame": "map",
    "mode": "online",
    "velocity_source": "patched_odometry_twist",
    "velocity_frame_alignment_mode": "identity",
    "velocity_frame_yaw_offset": 0.0,
    "horizon": 3.0,
    "lookahead_count": 10,
    "lookahead_spacing": 0.4,
    "replan_rate": 1.5,
    "belief_timeout": 0.3,
    "path_timeout": 0.8,
    "velocity_timeout": 0.3,
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
    "path_tube_radius": 0.8,
    "sigma_path_tube_radius": 1.0,
    "initial_delay": 0.5,
    "cold_start_delay": 1.2,
    "delay_p90_window": 20,
    "minimum_delay": 0.2,
    "maximum_delay": 1.5,
    "validation_time": 0.02,
    "commit_margin": 0.08,
    "commit_guard": 0.05,
    "commit_position_tolerance": 0.5,
    "commit_velocity_tolerance": 0.7,
    "commit_attitude_tolerance": 0.25,
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
        parameter("minimum_delay"),
        parameter("maximum_delay"),
        parameter("validation_time"),
        parameter("commit_margin"),
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
        )
    )
    return RuntimeComponents(nlp, adapter, delay, buffer, manager, gate)

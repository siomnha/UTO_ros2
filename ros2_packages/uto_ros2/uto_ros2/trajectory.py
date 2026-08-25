"""Physical trajectory schema, interpolation, and atomic active/candidate buffer."""

from dataclasses import dataclass
import json
import threading
from typing import Optional
import numpy as np


@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    generation: int
    commit_time_ns: int
    path_generation: str = ""
    frame_id: str = "map"
    mean_covariances: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.states = np.asarray(self.states, dtype=float)
        self.controls = np.asarray(self.controls, dtype=float)
        if self.times.ndim != 1 or len(self.times) < 2:
            raise ValueError("times must have at least two samples")
        if self.times[0] < 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")
        if self.states.shape != (len(self.times), 9):
            raise ValueError("states must have shape [samples,9]")
        if self.controls.ndim != 2 or self.controls.shape[1] != 4:
            raise ValueError("controls must have shape [samples,4]")
        if len(self.controls) not in (len(self.times), len(self.times) - 1):
            raise ValueError("control sample count is inconsistent")
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.controls)):
            raise ValueError("trajectory contains non-finite values")
        if self.mean_covariances is None:
            return
        self.mean_covariances = np.asarray(self.mean_covariances, dtype=float)
        if self.mean_covariances.shape != (len(self.times), 9, 9):
            raise ValueError("covariances must have shape [samples,9,9]")
        if not np.all(np.isfinite(self.mean_covariances)):
            raise ValueError("trajectory covariance contains non-finite values")

    @property
    def commit_time(self) -> float:
        return self.commit_time_ns * 1e-9

    @property
    def end_time(self) -> float:
        return self.commit_time + self.times[-1]

    def remaining(self, now: float) -> float:
        return max(0.0, self.end_time - now)

    def sample(self, now: float):
        relative = float(np.clip(now - self.commit_time, self.times[0], self.times[-1]))
        state = np.array(
            [np.interp(relative, self.times, self.states[:, column]) for column in range(9)]
        )
        unwrapped_yaw = np.unwrap(self.states[:, 8])
        state[8] = np.arctan2(
            np.sin(np.interp(relative, self.times, unwrapped_yaw)),
            np.cos(np.interp(relative, self.times, unwrapped_yaw)),
        )
        control_times = self.times[: len(self.controls)]
        control = np.array(
            [np.interp(relative, control_times, self.controls[:, column]) for column in range(4)]
        )
        return state, control

    def to_dict(self) -> dict:
        return {
            "schema": "uto_trajectory/v1",
            "generation": self.generation,
            "path_generation": self.path_generation,
            "frame_id": self.frame_id,
            "commit_time_ns": self.commit_time_ns,
            "times": self.times.tolist(),
            "states_physical": self.states.tolist(),
            "controls_physical": self.controls.tolist(),
            "mean_covariances": (
                self.mean_covariances.tolist() if self.mean_covariances is not None else None
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict):
        required = {
            "generation",
            "path_generation",
            "frame_id",
            "commit_time_ns",
            "times",
            "states_physical",
            "controls_physical",
            "mean_covariances",
        }
        if data.get("schema") != "uto_trajectory/v1" or not required.issubset(data):
            raise ValueError("invalid UTO trajectory schema")
        return cls(
            data["times"],
            data["states_physical"],
            data["controls_physical"],
            int(data["generation"]),
            int(data["commit_time_ns"]),
            data["path_generation"],
            data["frame_id"],
            data["mean_covariances"],
        )

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))


class TrajectoryBuffer:
    """Thread-safe pending/active storage; continuity is decided by the planner."""

    def __init__(self) -> None:
        self.active: Optional[Trajectory] = None
        self.candidate: Optional[Trajectory] = None
        self.latest_generation = -1
        self.stale_discards = 0
        self.lock = threading.RLock()

    def offer(self, trajectory: Trajectory) -> bool:
        with self.lock:
            if trajectory.generation < self.latest_generation:
                self.stale_discards += 1
                return False
            self.candidate = trajectory
            self.latest_generation = trajectory.generation
            return True

    def candidate_due(self, now: float) -> bool:
        with self.lock:
            return self.candidate is not None and now >= self.candidate.commit_time

    def commit_candidate(self) -> Optional[Trajectory]:
        with self.lock:
            if self.candidate is None:
                return None
            self.active = self.candidate
            self.candidate = None
            return self.active

    def discard_candidate(self) -> None:
        with self.lock:
            if self.candidate is not None:
                self.stale_discards += 1
            self.candidate = None

    def sample(self, now: float):
        with self.lock:
            if self.active is None or now > self.active.end_time:
                return None
            return self.active.sample(now)

    def remaining(self, now: float) -> float:
        with self.lock:
            return self.active.remaining(now) if self.active else 0.0


@dataclass
class ExecutionSetpoint:
    state: np.ndarray
    control: np.ndarray
    mode: str


class TrajectoryExecution:
    """Bridge-side execution policy with takeoff, terminal, and emergency holds."""

    def __init__(self, takeoff_position) -> None:
        takeoff_state = np.zeros(9)
        takeoff_state[:3] = np.asarray(takeoff_position, dtype=float)
        self.takeoff_hold = ExecutionSetpoint(
            takeoff_state, np.array([9.81, 0, 0, 0]), "TAKEOFF_HOLD"
        )
        self.trajectory: Optional[Trajectory] = None
        self.last_valid = self.takeoff_hold
        self.terminal_hold = self.takeoff_hold

    def accept_executable(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        terminal = trajectory.states[-1].copy()
        terminal[3:6] = 0.0
        terminal[6:8] = 0.0
        self.terminal_hold = ExecutionSetpoint(terminal, np.array([9.81, 0, 0, 0]), "TERMINAL_HOLD")

    def request_emergency_hold(self) -> None:
        """Stop execution and hold the last actually issued setpoint."""
        self.trajectory = None
        state = self.last_valid.state.copy()
        state[3:6] = 0.0
        state[6:8] = 0.0
        self.terminal_hold = ExecutionSetpoint(state, np.array([9.81, 0, 0, 0]), "EMERGENCY_HOLD")

    def request_hold_current(self) -> None:
        """Stop trajectory execution at the last issued position without declaring a fault."""
        self.trajectory = None
        state = self.last_valid.state.copy()
        state[3:6] = 0.0
        state[6:8] = 0.0
        self.last_valid = ExecutionSetpoint(
            state, np.array([9.81, 0, 0, 0]), "HOLD_CURRENT"
        )
        self.terminal_hold = self.last_valid

    def select(
        self, now: float, trajectory_allowed: bool, takeoff_complete: bool
    ) -> ExecutionSetpoint:
        if not takeoff_complete:
            return self.takeoff_hold
        if self.trajectory is not None and trajectory_allowed:
            if self.trajectory.commit_time <= now <= self.trajectory.end_time:
                state, control = self.trajectory.sample(now)
                self.last_valid = ExecutionSetpoint(state, control, "TRAJECTORY")
                return self.last_valid
            if now > self.trajectory.end_time:
                self.trajectory = None
                return self.terminal_hold
        if self.last_valid.mode == "TRAJECTORY":
            emergency = self.last_valid.state.copy()
            emergency[3:6] = 0.0
            return ExecutionSetpoint(emergency, np.array([9.81, 0, 0, 0]), "EMERGENCY_HOLD")
        return self.terminal_hold if takeoff_complete else self.takeoff_hold

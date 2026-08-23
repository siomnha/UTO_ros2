"""ROS-independent IFDS path-only geometry and status primitives."""

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SphereObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("obstacle center must be a finite 3-vector")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive")
        object.__setattr__(self, "center", center)


@dataclass(frozen=True)
class PathStatus:
    valid: bool
    path_stamp_ns: int
    path_generation: int
    goal_generation: int
    planned_at: float
    valid_until: float
    reason: str

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("status valid must be boolean")
        if any(
            type(value) is not int
            for value in (self.path_stamp_ns, self.path_generation, self.goal_generation)
        ):
            raise ValueError("status stamp and generations must be integers")
        numeric = (self.planned_at, self.valid_until)
        if not all(np.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("status times must be finite and nonnegative")
        if self.path_stamp_ns < 0 or self.path_generation < 0 or self.goal_generation < 0:
            raise ValueError("status generations and stamp must be nonnegative")
        if self.valid and (self.path_stamp_ns <= 0 or self.valid_until < self.planned_at):
            raise ValueError("valid status requires a live nonzero path stamp")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("status reason is required")

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)
        required = set(cls.__dataclass_fields__)
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("invalid IFDS path status schema")
        return cls(**data)


def status_matches_path(status: PathStatus, path_stamp_ns: int, now: float) -> bool:
    return bool(
        status.valid
        and status.path_stamp_ns == path_stamp_ns
        and status.planned_at <= now <= status.valid_until
    )


def segment_clearance(start, end, obstacle: SphereObstacle) -> float:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    alpha = np.clip(
        np.dot(obstacle.center - start, direction) / max(np.dot(direction, direction), 1e-12),
        0.0,
        1.0,
    )
    closest = start + alpha * direction
    return float(np.linalg.norm(closest - obstacle.center) - obstacle.radius)


def segment_is_safe(start, end, obstacles: Iterable[SphereObstacle], clearance: float) -> bool:
    return all(segment_clearance(start, end, obstacle) >= clearance for obstacle in obstacles)


def _detour(start, goal, obstacle: SphereObstacle, clearance: float) -> np.ndarray:
    direction = np.asarray(goal, float) - np.asarray(start, float)
    direction /= max(np.linalg.norm(direction), 1e-12)
    reference = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(direction, reference)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    lateral /= max(np.linalg.norm(lateral), 1e-12)
    side_a = obstacle.center + lateral * (obstacle.radius + clearance)
    side_b = obstacle.center - lateral * (obstacle.radius + clearance)
    return min((side_a, side_b), key=lambda point: np.linalg.norm(point - start) + np.linalg.norm(goal - point))


def plan_ifds_path(start, goal, obstacles=(), clearance=0.25, target_threshold=0.05):
    """Build a conservative path-only polyline and append the exact mission goal."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (3,) or goal.shape != (3,) or not np.all(np.isfinite([start, goal])):
        raise ValueError("IFDS start/goal must be finite 3-vectors")
    if target_threshold <= 0.0 or clearance < 0.0:
        raise ValueError("invalid IFDS thresholds")
    obstacles = tuple(obstacles)
    points = [start]
    current = start
    for _ in range(max(1, len(obstacles) * 2 + 1)):
        blockers = [o for o in obstacles if not segment_is_safe(current, goal, [o], clearance)]
        if not blockers:
            break
        blocker = min(blockers, key=lambda o: segment_clearance(current, goal, o))
        waypoint = _detour(current, goal, blocker, clearance)
        if not segment_is_safe(current, waypoint, obstacles, clearance):
            raise ValueError("NO_VALID_IFDS_PATH")
        points.append(waypoint)
        current = waypoint
    if not segment_is_safe(current, goal, obstacles, clearance):
        raise ValueError("NO_VALID_IFDS_PATH")
    # target_threshold is only the integration termination criterion; the
    # published path always contains the exact checked mission goal.
    if np.linalg.norm(points[-1] - goal) > 0.0:
        points.append(goal)
    if len(points) < 2:
        points.append(goal.copy())
    return np.asarray(points)


def rotate_vector_by_quaternion(vector, quaternion_xyzw):
    vector = np.asarray(vector, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if vector.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("invalid transform")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("invalid transform quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation @ vector


def transform_position(position, translation, quaternion_xyzw):
    return rotate_vector_by_quaternion(position, quaternion_xyzw) + np.asarray(translation, float)


def multiply_quaternions(left_xyzw, right_xyzw):
    x1, y1, z1, w1 = np.asarray(left_xyzw, dtype=float)
    x2, y2, z2, w2 = np.asarray(right_xyzw, dtype=float)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    norm = np.linalg.norm(result)
    if not np.all(np.isfinite(result)) or norm < 1e-12:
        raise ValueError("invalid composed quaternion")
    return result / norm

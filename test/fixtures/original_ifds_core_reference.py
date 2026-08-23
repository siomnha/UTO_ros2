"""Core Interfered Fluid Dynamical System (IFDS) planner.

This module ports the MATLAB velocity modulation in ``new_dynamic/src`` to a
ROS-friendly Python implementation for known static or dynamic super-ellipsoid
obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ObstacleMotion:
    """Optional deterministic obstacle motion model for simulation."""

    enabled: bool = False
    radius_y: float = 0.0
    radius_z: float = 0.0
    angular_speed: float = 0.0
    phase: float = 0.0
    motion_type: str = "static"
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict) -> "ObstacleMotion":
        motion = data.get("motion", {}) or {}
        if isinstance(motion, str):
            motion = {"type": motion}
        motion_type = str(motion.get("type", "static")).lower()
        enabled = bool(data.get("dynamic", False)) or motion_type in (
            "circular_y", "circle_y", "sin_y", "ping_pong"
        )
        radius = float(motion.get("radius", motion.get("radius_y", 0.0)))
        return cls(
            enabled=enabled,
            radius_y=radius,
            radius_z=float(motion.get("radius_z", 0.0)),
            angular_speed=float(motion.get("angular_speed", motion.get("omega", 0.0))),
            phase=float(motion.get("phase", 0.0)),
            motion_type=motion_type,
            start=np.asarray(
                motion.get("start", data.get("center", [0.0, 0.0, 0.0])), dtype=float
            ),
            end=np.asarray(
                motion.get("end", data.get("center", [0.0, 0.0, 0.0])), dtype=float
            ),
            speed=abs(float(motion.get("velocity", 0.0))),
        )

    def offset(self, time_s: float) -> np.ndarray:
        if self.motion_type == "ping_pong":
            return self._ping_pong(time_s)[0] - self.start
        if not self.enabled or abs(self.angular_speed) < 1e-12:
            return np.zeros(3)
        theta = self.angular_speed * time_s + self.phase
        return np.array([0.0, self.radius_y * np.sin(theta), self.radius_z * np.cos(theta)], dtype=float)

    def velocity(self, time_s: float) -> np.ndarray:
        """Instantaneous obstacle velocity at the current planning time."""

        if self.motion_type == "ping_pong":
            return self._ping_pong(time_s)[1]
        if not self.enabled or abs(self.angular_speed) < 1e-12:
            return np.zeros(3)
        theta = self.angular_speed * time_s + self.phase
        return np.array(
            [
                0.0,
                self.radius_y * self.angular_speed * np.cos(theta),
                -self.radius_z * self.angular_speed * np.sin(theta),
            ],
            dtype=float,
        )

    def _ping_pong(self, time_s: float) -> Tuple[np.ndarray, np.ndarray]:
        delta = self.end - self.start
        length = float(np.linalg.norm(delta))
        if not self.enabled or length < 1e-12 or self.speed < 1e-12:
            return self.start, np.zeros(3)
        direction = delta / length
        wrapped = float(np.fmod(self.speed * time_s, 2.0 * length))
        forward = wrapped < length
        distance = wrapped if forward else 2.0 * length - wrapped
        velocity = self.speed * direction if forward else -self.speed * direction
        return self.start + distance * direction, velocity


@dataclass(frozen=True)
class Obstacle:
    """Known super-ellipsoid obstacle used by IFDS."""

    name: str
    center: np.ndarray
    axes: np.ndarray
    exponents: np.ndarray
    safety_margin: float = 0.0
    motion: ObstacleMotion = field(default_factory=ObstacleMotion)

    @classmethod
    def from_mapping(cls, data: dict) -> "Obstacle":
        return cls(
            name=str(data.get("name", "obstacle")),
            center=np.asarray(data["center"], dtype=float),
            # Keep axes equal to physical geometry.  The AIAA IFDS safeguarding
            # distance is applied only through rho0* via IFDSConfig.delta_g.
            axes=np.asarray(data["axes"], dtype=float),
            exponents=np.asarray(data.get("exponents", [1.0, 1.0, 1.0]), dtype=float),
            safety_margin=float(data.get("safety_margin", 0.0)),
            motion=ObstacleMotion.from_mapping(data),
        )

    @property
    def rstar(self) -> float:
        return float(np.min(self.axes))

    def center_at(self, time_s: float, dynamic_obstacles: bool) -> np.ndarray:
        if not dynamic_obstacles:
            return self.center
        if self.motion.motion_type == "ping_pong":
            return self.motion._ping_pong(time_s)[0]
        return self.center + self.motion.offset(time_s)

    def velocity_at(self, time_s: float, dynamic_obstacles: bool) -> np.ndarray:
        if not dynamic_obstacles:
            return np.zeros(3)
        return self.motion.velocity(time_s)

    def gamma_normal_tangent(
        self, position: np.ndarray, alpha_deg: float, time_s: float, dynamic_obstacles: bool
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        center = self.center_at(time_s, dynamic_obstacles)
        rel = (position - center) / self.axes
        p, q, r = self.exponents
        powers = np.array([2 * p, 2 * q, 2 * r], dtype=float)
        gamma_terms = np.array([_stable_abs_power(value, power) for value, power in zip(rel, powers)], dtype=float)
        gamma = float(np.sum(gamma_terms))

        # Same super-ellipsoid gradient as create_shape.m, evaluated in log-space
        # so high-order shapes remain numerically stable instead of raising
        # OverflowError on large coordinates.
        d_g = np.array(
            [
                (2 * p * _stable_signed_power(rel[0], 2 * p - 1)) / self.axes[0],
                (2 * q * _stable_signed_power(rel[1], 2 * q - 1)) / self.axes[1],
                (2 * r * _stable_signed_power(rel[2], 2 * r - 1)) / self.axes[2],
            ],
            dtype=float,
        )
        alpha = np.deg2rad(alpha_deg)
        rot = np.array(
            [
                [d_g[1], d_g[0] * d_g[2], d_g[0]],
                [-d_g[0], d_g[1] * d_g[2], d_g[1]],
                [0.0, -(d_g[0] ** 2) - (d_g[1] ** 2), d_g[2]],
            ],
            dtype=float,
        )
        tangent = rot @ np.array([np.cos(alpha), np.sin(alpha), 0.0])
        return gamma, d_g, tangent, center


@dataclass(frozen=True)
class PlanarWall:
    """Planar corridor boundary whose normal points into valid free space."""

    name: str
    axis: int
    boundary: float
    inside_sign: float

    @classmethod
    def from_mapping(cls, data: dict) -> "PlanarWall":
        axis_value = data["axis"]
        axis = {"x": 0, "y": 1, "z": 2}.get(str(axis_value).lower(), axis_value)
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError(f"invalid wall axis: {axis_value}")
        inside_sign = float(data["inside_sign"])
        if inside_sign not in (-1.0, 1.0):
            raise ValueError("wall inside_sign must be -1 or 1")
        return cls(str(data.get("name", "wall")), axis, float(data["boundary"]), inside_sign)


ObstacleGeometry = Union[Obstacle, PlanarWall]


def obstacle_from_mapping(data: dict) -> ObstacleGeometry:
    """Parse wall entries while retaining the legacy superellipsoid default."""

    geometry_type = str(data.get("type", "superellipsoid")).lower()
    if geometry_type == "wall":
        return PlanarWall.from_mapping(data)
    if geometry_type == "superellipsoid":
        return Obstacle.from_mapping(data)
    raise ValueError(f"unsupported obstacle type: {geometry_type}")


@dataclass
class IFDSConfig:
    rho0: float = 2.5
    sigma0: float = 0.01
    cruise_speed: float = 2.0
    dt: float = 0.1
    max_iterations: int = 1000
    target_threshold: float = 1.0
    delta_g: float = 2.0
    alpha_deg: float = 0.0
    shape_following: bool = False
    min_gamma: float = 1.02
    dynamic_obstacles: bool = False
    velocity_mode: str = "normal"
    optimizer_mode: int = 0
    local_optimizer_period_steps: int = 5
    wall_modulation_gain: float = 1.5
    wall_influence_distance: float = 1.0


class IFDSPlanner:
    """IFDS waypoint integrator for known static or dynamic obstacles."""

    def __init__(
        self,
        config: IFDSConfig,
        obstacles: Iterable[ObstacleGeometry],
        plan_time_s: float = 0.0,
    ):
        if config.velocity_mode not in ("normal", "relative"):
            raise ValueError(
                f"unsupported velocity_mode {config.velocity_mode!r}; expected 'normal' or 'relative'"
            )
        self.config = config
        self.obstacles: List[ObstacleGeometry] = list(obstacles)
        # Match the MATLAB planner: each IFDS() call freezes obstacle geometry
        # for the whole candidate path.  Dynamic obstacle centers are refreshed
        # only when the outer ROS replanning loop invokes a new plan.
        self.plan_time_s = float(plan_time_s)
        self.rho = float(config.rho0)
        self.sigma = float(config.sigma0)

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Tuple[bool, np.ndarray, str]:
        waypoints = [np.asarray(start, dtype=float)]
        goal = np.asarray(goal, dtype=float)
        for step in range(self.config.max_iterations):
            current = waypoints[-1]
            outside_wall = next(
                (
                    wall.name
                    for wall in self.obstacles
                    if isinstance(wall, PlanarWall)
                    and wall.inside_sign * (current[wall.axis] - wall.boundary) < 0.0
                ),
                None,
            )
            if outside_wall is not None:
                return False, np.vstack(waypoints), f"outside corridor boundary: {outside_wall}"
            if float(np.linalg.norm(current - goal)) <= self.config.target_threshold:
                return True, np.vstack(waypoints), "target reached"
            try:
                ubar = self._modulated_velocity(current, goal, step)
            except FloatingPointError as exc:
                return False, np.vstack(waypoints), str(exc)
            if not np.all(np.isfinite(ubar)) or float(np.linalg.norm(ubar)) < 1e-6:
                return False, np.vstack(waypoints), "invalid IFDS velocity"
            waypoints.append(current + ubar * self.config.dt)
        reached = float(np.linalg.norm(waypoints[-1] - goal)) <= self.config.target_threshold
        return reached, np.vstack(waypoints), "target reached" if reached else "iteration limit exceeded"

    def _modulated_velocity(self, position: np.ndarray, goal: np.ndarray, step: int) -> np.ndarray:
        dist = float(np.linalg.norm(position - goal))
        if dist < 1e-9:
            return np.zeros(3)
        u = -self.config.cruise_speed * (position - goal) / dist
        if not self.obstacles:
            return u

        matrices = []
        obstacle_velocities = []
        gammas = []
        time_s = self.plan_time_s
        for obstacle in self.obstacles:
            if isinstance(obstacle, PlanarWall):
                clearance = obstacle.inside_sign * (position[obstacle.axis] - obstacle.boundary)
                if clearance < 0.0:
                    raise FloatingPointError(f"outside corridor boundary: {obstacle.name}")
                influence = self.config.wall_influence_distance
                if influence <= 0.0 or clearance >= influence:
                    continue
                normal = np.zeros(3)
                normal[obstacle.axis] = obstacle.inside_sign
                activation = (1.0 - clearance / influence) ** 2
                matrix = np.eye(3)
                if float(normal @ u) < 0.0:
                    projector = np.outer(normal, normal) / float(normal @ normal)
                    matrix -= self.config.wall_modulation_gain * activation * projector
                matrices.append(matrix)
                obstacle_velocities.append(np.zeros(3))
                gammas.append(1.0 + clearance / influence)
                continue
            gamma, normal, tangent, center = obstacle.gamma_normal_tangent(
                position, self.config.alpha_deg, time_s, self.config.dynamic_obstacles
            )
            if gamma <= self.config.min_gamma:
                raise FloatingPointError(f"inside safety boundary of {obstacle.name}: gamma={gamma:.3f}")
            obstacle_velocity = obstacle.velocity_at(time_s, self.config.dynamic_obstacles)
            input_velocity = u if self.config.velocity_mode == "normal" else u - obstacle_velocity
            n_norm2 = float(normal @ normal)
            t_norm = float(np.linalg.norm(tangent))
            if n_norm2 < 1e-12 or t_norm < 1e-12:
                matrices.append(np.eye(3))
                obstacle_velocities.append(obstacle_velocity)
                gammas.append(gamma)
                continue
            incoming = float(normal @ input_velocity) < 0.0 or self.config.shape_following
            if not incoming:
                matrices.append(np.eye(3))
                obstacle_velocities.append(obstacle_velocity)
                gammas.append(gamma)
                continue
            dist_obj = float(np.linalg.norm(position - center))
            if self._should_run_local_optimizer(step):
                self.rho, self.sigma = self._local_optimize(
                    gamma, normal, tangent, input_velocity, dist, dist_obj, obstacle.rstar
                )
            matrices.append(self._modulation_matrix(gamma, normal, tangent, dist, dist_obj, obstacle.rstar, self.rho, self.sigma))
            obstacle_velocities.append(obstacle_velocity)
            gammas.append(gamma)

        if not matrices:
            return u
        weights = self._weights(gammas)
        if self.config.velocity_mode == "relative":
            return sum(
                weight * (matrix @ (u - obstacle_velocity) + obstacle_velocity)
                for weight, matrix, obstacle_velocity in zip(weights, matrices, obstacle_velocities)
            )
        modulation = sum(weight * matrix for weight, matrix in zip(weights, matrices))
        return modulation @ u

    def _should_run_local_optimizer(self, step: int) -> bool:
        period = max(1, int(self.config.local_optimizer_period_steps))
        return int(self.config.optimizer_mode) == 2 and step % period == 0

    def _modulation_matrix(
        self,
        gamma: float,
        normal: np.ndarray,
        tangent: np.ndarray,
        dist: float,
        dist_obj: float,
        rstar: float,
        rho0: float,
        sigma0: float,
    ) -> np.ndarray:
        n_norm2 = float(normal @ normal)
        t_norm = float(np.linalg.norm(tangent))
        rho_star = self._rho_star(gamma, rstar, rho0)
        common_exp = np.exp(1.0 - 1.0 / max(dist_obj * dist, 1e-9))
        rho = max(rho_star * common_exp, 1e-6)
        sigma = max(sigma0 * common_exp, 1e-6)
        gamma_rho = _stable_abs_power(gamma, 1.0 / rho)
        gamma_sigma = _stable_abs_power(gamma, 1.0 / sigma)
        repulsive = np.outer(normal, normal) / (gamma_rho * n_norm2)
        tangential = np.outer(tangent, normal) / (gamma_sigma * t_norm * np.sqrt(n_norm2))
        return np.eye(3) - repulsive + tangential

    def _local_optimize(
        self, gamma: float, normal: np.ndarray, tangent: np.ndarray, u: np.ndarray, dist: float, dist_obj: float, rstar: float
    ) -> Tuple[float, float]:
        """Lightweight local optimizer equivalent to MATLAB path_opt2/norm_ubar.

        MATLAB uses fmincon over rho0/sigma0.  For deterministic ROS runtime we
        use a bounded local grid around the current parameters and minimize
        ``||M(rho0, sigma0) u||^2`` without adding SciPy as a dependency.
        """

        rho_candidates = np.clip(
            np.array([self.rho - 0.2, self.rho - 0.1, self.rho, self.rho + 0.1, self.rho + 0.2]), 0.05, 2.0
        )
        sigma_candidates = np.clip(
            np.array([self.sigma - 0.1, self.sigma - 0.05, self.sigma, self.sigma + 0.05, self.sigma + 0.1]), 0.0, 1.0
        )
        best = (float("inf"), self.rho, self.sigma)
        for rho0 in rho_candidates:
            for sigma0 in sigma_candidates:
                matrix = self._modulation_matrix(gamma, normal, tangent, dist, dist_obj, rstar, float(rho0), float(sigma0))
                score = float((matrix @ u) @ (matrix @ u))
                if score < best[0]:
                    best = (score, float(rho0), float(sigma0))
        return best[1], best[2]

    def _rho_star(self, gamma: float, rstar: float, rho0: float) -> float:
        gap_scale = ((rstar + self.config.delta_g) / rstar) ** 2
        denom = np.log(max(abs(gamma - gap_scale + 1.0), 1e-300))
        if abs(denom) < 1e-9:
            return rho0
        return float(np.log(max(abs(gamma), 1e-300)) / denom * rho0)

    @staticmethod
    def _weights(gammas: List[float]) -> List[float]:
        if len(gammas) == 1:
            return [1.0]
        raw = []
        for j, gamma_j in enumerate(gammas):
            w = 1.0
            for i, gamma_i in enumerate(gammas):
                if i == j:
                    continue
                denom = (gamma_j - 1.0) + (gamma_i - 1.0)
                w *= (gamma_i - 1.0) / denom if abs(denom) > 1e-9 else 0.0
            raw.append(w)
        total = float(sum(raw))
        if abs(total) < 1e-12:
            return [1.0 / len(gammas)] * len(gammas)
        return [w / total for w in raw]


def _stable_abs_power(value: float, power: float) -> float:
    magnitude = abs(float(value))
    if magnitude < 1e-300:
        return 0.0
    return float(np.exp(np.clip(float(power) * np.log(magnitude), -745.0, 700.0)))


def _stable_signed_power(value: float, power: float) -> float:
    magnitude_power = _stable_abs_power(value, power)
    rounded = round(float(power))
    if float(value) < 0.0 and abs(float(power) - rounded) < 1e-9 and int(rounded) % 2 == 1:
        return -magnitude_power
    return magnitude_power

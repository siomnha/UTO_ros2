import hashlib
import numpy as np

from .ifds_core import Obstacle, PlanarWall


class Polyline:
    def __init__(self, points):
        self.p = np.asarray(points, float).reshape(-1, 3)
        if len(self.p) < 2 or not np.all(np.isfinite(self.p)):
            raise ValueError("invalid path")
        self.ds = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.r_[0, np.cumsum(self.ds)]
        if self.s[-1] <= 0:
            raise ValueError("zero length path")

    def project(self, q):
        q = np.asarray(q)
        best = (np.inf, 0.0, self.p[0])
        for i, d in enumerate(np.diff(self.p, axis=0)):
            a = np.clip(np.dot(q - self.p[i], d) / max(np.dot(d, d), 1e-12), 0, 1)
            x = self.p[i] + a * d
            dist = np.linalg.norm(q - x)
            if dist < best[0]:
                best = (dist, self.s[i] + a * self.ds[i], x)
        return best[1], best[2], best[0]

    def at(self, s):
        s = float(np.clip(s, 0, self.s[-1]))
        i = min(np.searchsorted(self.s, s, side="right") - 1, len(self.ds) - 1)
        a = (s - self.s[i]) / max(self.ds[i], 1e-12)
        return self.p[i] + a * (self.p[i + 1] - self.p[i])

    def lookahead(self, q, count, spacing):
        s, _, _ = self.project(q)
        return np.array([self.at(s + i * spacing) for i in range(count)])


def path_generation(stamp_or_points, points=None, resolution=0.05):
    """Hash quantized geometry only; timestamps remain freshness metadata."""
    geometry = stamp_or_points if points is None else points
    geometry = np.asarray(geometry, dtype=float).reshape(-1, 3)
    if resolution <= 0.0 or not np.isfinite(resolution):
        raise ValueError("path generation resolution must be positive")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("path generation geometry must be finite")
    quantized = np.rint(geometry / resolution).astype(np.int64)
    return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]


def append_exact_ifds_goal(planner, waypoints, goal):
    """Append the exact goal only when original IFDS wall/gamma tests permit it."""
    points = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    if np.linalg.norm(points[-1] - goal) <= 1e-10:
        return True, points, "PATH_REACHES_MISSION_GOAL"
    distance = float(np.linalg.norm(goal - points[-1]))
    step = max(planner.config.cruise_speed * planner.config.dt, 0.02)
    count = max(2, int(np.ceil(distance / step)) + 1)
    for point in np.linspace(points[-1], goal, count)[1:]:
        for obstacle in planner.obstacles:
            if isinstance(obstacle, PlanarWall):
                if obstacle.inside_sign * (point[obstacle.axis] - obstacle.boundary) < 0.0:
                    return False, points, f"EXACT_GOAL_SEGMENT_OUTSIDE_WALL:{obstacle.name}"
            elif isinstance(obstacle, Obstacle):
                gamma, _, _, _ = obstacle.gamma_normal_tangent(
                    point, planner.config.alpha_deg, planner.plan_time_s,
                    planner.config.dynamic_obstacles,
                )
                if not np.isfinite(gamma) or gamma <= planner.config.min_gamma:
                    return False, points, f"EXACT_GOAL_SEGMENT_UNSAFE:{obstacle.name}"
    return True, np.vstack((points, goal)), "PATH_REACHES_MISSION_GOAL"

"""ROS-independent IFDS/UTO transport, pairing, frame, and world contracts."""

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def selected_odometry_topic(gnss_denied: bool, gnss_topic: str, fast_lio_topic: str) -> str:
    """Select the single IFDS mean-position source for the localization mode."""
    return fast_lio_topic if gnss_denied else gnss_topic


@dataclass(frozen=True)
class PathStatus:
    valid: bool
    path_stamp_ns: int
    path_generation: int
    goal_generation: int
    obstacle_generation: int
    planned_at: float
    valid_until: float
    reason: str
    terminal: bool = False

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("status valid must be boolean")
        if type(self.terminal) is not bool or (self.valid and self.terminal):
            raise ValueError("terminal status must be boolean and cannot carry a path")
        integers = (
            self.path_stamp_ns,
            self.path_generation,
            self.goal_generation,
            self.obstacle_generation,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("status stamps and generations must be nonnegative integers")
        if not all(np.isfinite(value) and value >= 0.0 for value in (self.planned_at, self.valid_until)):
            raise ValueError("status times must be finite and nonnegative")
        if self.valid and (self.path_stamp_ns <= 0 or self.valid_until < self.planned_at):
            raise ValueError("valid status requires a live nonzero path stamp")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("status reason is required")

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("invalid IFDS path status schema")
        return cls(**data)


@dataclass(frozen=True)
class PathStatusPair:
    path: object
    status: PathStatus


class PathPairCache:
    """Pair cross-topic Path/status messages without assuming DDS arrival order."""

    def __init__(self) -> None:
        self.paths = {}
        self.statuses = {}

    def add_path(self, stamp_ns: int, path, received: float):
        self.paths[int(stamp_ns)] = (path, float(received))
        return self._take(int(stamp_ns))

    def add_status(self, status: PathStatus, received: float):
        if not status.valid:
            raise ValueError("invalid statuses are immediate events, not pair candidates")
        self.statuses[status.path_stamp_ns] = (status, float(received))
        return self._take(status.path_stamp_ns)

    def _take(self, stamp_ns: int):
        if stamp_ns not in self.paths or stamp_ns not in self.statuses:
            return None
        path, _ = self.paths.pop(stamp_ns)
        status, _ = self.statuses.pop(stamp_ns)
        return PathStatusPair(path, status)

    def expire(self, now: float, timeout: float):
        expired = []
        for cache in (self.paths, self.statuses):
            for stamp, (_, received) in list(cache.items()):
                if now - received > timeout:
                    expired.append(stamp)
                    del cache[stamp]
        return sorted(set(expired))

    def clear(self) -> None:
        self.paths.clear()
        self.statuses.clear()

    def counts(self) -> tuple[int, int]:
        return len(self.paths), len(self.statuses)


@dataclass(frozen=True)
class PathPairTimeout:
    reason: str
    expired_stamp_ns: int
    pending_path_count: int
    pending_status_count: int


def path_pair_timeout_events(cache: PathPairCache, now: float, timeout: float):
    """Expire incomplete pairs and return structured, deterministic diagnostics."""
    stamps = set()
    for entries in (cache.paths, cache.statuses):
        stamps.update(
            stamp
            for stamp, (_, received) in entries.items()
            if now - received > timeout
        )
    events = tuple(
        PathPairTimeout(
            "IFDS_PATH_PAIR_TIMEOUT",
            stamp,
            int(stamp in cache.paths),
            int(stamp in cache.statuses),
        )
        for stamp in sorted(stamps)
    )
    cache.expire(now, timeout)
    return events


def status_matches_path(status: PathStatus, path_stamp_ns: int, now: float) -> bool:
    return bool(
        status.valid
        and status.path_stamp_ns == path_stamp_ns
        and status.planned_at <= now <= status.valid_until
    )


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


def _arc(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _sample(points, spacing, start=None):
    points, arc = _arc(points)
    if start is not None:
        start = np.asarray(start, float)
        best = (np.inf, 0.0)
        for index, delta in enumerate(np.diff(points, axis=0)):
            alpha = np.clip(np.dot(start - points[index], delta) / max(delta @ delta, 1e-12), 0, 1)
            projected = points[index] + alpha * delta
            candidate = (np.linalg.norm(start - projected), arc[index] + alpha * np.linalg.norm(delta))
            if candidate[0] < best[0]:
                best = candidate
        start_arc = best[1]
    else:
        start_arc = 0.0
    if arc[-1] <= start_arc:
        return points[-1:]
    samples = np.arange(start_arc, arc[-1], spacing)
    samples = np.r_[samples, arc[-1]]
    return np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in range(3)])


def _sample_count(points, count):
    points, arc = _arc(points)
    if arc[-1] <= 1e-12:
        return np.repeat(points[-1:], count, axis=0)
    samples = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in range(3)])


class SemanticPathGeneration:
    """Own IFDS generations while ignoring timestamp and along-path start progress."""

    def __init__(self, threshold=0.05, spacing=0.10):
        if threshold <= 0 or spacing <= 0:
            raise ValueError("semantic path thresholds must be positive")
        self.threshold = float(threshold)
        self.spacing = float(spacing)
        self.generation = 0
        self.path = None
        self.goal_generation = -1
        self.obstacle_generation = -1
        self.last_metrics = (np.inf, np.inf)

    def update(self, points, start, goal_generation, obstacle_generation):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        changed_context = (
            goal_generation != self.goal_generation
            or obstacle_generation != self.obstacle_generation
        )
        changed_geometry = self.path is None
        if self.path is not None and not changed_context:
            old = _sample(self.path, self.spacing, start=start)
            new = _sample(points, self.spacing)
            count = max(len(old), len(new), 2)
            old_arc = _sample_count(old, count)
            new_arc = _sample_count(new, count)
            errors = np.linalg.norm(old_arc - new_arc, axis=1)
            maximum = float(np.max(errors)) if len(errors) else np.inf
            rms = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.inf
            self.last_metrics = (maximum, rms)
            changed_geometry = maximum > self.threshold or rms > self.threshold
        if changed_context or changed_geometry:
            self.generation += 1
        self.path = points.copy()
        self.goal_generation = int(goal_generation)
        self.obstacle_generation = int(obstacle_generation)
        return self.generation


def _pose_and_geometry(model, name, reasons):
    pose_element = model.find("pose")
    if pose_element is None or pose_element.get("relative_to"):
        reasons.append(f"unsupported relative pose {name}")
        return None, None
    pose = np.fromstring(pose_element.text or "", sep=" ")
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        reasons.append(f"invalid pose {name}")
        return None, None
    if not np.allclose(pose[3:], 0.0, atol=1e-12):
        reasons.append(f"unsupported rotated model {name}")
        return None, None
    return pose[:3], model.find(".//geometry")


def _validate_wall(item, model, pose, geometry, spans, tolerance, reasons):
    name = str(item["name"])
    if model.findtext("static", "false").strip().lower() != "true":
        reasons.append(f"wall is not static {name}")
    axis_name = str(item.get("axis", "")).lower()
    if axis_name not in ("x", "y", "z"):
        reasons.append(f"invalid wall axis {name}")
        return
    inside_sign = item.get("inside_sign")
    if inside_sign not in (-1, 1, -1.0, 1.0):
        reasons.append(f"invalid wall inside_sign {name}")
        return
    size_text = geometry.findtext("box/size") if geometry is not None else None
    size = np.fromstring(size_text or "", sep=" ")
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        reasons.append(f"invalid wall box size {name}")
        return
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    for other_axis in range(3):
        if other_axis == axis:
            continue
        required = spans.get(other_axis, 0.0)
        if size[other_axis] <= 0.0 or (required > 0.0 and size[other_axis] < required - tolerance):
            reasons.append(f"wall coverage too short {name}")
    expected = pose[axis] + float(inside_sign) * size[axis] / 2.0
    if not np.isclose(expected, float(item.get("boundary", np.nan)), atol=tolerance):
        reasons.append(f"wall boundary/axis/inside_sign mismatch {name}")


def validate_obstacle_world(obstacle_yaml, world_sdf, planning_frame="map", tolerance=1e-5):
    """Bidirectionally validate the original obstacle YAML against an SDF world."""
    try:
        data = yaml.safe_load(Path(obstacle_yaml).read_text()) or {}
        root = ET.parse(world_sdf).getroot()
    except (OSError, ET.ParseError, yaml.YAMLError) as exception:
        return False, [f"world validation input error: {exception}"]
    if (data.get("header") or {}).get("frame_id") != planning_frame:
        return False, ["planning frame mismatch"]
    reasons = []
    items = data.get("obstacles", [])
    if not isinstance(items, list):
        return False, ["obstacles must be a list"]
    yaml_names = [str(item.get("name", "")) for item in items]
    if "" in yaml_names:
        reasons.append("empty YAML obstacle name")
    if len(yaml_names) != len(set(yaml_names)):
        reasons.append("duplicate YAML obstacle name")
    model_list = root.findall(".//model")
    model_names = [str(model.get("name", "")) for model in model_list]
    if len(model_names) != len(set(model_names)):
        reasons.append("duplicate SDF model name")
    models = {model.get("name"): model for model in model_list}
    sdf_obstacle_names = {
        name for name in model_names
        if name.startswith(("corridor_wall_", "static_obstacle_", "dynamic_obstacle_"))
    }
    yaml_name_set = set(yaml_names)
    wall_boundaries = {0: [], 1: [], 2: []}
    for item in items:
        if str(item.get("type", "")).lower() == "wall":
            axis = {"x": 0, "y": 1, "z": 2}.get(str(item.get("axis", "")).lower())
            try:
                boundary = float(item.get("boundary", np.nan))
            except (TypeError, ValueError):
                boundary = np.nan
            if axis is not None and np.isfinite(boundary):
                wall_boundaries[axis].append(boundary)
    spans = {
        axis: max(values) - min(values)
        for axis, values in wall_boundaries.items()
        if len(values) >= 2
    }
    for name in sorted(yaml_name_set - sdf_obstacle_names):
        reasons.append(f"YAML obstacle missing from SDF {name}")
    for name in sorted(sdf_obstacle_names - yaml_name_set):
        reasons.append(f"SDF obstacle missing from YAML {name}")
    for item in items:
        name = str(item.get("name"))
        model = models.get(name)
        if model is None:
            continue
        pose, geometry = _pose_and_geometry(model, name, reasons)
        if pose is None:
            continue
        if str(item.get("type", "superellipsoid")).lower() == "wall":
            _validate_wall(item, model, pose, geometry, spans, tolerance, reasons)
            continue
        sdf_static = model.findtext("static", "false").strip().lower() == "true"
        if sdf_static == bool(item.get("dynamic", False)):
            reasons.append(f"static/dynamic mismatch {name}")
        if not np.allclose(pose, item["center"], atol=tolerance):
            reasons.append(f"center mismatch {name}")
        radii_text = geometry.findtext("ellipsoid/radii") if geometry is not None else None
        radius_text = geometry.findtext("sphere/radius") if geometry is not None else None
        expected_axes = np.asarray(item["axes"], float)
        actual_axes = (
            np.fromstring(radii_text, sep=" ")
            if radii_text
            else np.repeat(float(radius_text), 3) if radius_text else np.array([])
        )
        # Original worlds use SDF ellipsoid radii (semi-axes), not full box size.
        if actual_axes.shape != (3,) or not np.allclose(actual_axes, expected_axes, atol=tolerance):
            reasons.append(f"axes/radii mismatch {name}")
        motion = item.get("motion") or {}
        if item.get("dynamic") and motion.get("type") == "ping_pong":
            plugin = model.find("plugin[@name='ifds::sim::ObstaclePath']")
            if plugin is None:
                reasons.append(f"missing path plugin {name}")
            else:
                for key in ("start", "end"):
                    actual = np.fromstring(plugin.findtext(key, ""), sep=" ")
                    if not np.allclose(actual, motion[key], atol=tolerance):
                        reasons.append(f"{key} mismatch {name}")
                if not np.isclose(float(plugin.findtext("velocity", "nan")), motion["velocity"]):
                    reasons.append(f"velocity mismatch {name}")
        if item.get("dynamic") and motion.get("type") in ("circular_y", "circle_y", "sin_y"):
            plugin = model.find("plugin[@name='ifds::sim::ObstacleOscillator']")
            if plugin is None:
                reasons.append(f"missing oscillator plugin {name}")
            else:
                expected = {
                    "amplitude_y": motion.get("radius", motion.get("radius_y", 0.0)),
                    "angular_speed": motion.get("angular_speed", motion.get("omega", 0.0)),
                    "phase": motion.get("phase", 0.0),
                }
                for key, value in expected.items():
                    if not np.isclose(float(plugin.findtext(key, "nan")), value):
                        reasons.append(f"plugin {key} mismatch {name}")
    return not reasons, reasons

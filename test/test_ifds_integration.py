"""Pure integration-contract tests for the IFDS path-only stage."""

import ast
import json
from pathlib import Path
import numpy as np
import pytest

from uto_ros2.ifds_core import (
    PathStatus,
    SphereObstacle,
    plan_ifds_path,
    status_matches_path,
    transform_position,
)
from uto_ros2.ifds_path_adapter import path_generation
from uto_ros2.planner_runtime import PlannerState, invalid_ifds_path_requires_hold
from uto_ros2.trajectory import Trajectory, TrajectoryBuffer


def test_semantic_path_generation_ignores_stamp_and_quantizes_geometry():
    points = np.array([[0.01, 0.01, 1.01], [1.01, 0.01, 1.01]])
    assert path_generation(1.0, points, 0.05) == path_generation(2.0, points, 0.05)
    jitter = points + np.array([[0.009, -0.009, 0.009], [0.009, -0.009, 0.009]])
    assert path_generation(1.0, points, 0.05) == path_generation(2.0, jitter, 0.05)
    changed = points.copy()
    changed[1, 1] += 0.2
    assert path_generation(1.0, points, 0.05) != path_generation(1.0, changed, 0.05)


def test_path_status_schema_stamp_match_expiry_and_invalid_fail_closed():
    status = PathStatus(True, 123, 4, 2, 1.0, 1.8, "PLAN_OK")
    decoded = PathStatus.from_json(status.to_json())
    assert decoded == status
    assert status_matches_path(status, 123, 1.5)
    assert not status_matches_path(status, 124, 1.5)
    assert not status_matches_path(status, 123, 1.9)
    invalid = PathStatus(False, 0, 4, 2, 2.0, 2.0, "NO_VALID_IFDS_PATH")
    assert not status_matches_path(invalid, 123, 2.0)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        PathStatus.from_json("not-json")


def test_ifds_path_ends_at_exact_goal_and_avoids_sphere():
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([2.0, 0.0, 1.0])
    path = plan_ifds_path(
        start,
        goal,
        [SphereObstacle(np.array([1.0, 0.0, 1.0]), 0.2)],
        clearance=0.25,
        target_threshold=0.05,
    )
    assert np.array_equal(path[-1], goal)
    assert len(path) >= 3


def test_tf_failure_is_not_frame_relabeling():
    transformed = transform_position([1, 0, 0], [2, 0, 0], [0, 0, 0, 1])
    assert np.allclose(transformed, [3, 0, 0])
    with pytest.raises(ValueError):
        transform_position([1, 0, 0], [0, 0, 0], [0, 0, 0, 0])


def test_invalid_status_discards_candidate_and_requires_runtime_hold():
    states = np.zeros((2, 9))
    controls = np.array([[9.81, 0, 0, 0]] * 2)
    candidate = Trajectory([0, 1], states, controls, 1, 1_000_000_000, "path", "map")
    buffer = TrajectoryBuffer()
    buffer.offer(candidate)
    assert invalid_ifds_path_requires_hold(PlannerState.TRAJECTORY_READY)
    assert invalid_ifds_path_requires_hold(PlannerState.EXECUTING)
    assert invalid_ifds_path_requires_hold(PlannerState.REPLANNING)
    assert not invalid_ifds_path_requires_hold(PlannerState.WAIT_IFDS_INITIAL_PATH)
    buffer.discard_candidate()
    assert buffer.candidate is None


def test_ifds_source_has_no_flight_setpoint_publishers_and_setup_registers_entrypoint():
    source = Path("uto_ros2/ifds_planner_node.py").read_text()
    forbidden = ("/mavros/", "/fmu/in/", "setpoint_position", "_hold_setpoint")
    assert not any(token in source for token in forbidden)
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    setup = Path("setup.py").read_text()
    assert "ifds_planner=uto_ros2.ifds_planner_node:main" in setup
    config = Path("config/ifds_planner.yaml").read_text()
    assert "planner_only: true" in config
    assert "target_threshold: 0.05" in config
    ast.parse(source)


def test_launch_contains_single_ifds_uto_px4_chain():
    source = Path("launch/uto_ifds_gazebo.launch.py").read_text()
    tree = ast.parse(source)
    assert tree is not None
    for executable in ("ifds_planner", "uto_planner", "px4_offboard_bridge"):
        assert f'executable="{executable}"' in source
    assert '"planner_only": True' in source

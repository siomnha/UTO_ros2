"""Pure tests for the original-IFDS path-only integration contracts."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from uto_ros2.ifds_contract import (
    PathPairCache, PathStatus, SemanticPathGeneration, status_matches_path,
    selected_odometry_topic, transform_position, validate_obstacle_world,
)
from uto_ros2.ifds_core import IFDSConfig, IFDSPlanner, Obstacle, PlanarWall, obstacle_from_mapping
from uto_ros2.ifds_path_adapter import append_exact_ifds_goal
from uto_ros2.planner_runtime import PlannerState, invalid_ifds_path_requires_hold


def _reference_module():
    path = Path("test/fixtures/original_ifds_core_reference.py")
    spec = importlib.util.spec_from_file_location("original_ifds_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_core_is_numerically_identical_to_original_fixture():
    reference = _reference_module()
    mapping = {"name": "o", "type": "superellipsoid", "center": [2, .4, 1],
               "axes": [.5, .4, 1], "exponents": [1, 1, 1]}
    config = IFDSConfig(target_threshold=.05, max_iterations=200)
    actual = IFDSPlanner(config, [obstacle_from_mapping(mapping)], 0.0).plan(
        np.array([0., 0., 1.]), np.array([4., 0., 1.]))
    expected = reference.IFDSPlanner(
        reference.IFDSConfig(target_threshold=.05, max_iterations=200),
        [reference.obstacle_from_mapping(mapping)], 0.0,
    ).plan(np.array([0., 0., 1.]), np.array([4., 0., 1.]))
    assert actual[0] == expected[0] and actual[2] == expected[2]
    assert np.array_equal(actual[1], expected[1])


def test_original_yaml_schema_preserves_walls_superellipsoids_and_motion():
    data = yaml.safe_load(Path("config/corridor_dynamic_4_obstacles.yaml").read_text())
    parsed = [obstacle_from_mapping(item) for item in data["obstacles"]]
    assert data["header"]["frame_id"] == "map"
    assert any(isinstance(item, PlanarWall) for item in parsed)
    moving = [item for item in parsed if isinstance(item, Obstacle) and item.motion.enabled]
    assert moving and moving[0].motion.motion_type == "ping_pong"
    assert moving[0].axes.shape == (3,) and moving[0].exponents.shape == (3,)
    assert np.linalg.norm(moving[0].velocity_at(0.0, True)) > 0


def test_original_ifds_threshold_result_is_extended_to_exact_goal():
    goal = np.array([1.03, 0.0, 1.0])
    planner = IFDSPlanner(IFDSConfig(target_threshold=.05), [], 0.0)
    found, points, _ = planner.plan(np.array([0.0, 0.0, 1.0]), goal)
    found, points, reason = append_exact_ifds_goal(planner, points, goal)
    assert found and reason == "PATH_REACHES_MISSION_GOAL"
    assert np.array_equal(points[-1], goal)


def test_path_status_pairing_accepts_both_cross_topic_orders_and_expires():
    status = PathStatus(True, 123, 4, 2, 3, 1.0, 1.8, "PLAN_OK")
    cache = PathPairCache()
    assert cache.add_path(123, "path", 1.0) is None
    assert cache.add_status(status, 1.01).path == "path"
    assert cache.add_status(status, 1.1) is None
    assert cache.add_path(123, "path2", 1.11).path == "path2"
    cache.add_path(456, "incomplete", 2.0)
    assert cache.expire(2.21, .2) == [456]
    assert status_matches_path(status, 123, 1.5)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        PathStatus.from_json("not-json")


def test_ifds_semantic_generation_ignores_refresh_and_progress_but_detects_change():
    tracker = SemanticPathGeneration(.05, .1)
    path = np.column_stack((np.linspace(0, 4, 41), np.zeros(41), np.ones(41)))
    first = tracker.update(path, path[0], 1, 1)
    assert tracker.update(path.copy(), path[5], 1, 1) == first
    jittered = path[5:].copy()
    jittered[:, 1] += .01
    assert tracker.update(jittered, path[5], 1, 1) == first
    changed = jittered.copy()
    changed[:, 1] += .2
    assert tracker.update(changed, path[5], 1, 1) == first + 1
    assert tracker.update(changed, path[5], 2, 1) == first + 2


def test_world_yaml_consistency_for_original_corridors():
    pairs = [
        ("config/corridor_static_4_obstacles.yaml", "IFDS_integration_node/world_sdf/my_rgl_corridor_static_4.sdf"),
        ("config/corridor_dynamic_4_obstacles.yaml", "IFDS_integration_node/world_sdf/my_rgl_corridor_dynamic_4.sdf"),
    ]
    for obstacle_yaml, sdf in pairs:
        valid, reasons = validate_obstacle_world(obstacle_yaml, sdf)
        assert valid, reasons


def test_contract_tf_failure_and_fail_closed_runtime_states():
    assert np.allclose(transform_position([1, 0, 0], [2, 0, 0], [0, 0, 0, 1]), [3, 0, 0])
    with pytest.raises(ValueError):
        transform_position([1, 0, 0], [0, 0, 0], [0, 0, 0, 0])
    assert invalid_ifds_path_requires_hold(PlannerState.EXECUTING)
    assert not invalid_ifds_path_requires_hold(PlannerState.WAIT_IFDS_INITIAL_PATH)


def test_gnss_supported_and_denied_select_exactly_one_mean_topic():
    assert selected_odometry_topic(False, "/x500/gnss/odometry", "/Odometry") == "/x500/gnss/odometry"
    assert selected_odometry_topic(True, "/x500/gnss/odometry", "/Odometry") == "/Odometry"


def test_ifds_runtime_is_path_only_and_nested_reference_is_colcon_ignored():
    source = Path("uto_ros2/ifds_planner_node.py").read_text()
    assert not any(token in source for token in ("/mavros/", "/fmu/in/", "CarrotPathTracker"))
    assert "IFDSPlanner(" in source and "append_exact_ifds_goal" in source
    assert Path("IFDS_integration_node/ifds_ros2/COLCON_IGNORE").exists()
    assert "ifds_planner=uto_ros2.ifds_planner_node:main" in Path("setup.py").read_text()
    ast.parse(source)


def test_launch_modes_do_not_default_to_empty_obstacle_map():
    source = Path("launch/uto_ifds_gazebo.launch.py").read_text()
    for token in ("corridor_", "uto_belief_topic", "gnss_denied", "allow_empty_obstacles"):
        assert token in source
    for executable in ("ifds_planner", "uto_planner", "px4_offboard_bridge"):
        assert f'executable="{executable}"' in source

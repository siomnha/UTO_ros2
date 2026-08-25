"""Deterministic tests for global one-shot planning policy."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import yaml

from uto_ros2.ifds_contract import validate_obstacle_world
from uto_ros2.ifds_path_adapter import Polyline, static_ifds_replan_requested
from uto_ros2.planner_runtime import (
    global_one_shot_plan_allowed,
    global_post_solve_commit_time,
    global_preflight_ready,
    px4_status_payload,
    commit_continuity_accepted,
    select_planning_initial_state,
    terminal_segment_matches_goal,
)
from uto_ros2.trajectory import Trajectory


PACKAGE = Path(__file__).resolve().parents[1]


def _belief():
    return SimpleNamespace(
        sigma_states=np.arange(63.0).reshape(7, 9),
        mean_state=np.arange(9.0),
        rotation=np.eye(3),
        covariance=np.eye(6),
    )


def test_global_requires_connected_hold_stable_belief_path_and_goal():
    px4 = {"connected": True, "hold_ready": True, "failsafe": False}
    assert global_preflight_ready(px4, True, True, True)
    for index in range(4):
        conditions = [True, True, True, True]
        conditions[index] = False
        if index == 0:
            changed_px4 = dict(px4, connected=False)
            assert not global_preflight_ready(changed_px4, *conditions[1:])
        else:
            assert not global_preflight_ready(px4, *conditions[1:])
    assert not global_preflight_ready(dict(px4, failsafe=True), True, True, True)


def test_disabled_delay_uses_independent_frozen_belief_and_never_calls_predictor():
    belief = _belief()
    called = []
    result = select_planning_initial_state(
        belief, False, lambda: called.append(True)
    )
    assert called == []
    for actual, expected in zip(
        result,
        (belief.sigma_states, belief.mean_state, belief.rotation, belief.covariance),
    ):
        assert np.array_equal(actual, expected)
        assert actual is not expected


def test_online_delay_path_remains_enabled():
    delayed = (np.ones((7, 9)), np.ones(9), np.eye(3), np.eye(6))
    called = []
    result = select_planning_initial_state(
        _belief(), True, lambda: called.append(True) or delayed
    )
    assert called == [True]
    assert result is delayed


def test_commit_time_is_created_from_solve_completion_not_request_time():
    assert global_post_solve_commit_time(12.5, 0.1) == 12.6


def test_continuity_failure_rejects_candidate_before_execution():
    states = np.zeros((2, 9))
    states[:, 0] = 1.0
    candidate = Trajectory(
        [0.0, 1.0], states, np.zeros((2, 4)), 1, 1_000_000_000, "path", "map"
    )
    belief = SimpleNamespace(mean_state=np.zeros(9), rotation=np.eye(3))
    accepted, errors = commit_continuity_accepted(
        candidate, belief, (0.15, 0.20, 0.10)
    )
    assert not accepted
    assert errors[0] == 1.0


def test_committed_one_shot_blocks_replanning_until_new_goal_resets_flag():
    assert global_one_shot_plan_allowed(False)
    assert not global_one_shot_plan_allowed(True)
    # The planner's new-goal callback resets the committed flag to False.
    assert global_one_shot_plan_allowed(False)


def test_final_segment_uses_explicit_goal_and_references_never_overshoot():
    path = Polyline([[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]])
    references = path.full_mission_references([0.0, 0.0, 1.5], 6, 2.0)
    goal = np.array([10.0, 0.0, 1.5])
    assert np.array_equal(references[-1], goal)
    assert terminal_segment_matches_goal(references, goal, 0.1)
    assert not terminal_segment_matches_goal(references, [11.0, 0.0, 1.5], 0.1)
    detour = Polyline([[0.0, 0.0, 1.5], [5.0, 2.0, 1.5], [10.0, 0.0, 1.5]])
    detour_references = detour.full_mission_references([0.0, 0.0, 1.5], 6, 2.0)
    assert np.array_equal(detour_references[-1], goal)


def test_static_ifds_does_not_replan_for_each_odometry_after_valid_path():
    assert static_ifds_replan_requested(True, True, False)
    assert not static_ifds_replan_requested(True, True, True)
    assert static_ifds_replan_requested(True, False, True)


def test_global_and_online_configuration_keep_distinct_policies():
    global_config = yaml.safe_load((PACKAGE / "config/uto_global.yaml").read_text())
    online_config = yaml.safe_load((PACKAGE / "config/uto_online.yaml").read_text())
    global_params = global_config["uto_planner"]["ros__parameters"]
    online_params = online_config["uto_planner"]["ros__parameters"]
    assert global_params["global_one_shot"] is True
    assert global_params["global_replan_enabled"] is False
    assert global_params["delay_compensation_enabled"] is False
    assert online_params["global_one_shot"] is False
    assert online_params["global_replan_enabled"] is True
    assert online_params["delay_compensation_enabled"] is True


def test_simple_world_matches_installed_obstacle_yaml():
    valid, reasons = validate_obstacle_world(
        PACKAGE / "config/simple_obstacles.yaml",
        PACKAGE / "worlds/my_rgl_simple.sdf",
    )
    assert valid, reasons
    launch = (PACKAGE / "launch/uto_ifds_gazebo.launch.py").read_text()
    setup = (PACKAGE / "setup.py").read_text()
    assert '"simple_obstacles.yaml"' in launch
    assert '"my_rgl_simple.sdf"' in launch
    assert 'glob("worlds/*.sdf")' in setup


def test_humble_parameter_declaration_is_idempotent_and_px4_status_is_json_safe():
    for module in (
        "uto_ros2/uto_planner_node.py",
        "uto_ros2/ifds_planner_node.py",
        "uto_ros2/px4_offboard_bridge_node.py",
    ):
        source = (PACKAGE / module).read_text()
        assert "if not self.has_parameter(name):" in source
    payload = px4_status_payload(
        connected=True,
        hold_ready=True,
        mode="READY",
        setpoint_publish_count=np.int64(40),
        setpoint_max_jitter=np.float64(0.001),
    )
    assert json.loads(json.dumps(payload))["setpoint_publish_count"] == 40

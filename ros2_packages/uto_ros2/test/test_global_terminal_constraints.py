"""Pure regression tests for full-reference and fixed-graph terminal policies."""
import numpy as np
import yaml

from uto_ros2.lgr import lgr_operators
from uto_ros2.planner_runtime import (
    PlanningRequest,
    nearest_equivalent_yaw,
    preflight_rejection_transition,
    PlannerState,
    terminal_parameter_policy,
    wrapped_yaw_error,
)
from uto_ros2.uto_nlp import (
    reference_interpolation_schedule,
    terminal_position_within_tolerance,
)


def test_twenty_references_cover_first_and_last_intervals_on_two_by_five_lgr():
    tau, _ = lgr_operators(5)
    schedule = reference_interpolation_schedule(tau, 2, 20)
    assert len(schedule) == 10
    assert schedule[0][1:3] == (0, 1)
    assert schedule[-1][2] == 19
    assert any(lower >= 10 for _, lower, _, _, _ in schedule)
    assert max(upper for _, _, upper, _, _ in schedule) == 19
    assert all(0.0 <= fraction <= 1.0 for _, _, _, fraction, _ in schedule)


def test_terminal_position_uses_euclidean_ball_not_axis_box():
    assert not terminal_position_within_tolerance(np.array([0.30, 0.081, 0.0]), 0.30)
    assert terminal_position_within_tolerance(np.array([0.20, 0.10, 0.0]), 0.30)


def test_goal_yaw_uses_nearest_periodic_branch_and_wrapped_diagnostics():
    reference = nearest_equivalent_yaw(-np.pi + 0.02, np.pi - 0.01)
    assert abs(reference - (np.pi + 0.02)) < 1e-12
    assert abs(wrapped_yaw_error(np.pi - 0.01, -np.pi + 0.02) + 0.03) < 1e-12


def test_final_and_nonfinal_terminal_parameter_policy():
    final = terminal_parameter_policy(True, 0.6, 4.0, 0.05, 0.03, True, 0.0, 0.4, 0.2)
    assert final["roll_pitch_tolerance"] == 0.05
    assert final["speed_tolerance"] == 0.03
    assert final["yaw_lower"] == -0.2 and final["yaw_upper"] == 0.2
    disabled = terminal_parameter_policy(True, 0.6, 4.0, 0.05, 0.03, False, 0.0, 0.4, 0.2)
    assert disabled["goal_yaw"] is None
    assert disabled["yaw_lower"] < -1e5 and disabled["yaw_upper"] > 1e5
    nonfinal = terminal_parameter_policy(False, 0.6, 4.0, 0.05, 0.03, True, 0.0, 0.4, 0.2)
    assert nonfinal["roll_pitch_tolerance"] == 0.6
    assert np.isclose(nonfinal["speed_tolerance"], np.sqrt(3.0) * 4.0)
    assert nonfinal["goal_yaw"] is None


def test_planning_request_snapshots_optional_goal_yaw_without_breaking_defaults():
    fields = PlanningRequest.__dataclass_fields__
    assert fields["goal_yaw"].default is None
    assert list(fields).index("goal_yaw") > list(fields).index("planner_mode")


def test_global_configuration_exposes_requested_terminal_contract():
    with open("config/uto_global.yaml", encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream)["uto_planner"]["ros__parameters"]
    assert parameters["lookahead_count"] == 20
    assert parameters["regions"] == 2
    assert parameters["lgr_nodes_per_region"] == 5
    assert parameters["sigma_count"] == 7
    assert parameters["goal_yaw_enabled"] is True
    assert parameters["terminal_velocity_tolerance"] == 0.03
    assert parameters["terminal_roll_pitch_tolerance"] == 0.05


def test_maximum_preflight_rejection_leaves_building_state_with_explicit_reason():
    state, reason, blocked = preflight_rejection_transition(3, 3, True)
    assert state == PlannerState.WAIT_IFDS_INITIAL_PATH
    assert reason == "preflight candidate rejected after maximum attempts"
    assert blocked

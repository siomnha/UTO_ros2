"""Pure tests for DTO selection, validation statistics, pairing, and PX4 safety policy."""
from pathlib import Path

import numpy as np
import pytest

from uto_ros2.experiment_runner import (
    ExperimentRunner, InitialStateInjector, paired_trial_specs, readiness_failure, wait_for,
)
from uto_ros2.planner_runtime import (
    PLANNER_PARAMETER_DEFAULTS,
    build_runtime_components,
    px4_bridge_topic_defaults,
    status_dropout_action,
)
from uto_ros2.trajectory import Trajectory
from uto_ros2.uto_nlp import DeterministicNLP, UTONLP, UTOConfig
from uto_ros2.validation import TrialRecord, ValidationStore, build_outputs, method_statistics, read_records


def _record(mode, seed, position, success=True):
    goal = np.array([1.0, 2.0, 3.0])
    position = np.asarray(position, dtype=float)
    error = position - goal
    return TrialRecord(
        f"{seed}_{mode}", mode, seed, seed, success, "" if success else "aborted",
        *goal, *np.zeros(6), *position, *error, float(np.linalg.norm(error)),
        0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 0.1, 0.01, 0.2, 0.01, 0.3, np.nan,
    )


def test_real_deterministic_nlp_has_one_state_trajectory_and_no_covariance_component():
    dto = DeterministicNLP(UTOConfig(sigma=7))
    uto = UTONLP(UTOConfig())
    assert dto.cfg.sigma == 1
    assert dto.planner_mode == "deterministic"
    assert dto.include_terminal_covariance is False
    assert uto.cfg.sigma == 7
    assert uto.include_terminal_covariance is True
    assert dto.cfg.state_scale == uto.cfg.state_scale
    assert dto.cfg.control_scale == uto.cfg.control_scale
    assert PLANNER_PARAMETER_DEFAULTS["planner_mode"] == "uto"


def test_invalid_planner_mode_fails_before_solver_build():
    parameters = dict(PLANNER_PARAMETER_DEFAULTS, planner_mode="invalid")
    with pytest.raises(ValueError, match="planner_mode"):
        build_runtime_components(parameters.__getitem__)


def test_dto_and_uto_trajectory_json_share_bridge_schema():
    for covariance in (None, np.zeros((2, 9, 9))):
        trajectory = Trajectory([0, 1], np.zeros((2, 9)), np.zeros((2, 4)), 1, 0, "p", "map", covariance)
        restored = Trajectory.from_json(trajectory.to_json())
        assert restored.states.shape == (2, 9)
        assert (restored.mean_covariances is None) == (covariance is None)


def test_known_statistics_use_sample_covariance_and_keep_methods_separate():
    dto_positions = np.column_stack((np.arange(10.0), np.zeros(10), np.zeros(10)))
    uto_positions = dto_positions + np.array([1.0, 0.0, 0.0])
    records = [_record("deterministic", i, p) for i, p in enumerate(dto_positions)]
    records += [_record("uto", i, p) for i, p in reversed(list(enumerate(uto_positions)))]
    dto = method_statistics(records, "deterministic")
    expected = np.cov(dto_positions, rowvar=False, ddof=1)
    assert np.allclose(dto["terminal_covariance"], expected)
    summary, matrices = build_outputs(records)
    assert summary["paired_trial_count"] == 10
    assert matrices["terminal_positions_DTO"].shape == (10, 3)
    assert matrices["terminal_positions_UTO"].shape == (10, 3)
    expected_difference = np.array([
        np.linalg.norm(uto_positions[i] - [1, 2, 3]) - np.linalg.norm(dto_positions[i] - [1, 2, 3])
        for i in range(10)
    ])
    assert np.allclose(matrices["paired_error_difference"], expected_difference)


def test_insufficient_covariance_and_aborted_trial_persistence(tmp_path):
    records = [_record("uto", 1, [1, 2, 3]), _record("uto", 2, [1, 2, 3], False)]
    with pytest.warns(UserWarning):
        stats = method_statistics(records, "uto")
    assert stats["terminal_covariance"] is None
    assert stats["trial_count"] == 2 and stats["successful_trial_count"] == 1
    store = ValidationStore(tmp_path)
    for record in records:
        store.append(record)
    assert len(read_records(store.csv_path)) == 2
    assert (tmp_path / "validation_summary.json").exists()
    assert (tmp_path / "validation_matrices.npz").exists()


def test_paired_seed_generation_is_repeatable_and_injection_fails_closed():
    std = [0.1] * 6
    first = paired_trial_specs(2, 1, ["deterministic", "uto"], std)
    second = paired_trial_specs(2, 1, ["deterministic", "uto"], std)
    assert first == second
    assert first[0].initial_error == first[1].initial_error
    assert first[0].initial_error != first[2].initial_error
    assert InitialStateInjector().inject(first[0]) is False


def test_readiness_timeout_and_px4_version_dropout_policy():
    values = iter([0.0, 0.0, 1.1])
    ready, reason = wait_for(lambda: False, 1.0, clock=lambda: next(values), poll_period=0.0)
    assert not ready and reason == "READINESS_TIMEOUT"
    assert readiness_failure({}) == "READINESS_CLOCK_ADVANCING_TIMEOUT"
    assert readiness_failure({
        "clock_advancing": True, "px4_connected": True, "px4_hold_ready": True,
        "belief_stable": True, "ifds_path_valid": True, "trajectory_committed": True,
        "trajectory_finished": True, "goal_reached": True,
    }) == ""
    topics = px4_bridge_topic_defaults()
    assert topics["vehicle_status_topic"].endswith("vehicle_status_v4")
    assert topics["vehicle_local_position_topic"].endswith("vehicle_local_position_v1")
    overridden = dict(topics, vehicle_status_topic="/custom/status")
    assert overridden["vehicle_local_position_topic"] == topics["vehicle_local_position_topic"]
    assert status_dropout_action(False, True) == "TAKEOFF_SAFETY"
    assert status_dropout_action(True, True) == "ABORT_TRAJECTORY_HOLD_LAST"
    assert status_dropout_action(True, False) == "HOLD_LAST"


def test_runner_aborts_unconfirmed_injection_and_cleans_only_owned_process():
    events = []

    class Process:
        def start(self):
            events.append("owned-start")

        def stop(self):
            events.append("owned-stop")

    trial = paired_trial_specs(1, 1, ["uto"], [0.1] * 6)[0]
    runner = ExperimentRunner(lambda spec: Process(), InitialStateInjector(), lambda: True, lambda: True)
    success, reason = runner.run_trial(trial, 1.0, 1.0)
    assert not success and reason == "INITIAL_STATE_INJECTION_UNCONFIRMED"
    assert events == ["owned-start", "owned-stop"]


def test_configuration_and_entry_points_are_installed():
    package = Path(__file__).resolve().parents[1]
    setup = (package / "setup.py").read_text()
    assert "uto_validation_metrics=" in setup
    assert "uto_experiment_runner=" in setup
    assert "summarize_validation=" in setup
    assert "vehicle_status_v1" not in (package / "config/gazebo_harmonic_px4.yaml").read_text()

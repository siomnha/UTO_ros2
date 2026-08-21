import time
from types import SimpleNamespace
import numpy as np
import pytest
from uto_ros2.belief_adapter import (
    reconstruct_belief_from_sigma,
    sanitize_covariance,
    sigma_states,
    simplex_sigma_points,
)
from uto_ros2.dynamics import dynamics
from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.lgr import (
    control_check_grid,
    interpolate_control,
    lgr_operators,
    quadrature_weights,
)
from uto_ros2.math_utils import (
    align_enu_velocity,
    euler_to_rot,
    so3_exp,
    so3_log,
)
from uto_ros2.planner_runtime import (
    CommandState,
    DelayPredictor,
    FeasibilityGate,
    GateConfig,
    LatestWinsWorker,
    PlannerState,
    PlanningRequest,
    Px4CommandSequencer,
    can_resume,
    commit_continuity_errors,
)
from uto_ros2.trajectory import Trajectory, TrajectoryExecution


def make_trajectory(generation=1, commit=10.0):
    states = np.zeros((2, 9))
    states[:, 2] = 1.0
    states[1, 0] = 5.0
    states[1, 8] = 0.4
    return Trajectory(
        [0.0, 1.0],
        states,
        np.array([[9.81, 0, 0, 0]] * 2),
        generation,
        int(commit * 1e9),
        "path",
        "map",
        np.zeros((2, 9, 9)),
    )


def test_simplex_and_so3_reconstruction_with_cross_covariance():
    matrix = np.arange(36, dtype=float).reshape(6, 6) / 100
    covariance = matrix @ matrix.T + np.eye(6) * 0.1
    points, weights = simplex_sigma_points(np.zeros(6), covariance)
    error = points - (points @ weights)[:, None]
    assert np.allclose((error * weights) @ error.T, covariance)
    mean_rotation = so3_exp([0.2, -0.1, 0.3])
    sigma, _ = sigma_states([1, 2, 3], mean_rotation, [0.1, 0.2, 0.3], covariance)
    mean, recovered_rotation, recovered_covariance = reconstruct_belief_from_sigma(sigma)
    assert np.linalg.norm(so3_log(mean_rotation.T @ recovered_rotation)) < 1e-8
    assert np.allclose(recovered_covariance, covariance, atol=1e-8)
    assert np.allclose(mean[:6], [1, 2, 3, 0.1, 0.2, 0.3])


def test_delay_uses_belief_absolute_time_and_process_noise_resamples():
    initial_covariance = np.diag([0.01] * 6)
    sigma, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], initial_covariance)
    queried_times = []

    def control_at(absolute_time):
        queried_times.append(absolute_time)
        return np.array([9.81 if absolute_time < 10.5 else 10.5, 0, 0, 0])

    predictor = DelayPredictor(0.5, 20, 0.1, 2.0, 0.0, 0.0)
    zero_sigma, _, _, zero_covariance = predictor.propagate(
        sigma, 10.0, 11.0, control_at, np.zeros(9)
    )
    noisy_sigma, _, _, noisy_covariance = predictor.propagate(
        sigma, 10.0, 11.0, control_at, np.ones(9) * 0.02
    )
    _, _, reconstructed_zero = reconstruct_belief_from_sigma(zero_sigma)
    _, _, reconstructed_noisy = reconstruct_belief_from_sigma(noisy_sigma)
    assert queried_times[0] == 10.0
    assert max(queried_times) < 11.0
    assert np.allclose(reconstructed_zero, zero_covariance, atol=1e-8)
    assert np.allclose(reconstructed_noisy, noisy_covariance, atol=1e-8)
    assert np.trace(noisy_covariance) > np.trace(zero_covariance)
    with pytest.raises(ValueError):
        predictor.propagate(sigma, 11.0, 10.0, control_at, np.zeros(9))


def test_commit_continuity_includes_so3_geodesic():
    candidate = make_trajectory()
    belief = SimpleNamespace(
        mean_state=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        rotation=euler_to_rot([0, 0, 0]),
    )
    position, velocity, attitude = commit_continuity_errors(candidate, belief)
    assert position == 0 and velocity == 0 and attitude == 0
    belief.rotation = euler_to_rot([0, 0, 0.5])
    assert commit_continuity_errors(candidate, belief)[2] == pytest.approx(0.5)


def test_lgr_dense_control_detects_between_node_overshoot():
    tau, differentiation = lgr_operators(5)
    assert differentiation.shape == (5, 6)
    checks = control_check_grid(tau, 31)
    assert len(np.linspace(-1, 1, 31)) <= len(checks)
    node_values = np.array([[1.0, -1.0, 1.0, -1.0, 1.0]])
    dense = interpolate_control(tau, node_values, checks)
    assert np.max(np.abs(node_values)) <= 1.0
    assert np.max(np.abs(dense)) > 1.0
    weights = quadrature_weights(tau)
    assert np.isclose(weights @ tau**4, 2.0 / 5.0)


def make_gate_result(residual=0.0):
    states = np.zeros((2, 9))
    states[:, 2] = 1.0
    states[1, 0] = 1.0
    return {
        "times": np.array([0.0, 0.5]),
        "states_physical": states,
        "controls_physical": np.array([[9.81, 0, 0, 0]] * 2),
        "sigma_states_physical": np.repeat(states[:, None, :], 7, axis=1),
        "mean_covariances": np.zeros((2, 9, 9)),
        "stats": {"success": True},
        "max_lgr_dynamics_residual": residual,
    }


def test_gate_rejects_real_lgr_residual():
    path = Polyline([[0, 0, 1], [1, 0, 1]])
    request = PlanningRequest(
        1,
        "path",
        1,
        0,
        1,
        np.zeros((9, 7)),
        np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        np.eye(3),
        np.eye(6),
        path.lookahead([0, 0, 1], 10, 0.2),
        path,
    )
    gate = FeasibilityGate(GateConfig(terminal_position_tolerance=0.1))
    assert gate.check(make_gate_result(0.0), request, 1).accepted
    rejected = gate.check(make_gate_result(0.01), request, 1)
    assert not rejected.accepted
    assert "LGR dynamics residual" in rejected.reasons


def test_trajectory_terminal_hold_never_returns_to_takeoff():
    execution = TrajectoryExecution([0, 0, 1.5])
    execution.accept_executable(make_trajectory(commit=10.0))
    active = execution.select(10.8, True, True)
    terminal = execution.select(11.1, True, True)
    assert active.mode == "TRAJECTORY"
    assert terminal.mode == "TERMINAL_HOLD"
    assert np.allclose(terminal.state[:3], [5, 0, 1])
    assert not np.allclose(terminal.state[:3], [0, 0, 1.5])
    assert execution.select(20.0, True, True).mode == "TERMINAL_HOLD"


def test_command_ack_retry_and_fault():
    sequencer = Px4CommandSequencer(True, 176, 400, 0.5, 1.0, 2)
    assert sequencer.tick(0, True, True, False, False, False) == 176
    assert sequencer.tick(0.2, True, True, False, False, False) is None
    assert sequencer.tick(0.5, True, True, False, False, False) == 176
    sequencer.tick(1.1, True, True, False, False, False)
    assert sequencer.state == CommandState.FAULT
    rejected = Px4CommandSequencer(True, 176, 400, 0.5, 1.0, 2)
    assert rejected.tick(0, True, True, False, False, False) == 176
    rejected.acknowledge(176, 1)
    assert rejected.state == CommandState.FAULT
    external = Px4CommandSequencer(False, 176, 400, 0.5, 1.0, 2)
    assert external.tick(0, True, True, True, True, False) is None
    assert external.state == CommandState.TAKEOFF_HOLD


def test_resume_is_conservative_and_fault_is_not_resumable():
    ready = {"connected": True, "failsafe": False, "hold_ready": True}
    assert can_resume(PlannerState.SAFE_HOLD, ready, True, True, True)
    assert not can_resume(PlannerState.SAFE_HOLD, ready, False, True, True)
    assert not can_resume(PlannerState.FAULT, ready, True, True, True)


def test_velocity_yaw_alignment():
    assert np.allclose(
        align_enu_velocity([1, 0, 2], "yaw_offset", np.pi / 2), [0, 1, 2], atol=1e-12
    )
    with pytest.raises(ValueError):
        align_enu_velocity([1, 0, 0], "tf", 0)


def test_first_request_cannot_be_replaced_and_worker_shutdown():
    completed = []
    worker = LatestWinsWorker(
        lambda request: (time.sleep(0.05), request)[1],
        lambda request, result, stale: completed.append((result, stale)),
    )
    assert worker.submit("first", replace_pending=False)
    time.sleep(0.01)
    assert not worker.submit("duplicate-first", replace_pending=False)
    time.sleep(0.08)
    assert completed == [("first", False)]
    assert worker.shutdown()


def test_yaml_files_have_no_duplicate_parameter_keys():
    from pathlib import Path

    for path in Path("config").glob("*.yaml"):
        keys = []
        for line in path.read_text().splitlines():
            if line.startswith("    ") and ":" in line:
                keys.append(line.strip().split(":", 1)[0])
        assert len(keys) == len(set(keys)), path


def test_hover_dynamics_and_covariance_sanitize():
    assert np.allclose(dynamics(np.zeros(9), [9.81, 0, 0, 0]), 0)
    with pytest.raises(ValueError):
        sanitize_covariance(np.zeros((6, 6)))


def test_bridge_execution_policy_returns_exactly_one_output_each_cycle():
    execution = TrajectoryExecution([0, 0, 1.5])
    outputs = [execution.select(now, False, False) for now in (0.0, 0.025, 0.05)]
    assert len(outputs) == 3
    assert all(output.mode == "TAKEOFF_HOLD" for output in outputs)

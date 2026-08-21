import time
from types import SimpleNamespace
import numpy as np
import pytest
from uto_ros2.belief_adapter import (
    BeliefStableDetector,
    joint_sigma_process_update,
    reconstruct_joint_tangent_from_sigma,
    StabilityConfig,
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
    CommitOutcome,
    ExecutionDecision,
    DelayPredictor,
    FeasibilityGate,
    GateConfig,
    LatestWinsWorker,
    PlannerState,
    PlanningRequest,
    Px4CommandSequencer,
    can_resume,
    execution_decision,
    goal_generation_changed,
    mission_goal_input_valid,
    mission_goal_satisfied,
    commit_continuity_errors,
    commit_due_status,
    offboard_control_flags,
    update_goal_dwell,
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
    _, _, reconstructed_zero, _ = reconstruct_joint_tangent_from_sigma(zero_sigma)
    _, _, reconstructed_noisy, _ = reconstruct_joint_tangent_from_sigma(noisy_sigma)
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
    tau, _ = lgr_operators(5)
    endpoints = np.repeat(states[:1], 7, axis=0)
    return {
        "times": np.array([0.0, 0.5]),
        "states_physical": states,
        "controls_physical": np.array([[9.81, 0, 0, 0]] * 2),
        "sigma_states_physical": np.repeat(states[:, None, :], 7, axis=1),
        "mean_covariances": np.zeros((2, 9, 9)),
        "stats": {"success": True},
        "max_lgr_dynamics_residual": residual,
        "physical_control_blocks": [np.repeat(np.array([[9.81], [0], [0], [0.0]]), 5, axis=1)],
        "region_endpoint_sigma_physical": [endpoints],
        "lgr_nodes": tau,
        "horizon": 0.5,
        "regions": 1,
    }


def test_gate_rejects_real_lgr_residual():
    path = Polyline([[0, 0, 1], [1, 0, 1]])
    request = PlanningRequest(
        1,
        "path",
        1,
        0,
        1,
        np.repeat(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]]).T, 7, axis=1),
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
    worker = LatestWinsWorker(lambda request: (time.sleep(0.05), request)[1])
    assert worker.submit("first", replace_pending=False)
    time.sleep(0.01)
    assert not worker.submit("duplicate-first", replace_pending=False)
    time.sleep(0.08)
    events = worker.drain_completions()
    assert len(events) == 1 and events[0].result == "first" and not events[0].stale
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


def test_delay_preserves_propagated_velocity_dispersion():
    covariance = np.diag([1e-8, 1e-8, 1e-8, 0.04, 0.04, 0.04])
    sigma, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], covariance)
    assert np.max(np.ptp(sigma[3:6], axis=1)) == 0
    predictor = DelayPredictor(0.5, 20, 0.1, 2.0, 0.0, 0.0)
    propagated, mean, _, _ = predictor.propagate(
        sigma, 1.0, 1.5, lambda _: np.array([9.81, 0, 0, 0]), np.ones(9) * 0.01
    )
    assert np.max(np.ptp(propagated[3:6], axis=1)) > 1e-5
    assert np.allclose(propagated[3:6].mean(axis=1), mean[3:6])


def test_belief_stability_uses_so3_yaw_wrap():
    detector = BeliefStableDetector(StabilityConfig(samples=2, mean_delta=np.deg2rad(5)))
    covariance = np.eye(6) * 0.001
    assert not detector.update(
        0.0, 0.0, np.zeros(3), euler_to_rot([0, 0, np.deg2rad(179)]), covariance
    )
    assert detector.update(
        0.01, 0.01, np.zeros(3), euler_to_rot([0, 0, np.deg2rad(-179)]), covariance
    )


def test_commit_frequency_lateness_and_goal_helpers():
    trajectory = make_trajectory(commit=10.0)
    assert commit_due_status(9.99, trajectory, 0.04)[0] == "waiting"
    assert commit_due_status(10.02, trajectory, 0.04)[0] == "due"
    assert commit_due_status(10.05, trajectory, 0.04)[0] == "late"
    since, reached = update_goal_dwell(None, 10.0, True, 0.0, 1.0)
    assert not reached
    _, reached = update_goal_dwell(since, 11.01, True, 0.0, 1.0)
    assert reached


def test_offboard_control_levels_are_mutually_exclusive():
    assert offboard_control_flags("position") == {
        "position": True,
        "velocity": False,
        "acceleration": False,
        "attitude": False,
        "body_rate": False,
    }
    assert offboard_control_flags("velocity")["velocity"]
    assert offboard_control_flags("acceleration")["acceleration"]
    with pytest.raises(ValueError):
        offboard_control_flags("attitude")


def test_trajectory_yaw_interpolation_uses_short_wrapped_path():
    trajectory = make_trajectory(commit=0.0)
    trajectory.states[0, 8] = np.deg2rad(179)
    trajectory.states[1, 8] = np.deg2rad(-179)
    yaw = trajectory.sample(0.5)[0][8]
    assert abs(abs(yaw) - np.pi) < np.deg2rad(2)


def test_dense_rollout_rejects_endpoint_mismatch():
    path = Polyline([[0, 0, 1], [1, 0, 1]])
    sigma = np.repeat(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]]).T, 7, axis=1)
    request = PlanningRequest(
        1,
        "path",
        1,
        0,
        1,
        sigma,
        sigma.mean(axis=1),
        np.eye(3),
        np.eye(6),
        path.lookahead([0, 0, 1], 10, 0.2),
        path,
    )
    result = make_gate_result(0.0)
    result["region_endpoint_sigma_physical"][0][:, 0] += 2.0
    checked = FeasibilityGate(GateConfig(terminal_position_tolerance=0.1)).check(result, request, 1)
    assert not checked.accepted
    assert "rollout endpoint position" in checked.reasons


def test_rejected_new_path_candidate_does_not_replace_active_generation():
    from uto_ros2.trajectory import TrajectoryBuffer

    buffer = TrajectoryBuffer()
    old = make_trajectory(generation=1, commit=0.0)
    old.path_generation = "old-path"
    buffer.offer(old)
    buffer.commit_candidate()
    new = make_trajectory(generation=2, commit=1.0)
    new.path_generation = "new-path"
    buffer.offer(new)
    buffer.discard_candidate()
    assert buffer.active.path_generation == "old-path"


def test_runtime_defaults_separate_planning_and_commit_rates():
    from uto_ros2.planner_runtime import PLANNER_PARAMETER_DEFAULTS

    assert PLANNER_PARAMETER_DEFAULTS["replan_rate"] == 1.5
    assert PLANNER_PARAMETER_DEFAULTS["commit_check_rate"] == 50.0


def test_legacy_runtime_modules_are_absent():
    from pathlib import Path

    legacy = tuple(
        "_".join(parts) + ".py"
        for parts in (
            ("async", "worker"),
            ("delay", "compensator"),
            ("feasibility", "gate"),
            ("state", "machine"),
            ("trajectory", "buffer"),
            ("planner", "core"),
        )
    )
    assert not any((Path("uto_ros2") / name).exists() for name in legacy)


def test_commit_timer_source_never_calls_solver():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("uto_ros2/uto_planner_node.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_on_commit_timer"
    )
    called = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "solve" not in called and "_solve_request" not in called


def test_commit_outcome_prioritizes_due_candidate_over_expired_active():
    decision = execution_decision(CommitOutcome.COMMITTED, True, 0.0, False)
    assert decision == ExecutionDecision.COMMIT
    assert execution_decision(CommitOutcome.LATE, True, 0.0, False) == ExecutionDecision.SAFE_HOLD
    assert (
        execution_decision(CommitOutcome.REJECTED, True, 0.0, False) == ExecutionDecision.SAFE_HOLD
    )
    assert (
        execution_decision(CommitOutcome.NONE, True, 0.0, False, True)
        == ExecutionDecision.CONTINUE
    )


def test_completion_processing_cannot_publish_safe_hold_directly():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("uto_ros2/uto_planner_node.py").read_text())
    completion = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_on_solve_complete"
    )
    calls = {
        node.func.attr
        for node in ast.walk(completion)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_enter_safe_hold" not in calls


def test_dense_endpoint_uses_separate_so3_position_velocity_errors():
    path = Polyline([[0, 0, 1], [1, 0, 1]])
    sigma = np.repeat(np.array([[0, 0, 1, 0, 0, 0, 0, 0, np.deg2rad(179)]]).T, 7, axis=1)
    request = PlanningRequest(
        1,
        "path",
        1,
        0,
        1,
        sigma,
        sigma.mean(axis=1),
        np.eye(3),
        np.eye(9),
        path.lookahead([0, 0, 1], 10, 0.2),
        path,
    )
    result = make_gate_result(0.0)
    result["region_endpoint_sigma_physical"][0][:] = sigma.T
    result["region_endpoint_sigma_physical"][0][:, 8] = np.deg2rad(-179)
    dense = FeasibilityGate(GateConfig())._dense_rollout(result, request)
    assert dense[4] < 1e-8 and dense[5] < 1e-8
    assert dense[6] == pytest.approx(np.deg2rad(2), abs=2e-3)
    result["region_endpoint_sigma_physical"][0][:, 0] += 1.0
    assert FeasibilityGate(GateConfig())._dense_rollout(result, request)[4] > 0.9
    result["region_endpoint_sigma_physical"][0][:, 0] -= 1.0
    result["region_endpoint_sigma_physical"][0][:, 3] += 1.0
    assert FeasibilityGate(GateConfig())._dense_rollout(result, request)[5] > 0.9
    result["region_endpoint_sigma_physical"][0][:, 3] -= 1.0
    result["region_endpoint_sigma_physical"][0][:, 8] = 0.0
    assert FeasibilityGate(GateConfig())._dense_rollout(result, request)[6] > 3.0


def test_joint_sigma_update_preserves_cross_covariance_and_identity():
    base, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], np.diag([0.02] * 6))
    # Inject correlated velocity using the existing latent pose/attitude vertices.
    base[3] = 2.0 * (base[0] - base[0].mean())
    base[4] = -1.5 * (base[7] - base[7].mean())
    before_mean, _, before_cov, _ = reconstruct_joint_tangent_from_sigma(base)
    unchanged, unchanged_mean, _, unchanged_cov = joint_sigma_process_update(base, np.zeros((6, 6)))
    assert np.allclose(unchanged, base)
    assert np.allclose(unchanged_cov, before_cov)
    assert np.allclose(unchanged_mean, before_mean)
    process = np.diag([0.005] * 6)
    updated, updated_mean, _, updated_cov = joint_sigma_process_update(base, process)
    assert np.all(np.isfinite(updated_cov)) and np.allclose(updated_cov, updated_cov.T)
    assert np.allclose(updated_mean[:6], before_mean[:6], atol=1e-8)
    assert np.linalg.norm(updated_cov[:3, 3:6]) > 1e-4
    assert np.linalg.norm(updated_cov[6:9, 3:6]) > 1e-4
    assert np.trace(updated_cov[np.ix_([0, 1, 2, 6, 7, 8], [0, 1, 2, 6, 7, 8])]) > np.trace(
        before_cov[np.ix_([0, 1, 2, 6, 7, 8], [0, 1, 2, 6, 7, 8])]
    )
    assert np.max(np.ptp(updated[3:6], axis=1)) > 0


def test_latency_estimator_cold_p90_clipping_and_margins():
    predictor = DelayPredictor(0.5, 4, 0.1, 2.0, 0.02, 0.03, 0.01, 0.05, 1.0)
    assert predictor.estimate_mode == "cold"
    assert predictor.estimate() == pytest.approx(0.56)
    predictor.record_latency(0.1)
    predictor.record_latency(0.2)
    predictor.record_latency(10.0)  # clipped to 1.0
    assert predictor.estimate_mode == "steady_p90"
    expected = np.percentile([0.1, 0.2, 1.0], 90) + 0.02 + 0.03 + 0.01
    assert predictor.estimate() == pytest.approx(expected)
    predictor.record_latency(0.8)  # includes a deterministic fake dense/queue delay
    assert predictor.percentile90() >= 0.8
    with pytest.raises(ValueError):
        predictor.record_latency(float("nan"))
    with pytest.raises(ValueError):
        predictor.record_latency(-0.1)


def test_explicit_mission_goal_not_local_path_endpoint_and_yaw_wrap():
    belief = SimpleNamespace(
        position=np.array([5.0, 0, 1]),
        velocity=np.zeros(3),
        mean_state=np.array([5.0, 0, 1, 0, 0, 0, 0, 0, np.deg2rad(179)]),
    )
    mission_goal = SimpleNamespace(position=np.array([10.0, 0, 1]), yaw=np.deg2rad(-179))
    assert not mission_goal_satisfied(belief, mission_goal, 0.2, 0.2, False, 0.1)
    belief.position = mission_goal.position.copy()
    belief.mean_state[:3] = belief.position
    assert mission_goal_satisfied(belief, mission_goal, 0.2, 0.2, True, np.deg2rad(3))
    assert not mission_goal_satisfied(belief, mission_goal, 0.2, 0.2, True, np.deg2rad(1))


def test_mission_goal_validation_and_generation_semantics():
    assert mission_goal_input_valid(9.0, 10.0, "map", "map", [1, 2, 3], 0.0)
    assert not mission_goal_input_valid(0.0, 10.0, "map", "map", [1, 2, 3], 0.0)
    assert not mission_goal_input_valid(9.0, 10.0, "odom", "map", [1, 2, 3], 0.0)
    assert not mission_goal_input_valid(1.0, 10.0, "map", "map", [1, 2, 3], 1.0)
    old = SimpleNamespace(generation="1,2,3,0")
    assert not goal_generation_changed(old, "1,2,3,0")
    assert goal_generation_changed(old, "4,5,6,0")

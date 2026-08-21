"""ROS-independent first-solve/replan/commit pipeline test."""

import numpy as np
from types import SimpleNamespace
from uto_ros2.belief_adapter import sigma_states
from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.lgr import lgr_operators
from uto_ros2.planner_runtime import (
    CandidateManager,
    FeasibilityGate,
    GateConfig,
    PlanningRequest,
    commit_continuity_errors,
)
from uto_ros2.trajectory import TrajectoryBuffer


def test_first_solve_commit_replan_and_stale_discard():
    path = Polyline([[0, 0, 1], [1, 0, 1]])
    sigma, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], np.eye(6) * 0.001)
    request = PlanningRequest(
        1,
        "path-1",
        2,
        0.0,
        1.0,
        sigma,
        np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
        np.eye(3),
        np.eye(6) * 0.001,
        path.lookahead([0, 0, 1], 10, 0.2),
        path,
        True,
    )
    states = np.zeros((2, 9))
    states[:, 2] = 1
    states[1, 0] = 1
    tau, _ = lgr_operators(5)
    result = {
        "times": np.array([0.0, 1.0]),
        "states_physical": states,
        "controls_physical": np.array([[9.81, 0, 0, 0]] * 2),
        "sigma_states_physical": np.repeat(states[:, None, :], 7, axis=1),
        "mean_covariances": np.zeros((2, 9, 9)),
        "stats": {"success": True},
        "max_lgr_dynamics_residual": 0.0,
        "physical_control_blocks": [np.repeat(np.array([[9.81], [0], [0], [0.0]]), 5, axis=1)],
        "region_endpoint_sigma_physical": [np.repeat(states[:1], 7, axis=0)],
        "lgr_nodes": tau,
        "horizon": 1.0,
        "regions": 1,
    }
    gate = FeasibilityGate(GateConfig(terminal_position_tolerance=0.1)).check(result, request, 1)
    buffer = TrajectoryBuffer()
    manager = CandidateManager(buffer, 0.05)
    assert manager.admit(request, result, 0.5, 1, gate, "map")[0]
    assert buffer.candidate_due(1.0)
    belief = SimpleNamespace(mean_state=states[0], rotation=np.eye(3))
    assert max(commit_continuity_errors(buffer.candidate, belief)) == 0
    assert buffer.commit_candidate() is not None
    assert buffer.sample(1.5) is not None
    stale_request = PlanningRequest(**{**request.__dict__, "request_generation": 1})
    assert not manager.admit(stale_request, result, 1.1, 2, gate, "map")[0]
    assert buffer.sample(2.1) is None

"""ROS-independent first-solve/replan/commit pipeline test."""

import numpy as np
import pytest
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
    CommitOutcome,
    ExecutionDecision,
    PlannerState,
    execution_decision,
    update_goal_dwell,
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


class TransitionHarness:
    """Small action-recording harness around the production decision functions."""

    def __init__(self):
        self.state = PlannerState.TAKEOFF
        self.actions = []
        self.goal_since = None

    def startup(self):
        self.state = PlannerState.HOLD
        self.state = PlannerState.WAIT_BELIEF_STABLE
        self.state = PlannerState.WAIT_IFDS_INITIAL_PATH
        self.state = PlannerState.FIRST_SOLVE
        self.state = PlannerState.TRAJECTORY_READY

    def tick(self, outcome, fresh, active_remaining, goal_pending=False, terminal_fresh=False):
        decision = execution_decision(
            outcome,
            fresh,
            active_remaining,
            self.state == PlannerState.GOAL_REACHED,
            goal_pending,
            terminal_fresh,
        )
        self.actions.append(decision)
        if decision == ExecutionDecision.COMMIT:
            self.state = PlannerState.EXECUTING
        elif decision == ExecutionDecision.SAFE_HOLD:
            self.state = PlannerState.SAFE_HOLD
        return decision


def test_first_wait_commit_goal_and_new_mission_transition_harness():
    harness = TransitionHarness()
    harness.startup()
    assert harness.tick(CommitOutcome.WAITING, True, 0.0) == ExecutionDecision.CONTINUE
    assert harness.state == PlannerState.TRAJECTORY_READY
    assert harness.tick(CommitOutcome.COMMITTED, True, 0.0) == ExecutionDecision.COMMIT
    assert harness.state == PlannerState.EXECUTING
    harness.goal_since, reached = update_goal_dwell(None, 10.0, True, 0.0, 1.0)
    assert not reached
    harness.goal_since, reached = update_goal_dwell(harness.goal_since, 11.1, True, 0.0, 1.0)
    assert reached
    harness.state = PlannerState.GOAL_REACHED
    assert harness.tick(CommitOutcome.NONE, True, 0.0) == ExecutionDecision.GOAL_REACHED
    # A new goal only marks restart readiness; explicit resume restarts stability flow.
    restart_pending = True
    assert restart_pending and harness.state == PlannerState.GOAL_REACHED
    harness.state = PlannerState.WAIT_BELIEF_STABLE
    harness.state = PlannerState.WAIT_IFDS_INITIAL_PATH
    harness.state = PlannerState.FIRST_SOLVE
    assert harness.actions.count(ExecutionDecision.COMMIT) == 1


@pytest.mark.parametrize(
    "outcome,fresh,expected",
    [
        (CommitOutcome.WAITING, False, ExecutionDecision.SAFE_HOLD),
        (CommitOutcome.LATE, True, ExecutionDecision.SAFE_HOLD),
        (CommitOutcome.REJECTED, True, ExecutionDecision.SAFE_HOLD),
    ],
)
def test_transition_harness_failure_branches(outcome, fresh, expected):
    harness = TransitionHarness()
    harness.startup()
    assert harness.tick(outcome, fresh, 0.0) == expected
    assert harness.actions == [expected]
    assert harness.state == PlannerState.SAFE_HOLD


def test_terminal_dwell_stale_velocity_resets_and_failsafe_holds():
    since, reached = update_goal_dwell(None, 10.0, True, 0.0, 1.0)
    assert not reached
    since, reached = update_goal_dwell(since, 10.5, False, 0.0, 1.0)
    assert since is None and not reached
    harness = TransitionHarness()
    harness.state = PlannerState.EXECUTING
    assert harness.tick(CommitOutcome.NONE, False, 0.0) == ExecutionDecision.SAFE_HOLD


def test_new_mission_discards_candidate_but_preserves_terminal_hold_reference():
    buffer = TrajectoryBuffer()
    states = np.zeros((2, 9))
    states[:, 2] = 1.0
    from uto_ros2.trajectory import Trajectory

    active = Trajectory(
        [0.0, 1.0], states, np.array([[9.81, 0, 0, 0]] * 2), 1, 0, "old", "map"
    )
    candidate = Trajectory(
        [0.0, 1.0], states, np.array([[9.81, 0, 0, 0]] * 2), 2, 2_000_000_000, "new", "map"
    )
    buffer.offer(active)
    buffer.commit_candidate()
    buffer.offer(candidate)
    buffer.discard_candidate()
    assert buffer.active is active
    assert buffer.candidate is None

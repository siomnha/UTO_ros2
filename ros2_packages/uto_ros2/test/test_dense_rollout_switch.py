"""Regression tests for the startup-only dense rollout gate switch."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.planner_runtime import (
    PLANNER_PARAMETER_DEFAULTS,
    FeasibilityGate,
    GateConfig,
    PlanningRequest,
)


def _request() -> PlanningRequest:
    path = Polyline([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    initial = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return PlanningRequest(
        1,
        "path",
        1,
        0.0,
        1.0,
        np.repeat(initial[:, None], 7, axis=1),
        initial,
        np.eye(3),
        np.eye(6),
        path.lookahead(initial[:3], 10, 0.2),
        path,
    )


def _result() -> dict:
    states = np.zeros((2, 9))
    states[:, 2] = 1.0
    states[1, 0] = 1.0
    sigma = np.repeat(states[:, None, :], 7, axis=1)
    return {
        "times": np.array([0.0, 0.5]),
        "states_physical": states,
        "controls_physical": np.array([[9.81, 0.0, 0.0, 0.0]] * 2),
        "sigma_states_physical": sigma,
        "stats": {"success": True},
        "max_lgr_dynamics_residual": 0.0,
    }


def _disabled_gate(**overrides) -> FeasibilityGate:
    settings = {"enable_dense_rollout_gate": False, "terminal_position_tolerance": 0.1}
    settings.update(overrides)
    return FeasibilityGate(GateConfig(**settings))


def test_dense_rollout_defaults_enabled_and_is_called(monkeypatch):
    assert PLANNER_PARAMETER_DEFAULTS["enable_dense_rollout_gate"] is True
    gate = FeasibilityGate(GateConfig(terminal_position_tolerance=0.1))
    calls = []

    def dense(result, request):
        calls.append((result, request))
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    monkeypatch.setattr(gate, "_dense_rollout", dense)
    checked = gate.check(_result(), _request(), 1)
    assert checked.accepted
    assert len(calls) == 1
    assert checked.dense_rollout_gate_enabled is True
    assert checked.dense_rollout_gate_skipped is False
    assert checked.max_dense_mean_path_error == 0.0


def test_disabled_dense_rollout_is_not_called_and_metrics_are_none(monkeypatch):
    gate = _disabled_gate()

    def must_not_run(*args, **kwargs):
        raise AssertionError("RK4 dense rollout was called while disabled")

    monkeypatch.setattr(gate, "_dense_rollout", must_not_run)
    checked = gate.check(_result(), _request(), 1)
    assert checked.accepted
    assert checked.dense_rollout_gate_enabled is False
    assert checked.dense_rollout_gate_skipped is True
    assert checked.max_dense_mean_path_error is None
    assert checked.max_dense_sigma_path_error is None
    assert checked.max_dense_velocity is None
    assert checked.max_dense_attitude is None
    assert checked.max_dense_endpoint_position_error is None
    assert checked.max_dense_endpoint_velocity_error is None
    assert checked.max_dense_endpoint_attitude_error is None
    assert not any("dense" in reason or "rollout endpoint" in reason for reason in checked.reasons)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda value: value["stats"].update(success=False), "solver status"),
        (lambda value: value.update(max_lgr_dynamics_residual=1.0), "LGR dynamics residual"),
        (lambda value: value["states_physical"].__setitem__((1, 0), 0.5), "terminal position"),
        (lambda value: value["controls_physical"].__setitem__((0, 0), 20.0), "control upper bound"),
        (lambda value: value["states_physical"].__setitem__((0, 1), 2.0), "mean path tube"),
        (lambda value: value["sigma_states_physical"].__setitem__((0, 0, 1), 2.0), "sigma path tube"),
    ],
)
def test_disabled_dense_rollout_preserves_non_dense_rejections(mutation, expected_reason):
    result = _result()
    mutation(result)
    checked = _disabled_gate().check(result, _request(), 1)
    assert not checked.accepted
    assert expected_reason in checked.reasons
    assert not any(reason.startswith("dense ") for reason in checked.reasons)


def test_mode_yaml_values_are_explicit():
    package = Path(__file__).resolve().parents[1]
    global_parameters = yaml.safe_load((package / "config/uto_global.yaml").read_text())
    online_parameters = yaml.safe_load((package / "config/uto_online.yaml").read_text())
    assert global_parameters["uto_planner"]["ros__parameters"]["enable_dense_rollout_gate"] is False
    assert online_parameters["uto_planner"]["ros__parameters"]["enable_dense_rollout_gate"] is True

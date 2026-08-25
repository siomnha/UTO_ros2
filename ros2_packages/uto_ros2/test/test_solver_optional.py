import importlib.util
import numpy as np
import pytest
from uto_ros2.belief_adapter import sigma_states
from uto_ros2.uto_nlp import DeterministicNLP, UTONLP, UTOConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="CasADi/IPOPT unavailable"
)


def test_deterministic_graph_is_single_trajectory_without_covariance_output():
    nlp = DeterministicNLP(
        UTOConfig(regions=1, nodes=2, references=2, max_iter=100, terminal_position_tolerance=0.5)
    ).build()
    initial = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=float).T
    references = np.array([[0, 0, 1], [0, 0, 1]], dtype=float)
    nlp.set_parameters(initial, references, 0.5, [0, 0, 0], [-1] * 3, [1] * 3, 1,
                       [1, 1, 999, 0.01, 1e-6, 1e-6])
    result = nlp.solve()
    assert result["sigma_states_physical"].shape[1] == 1
    assert result["terminal_covariance"] is None
    assert result["objective_components"]["terminal_covariance"] == 0.0
    assert result["planner_mode"] == "deterministic"


def test_casadi_lgr_build_solve_and_parameter_reuse():
    cfg = UTOConfig(
        regions=1, nodes=2, sigma=7, references=2, max_iter=100, terminal_position_tolerance=0.5
    )
    nlp = UTONLP(cfg).build()
    graph = id(nlp.opti)
    nlp.build()
    assert id(nlp.opti) == graph and nlp.build_count == 1
    covariance = np.diag([1e-4] * 6)
    initial, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], covariance)
    refs = np.array([[0, 0, 1], [0, 0, 1]])
    nlp.set_parameters(
        initial, refs, 0.5, [0, 0, 0], [-1] * 3, [1] * 3, 1, [1, 1, 1, 0.01, 1e-6, 1e-6]
    )
    first = nlp.solve()
    assert (
        first["build_count"] == 1
        and first["states_physical"].shape[1] == 9
        and np.unique(first["sigma_states_physical"][0].round(7), axis=0).shape[0] > 1
    )
    assert (
        np.all(np.isfinite(first["terminal_covariance"]))
        and np.linalg.eigvalsh(first["terminal_covariance"]).min() > -1e-8
    )
    changed = refs.copy()
    changed[-1, 0] = 0.1
    nlp.set_parameters(
        initial, changed, 0.5, [0, 0, 0], [-1] * 3, [1] * 3, 0, [1, 1, 1, 0.01, 1e-6, 1e-6]
    )
    second = nlp.solve()
    assert nlp.build_count == 1
    assert not np.allclose(first["states_physical"], second["states_physical"])


def test_extracted_lgr_residual_detects_tampered_state():
    config = UTOConfig(
        regions=1,
        nodes=2,
        sigma=7,
        references=2,
        max_iter=100,
        terminal_position_tolerance=0.5,
    )
    nlp = UTONLP(config).build()
    covariance = np.diag([1e-4] * 6)
    initial, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], covariance)
    references = np.array([[0, 0, 1], [0, 0, 1]])
    nlp.set_parameters(
        initial,
        references,
        0.5,
        [0, 0, 0],
        [-1] * 3,
        [1] * 3,
        1,
        [1, 1, 1, 0.01, 1e-6, 1e-6],
    )
    result = nlp.solve()
    original = nlp.compute_residual(result)
    result["normalized_state_blocks"][0][0][0, 0] += 0.1
    assert nlp.compute_residual(result) > original + 1e-4


def test_default_2x5x7_graph_reuses_solver_and_reports_timing():
    config = UTOConfig(max_iter=200, terminal_position_tolerance=0.5)
    nlp = UTONLP(config).build()
    covariance = np.diag([1e-4] * 6)
    initial, _ = sigma_states([0, 0, 1], np.eye(3), [0, 0, 0], covariance)
    references = np.repeat(np.array([[0.0, 0.0, 1.0]]), 10, axis=0)
    nlp.set_parameters(
        initial,
        references,
        1.0,
        [0, 0, 0],
        [-1] * 3,
        [1] * 3,
        1,
        [1, 1, 1, 0.01, 1e-6, 1e-6],
    )
    first = nlp.solve()
    first_time = nlp.solve_time
    nlp.set_parameters(
        initial,
        references,
        1.1,
        [0, 0, 0],
        [-1] * 3,
        [1] * 3,
        1,
        [1, 1, 1, 0.01, 1e-6, 1e-6],
    )
    second = nlp.solve()
    assert nlp.build_count == 1
    assert first["states_physical"].shape == (11, 9)
    assert first["max_lgr_dynamics_residual"] < 1e-4
    assert np.all(np.isfinite(first["terminal_covariance"]))
    assert np.linalg.eigvalsh(first["terminal_covariance"]).min() > -1e-8
    assert nlp.build_time > 0 and first_time > 0 and nlp.solve_time > 0
    assert first["iterations"] >= 0 and second["iterations"] >= 0
    print({"build": nlp.build_time, "first_solve": first_time, "second_solve": nlp.solve_time})

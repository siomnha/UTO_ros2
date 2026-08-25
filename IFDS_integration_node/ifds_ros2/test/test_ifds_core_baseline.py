import math

import numpy as np

from ifds_ros2.ifds_core import IFDSConfig, IFDSPlanner, Obstacle


def test_safety_margin_does_not_enlarge_axes_and_delta_g_sets_safeguard():
    obstacle = Obstacle.from_mapping(
        {
            'name': 'safe',
            'center': [0.0, 0.0, 0.0],
            'axes': [2.0, 3.0, 4.0],
            'safety_margin': 5.0,
        }
    )
    planner = IFDSPlanner(IFDSConfig(delta_g=2.0), [obstacle])

    assert np.allclose(obstacle.axes, [2.0, 3.0, 4.0])
    assert obstacle.rstar == 2.0
    assert math.isfinite(planner._rho_star(gamma=25.0, rstar=obstacle.rstar, rho0=2.5))


def test_dynamic_obstacle_center_refreshes_between_planning_cycles():
    obstacle = Obstacle.from_mapping(
        {
            'name': 'moving',
            'center': [0.0, -5.0, 0.0],
            'axes': [1.0, 1.0, 1.0],
            'dynamic': True,
            'motion': {'type': 'sin_y', 'radius': 4.0, 'angular_speed': 0.25, 'phase': 0.0},
        }
    )

    center_t0 = obstacle.center_at(0.0, dynamic_obstacles=True)
    center_t1 = obstacle.center_at(math.pi / 0.5, dynamic_obstacles=True)

    assert not np.allclose(center_t0, center_t1)
    assert np.allclose(obstacle.center_at(100.0, dynamic_obstacles=False), [0.0, -5.0, 0.0])


def test_modulation_matrix_stays_finite_for_large_gamma_and_exponents():
    obstacle = Obstacle.from_mapping(
        {
            'name': 'sharp',
            'center': [0.0, 0.0, 0.0],
            'axes': [1.0, 1.0, 1.0],
            'exponents': [20.0, 20.0, 20.0],
        }
    )
    config = IFDSConfig(sigma0=0.01, min_gamma=1.01)
    planner = IFDSPlanner(config, [obstacle])
    gamma, normal, tangent, center = obstacle.gamma_normal_tangent(
        np.array([20.0, 2.0, 1.5]), config.alpha_deg, 0.0, False
    )
    matrix = planner._modulation_matrix(
        gamma=gamma,
        normal=normal,
        tangent=tangent,
        dist=50.0,
        dist_obj=float(np.linalg.norm(np.array([20.0, 2.0, 1.5]) - center)),
        rstar=obstacle.rstar,
        rho0=config.rho0,
        sigma0=config.sigma0,
    )

    assert math.isfinite(gamma)
    assert np.all(np.isfinite(matrix))

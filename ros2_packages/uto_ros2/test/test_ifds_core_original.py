import math

import numpy as np

from uto_ros2.ifds_core import IFDSConfig, IFDSPlanner, Obstacle


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
from pathlib import Path

import numpy as np
import yaml

from uto_ros2.ifds_core import (
    IFDSConfig,
    IFDSPlanner,
    Obstacle,
    PlanarWall,
    obstacle_from_mapping,
)


PACKAGE_ROOT = Path(__file__).parents[1]


def _upper_wall():
    return PlanarWall('upper', axis=1, boundary=2.425, inside_sign=-1.0)


def _velocity(position, goal, *, gain=1.5, influence=1.0):
    planner = IFDSPlanner(
        IFDSConfig(
            cruise_speed=2.0,
            wall_modulation_gain=gain,
            wall_influence_distance=influence,
        ),
        [_upper_wall()],
    )
    return planner._modulated_velocity(np.asarray(position), np.asarray(goal), step=0)


def test_old_format_obstacle_still_defaults_to_superellipsoid():
    mapping = {'name': 'legacy', 'center': [0, 0, 0], 'axes': [1, 2, 3]}
    parsed = obstacle_from_mapping(mapping)
    original = Obstacle.from_mapping(mapping)

    assert isinstance(parsed, Obstacle)
    assert parsed.name == original.name
    assert np.array_equal(parsed.center, original.center)
    assert np.array_equal(parsed.axes, original.axes)
    assert np.array_equal(parsed.exponents, original.exponents)
    position = np.array([-3.0, 0.0, 0.0])
    goal = np.array([3.0, 0.0, 0.0])
    assert np.allclose(
        IFDSPlanner(IFDSConfig(), [parsed])._modulated_velocity(position, goal, 0),
        IFDSPlanner(IFDSConfig(), [original])._modulated_velocity(position, goal, 0),
    )


def test_wall_is_omitted_outside_influence_distance():
    actual = _velocity([0.0, 1.0, 0.0], [0.0, 3.0, 0.0])
    assert np.allclose(actual, [0.0, 2.0, 0.0])


def test_wall_only_modulates_velocity_toward_boundary():
    incoming = _velocity([0.0, 2.0, 0.0], [0.0, 3.0, 0.0], gain=1.0)
    outgoing = _velocity([0.0, 2.0, 0.0], [0.0, 1.0, 0.0], gain=1.0)

    assert 0.0 < incoming[1] < 2.0
    assert np.allclose(outgoing, [0.0, -2.0, 0.0])


def test_wall_gain_strengthens_only_normal_component():
    position = [0.0, 2.0, 0.0]
    goal = [2.0, 4.0, 0.0]
    gain_one = _velocity(position, goal, gain=1.0)
    gain_two = _velocity(position, goal, gain=2.0)

    assert gain_two[1] < gain_one[1]
    assert np.isclose(gain_two[0], gain_one[0])


def test_outside_wall_is_a_clear_planning_failure():
    planner = IFDSPlanner(IFDSConfig(), [_upper_wall()])
    found, _, reason = planner.plan(np.array([0.0, 2.5, 0.0]), np.array([2.0, 2.5, 0.0]))

    assert not found
    assert reason == 'outside corridor boundary: upper'


def test_corridor_yaml_counts_and_shared_static_geometry():
    static = yaml.safe_load((PACKAGE_ROOT / 'config/corridor_static_4_obstacles.yaml').read_text())['obstacles']
    dynamic = yaml.safe_load((PACKAGE_ROOT / 'config/corridor_dynamic_4_obstacles.yaml').read_text())['obstacles']

    assert len(static) == 13
    assert len(dynamic) == 17
    assert sum(item['type'] == 'wall' for item in static) == 4
    assert sum(item['type'] == 'superellipsoid' for item in static) == 9
    assert static == dynamic[:13]
    moving = dynamic[13:]
    assert [item['name'] for item in moving] == [
        'dynamic_obstacle_3', 'dynamic_obstacle_5', 'dynamic_obstacle_6', 'dynamic_obstacle_8'
    ]
    assert all(item['motion']['type'] == 'ping_pong' for item in moving)
    assert all(item['motion']['velocity'] == 0.5 for item in moving)
    parsed_wall = obstacle_from_mapping(static[0])
    assert isinstance(parsed_wall, PlanarWall)
    assert (parsed_wall.axis, parsed_wall.boundary, parsed_wall.inside_sign) == (1, 2.425, -1.0)
import math

import numpy as np

from uto_ros2.ifds_core import Obstacle


def _dynamic_obstacle():
    return Obstacle.from_mapping(
        {
            'name': 'osci',
            'center': [0.0, -5.0, 0.0],
            'axes': [1.0, 1.0, 1.0],
            'dynamic': True,
            'motion': {
                'type': 'sin_y',
                'radius': 4.0,
                'radius_z': 0.0,
                'angular_speed': 0.25,
                'phase': 0.0,
            },
        }
    )


def test_dynamic_obstacle_center_and_velocity_are_current_time_only():
    obstacle = _dynamic_obstacle()

    center_t0 = obstacle.center_at(0.0, dynamic_obstacles=True)
    velocity_t0 = obstacle.velocity_at(0.0, dynamic_obstacles=True)
    center_later = obstacle.center_at(math.pi / 0.5, dynamic_obstacles=True)

    assert np.allclose(center_t0, [0.0, -5.0, 0.0])
    assert np.allclose(velocity_t0, [0.0, 1.0, 0.0])
    assert not np.allclose(center_t0, center_later)


def test_static_mode_ignores_dynamic_motion_and_velocity():
    obstacle = _dynamic_obstacle()

    assert np.allclose(obstacle.center_at(100.0, dynamic_obstacles=False), [0.0, -5.0, 0.0])
    assert np.allclose(obstacle.velocity_at(100.0, dynamic_obstacles=False), [0.0, 0.0, 0.0])


def test_ping_pong_matches_gazebo_path_positions_and_velocity_reversal():
    obstacle = Obstacle.from_mapping(
        {
            'name': 'dynamic_obstacle_3',
            'center': [13.0, 1.0, 0.9],
            'axes': [0.25, 0.25, 0.9],
            'dynamic': True,
            'motion': {
                'type': 'ping_pong',
                'start': [13.0, 1.0, 0.9],
                'end': [13.0, -1.0, 0.9],
                'velocity': 0.5,
            },
        }
    )

    expected = {
        0.0: [13.0, 1.0, 0.9],
        1.0: [13.0, 0.5, 0.9],
        4.0: [13.0, -1.0, 0.9],
        5.0: [13.0, -0.5, 0.9],
        8.0: [13.0, 1.0, 0.9],
    }
    for time_s, center in expected.items():
        assert np.allclose(obstacle.center_at(time_s, dynamic_obstacles=True), center)

    assert np.allclose(obstacle.velocity_at(1.0, True), [0.0, -0.5, 0.0])
    assert np.allclose(obstacle.velocity_at(4.0, True), [0.0, 0.5, 0.0])
    assert np.allclose(obstacle.velocity_at(5.0, True), [0.0, 0.5, 0.0])
import numpy as np
import pytest

from uto_ros2.ifds_core import IFDSConfig, IFDSPlanner, Obstacle


POSITION = np.array([-3.0, 0.0, 0.0])
GOAL = np.array([3.0, 0.0, 0.0])


def _obstacle(*, angular_speed=0.0, phase=0.0):
    return Obstacle.from_mapping(
        {
            'name': 'test',
            'center': [0.0, 0.0, 0.0],
            'axes': [1.0, 1.0, 1.0],
            'dynamic': angular_speed != 0.0,
            'motion': {
                'type': 'sin_y',
                'radius': 2.0,
                'angular_speed': angular_speed,
                'phase': phase,
            },
        }
    )


def _velocity(mode, obstacle, *, dynamic=False):
    planner = IFDSPlanner(
        IFDSConfig(velocity_mode=mode, dynamic_obstacles=dynamic, optimizer_mode=0),
        [obstacle],
        plan_time_s=0.0,
    )
    return planner._modulated_velocity(POSITION, GOAL, step=0)


def test_normal_mode_matches_existing_matrix_aggregation():
    obstacle = _obstacle()
    planner = IFDSPlanner(IFDSConfig(velocity_mode='normal'), [obstacle])
    u = np.array([planner.config.cruise_speed, 0.0, 0.0])
    gamma, normal, tangent, center = obstacle.gamma_normal_tangent(POSITION, 0.0, 0.0, False)
    matrix = planner._modulation_matrix(
        gamma, normal, tangent, np.linalg.norm(POSITION - GOAL),
        np.linalg.norm(POSITION - center), obstacle.rstar, planner.rho, planner.sigma
    )

    assert np.allclose(planner._modulated_velocity(POSITION, GOAL, 0), matrix @ u)


def test_relative_equals_normal_for_static_obstacle():
    obstacle = _obstacle()
    assert np.allclose(_velocity('relative', obstacle), _velocity('normal', obstacle))


def test_relative_identity_contribution_does_not_drag_with_obstacle():
    # The obstacle is moving along +y, while the nominal +x flow is outgoing at
    # this position. Its identity contribution must therefore remain nominal u.
    obstacle = _obstacle(angular_speed=1.0)
    position = np.array([3.0, 0.0, 0.0])
    goal = np.array([6.0, 0.0, 0.0])
    planner = IFDSPlanner(IFDSConfig(velocity_mode='relative', dynamic_obstacles=True), [obstacle])

    assert np.allclose(planner._modulated_velocity(position, goal, 0), [2.0, 0.0, 0.0])


def test_approaching_moving_obstacle_changes_relative_modulation():
    # At phase pi the obstacle is centered at the origin and travels along -y,
    # toward the vehicle. This increases the closing speed in the relative frame.
    obstacle = _obstacle(angular_speed=1.0, phase=np.pi)
    position = np.array([0.0, -3.0, 0.0])
    goal = np.array([0.0, 3.0, 0.0])
    normal_planner = IFDSPlanner(IFDSConfig(velocity_mode='normal', dynamic_obstacles=True), [obstacle])
    relative_planner = IFDSPlanner(IFDSConfig(velocity_mode='relative', dynamic_obstacles=True), [obstacle])
    normal = normal_planner._modulated_velocity(position, goal, 0)
    relative = relative_planner._modulated_velocity(position, goal, 0)

    assert not np.allclose(relative, normal)


def test_relative_equals_normal_for_moving_model_with_zero_velocity():
    obstacle = _obstacle(angular_speed=1.0, phase=np.pi / 2.0)
    assert np.allclose(
        _velocity('relative', obstacle, dynamic=True),
        _velocity('normal', obstacle, dynamic=True),
    )


def test_invalid_velocity_mode_is_rejected():
    with pytest.raises(ValueError, match='unsupported velocity_mode'):
        IFDSPlanner(IFDSConfig(velocity_mode='invalid'), [])

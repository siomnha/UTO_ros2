import numpy as np
import pytest

from ifds_ros2.ifds_core import IFDSConfig, IFDSPlanner, Obstacle


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

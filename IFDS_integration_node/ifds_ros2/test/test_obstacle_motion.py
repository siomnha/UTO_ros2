import math

import numpy as np

from ifds_ros2.ifds_core import Obstacle


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

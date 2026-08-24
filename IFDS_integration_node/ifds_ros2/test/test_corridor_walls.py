from pathlib import Path

import numpy as np
import yaml

from ifds_ros2.ifds_core import (
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

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]


def _obstacle_block(name, filename='known_obstacles_osci.yaml'):
    text = (PACKAGE_ROOT / 'config' / filename).read_text(encoding='utf-8')
    match = re.search(rf'^  - name: {name}\n(?P<body>.*?)(?=^  - name:|\Z)', text, re.MULTILINE | re.DOTALL)
    assert match is not None
    return match.group('body')


def _list_value(block, key):
    match = re.search(rf'^    {key}: \[([^]]+)\]$', block, re.MULTILINE)
    assert match is not None
    return [float(value.strip()) for value in match.group(1).split(',')]


def _motion_value(block, key):
    match = re.search(rf'^      {key}: ([-+0-9.]+)$', block, re.MULTILINE)
    assert match is not None
    return float(match.group(1))


def _world_model(root, name):
    model = root.find(f"./world/model[@name='{name}']")
    assert model is not None
    return model


def _geometry_radii(model, geometry_kind):
    ellipsoid = model.findtext(f'./link/{geometry_kind}/geometry/ellipsoid/radii')
    if ellipsoid is not None:
        return [float(value) for value in ellipsoid.split()]
    radius = float(model.findtext(f'./link/{geometry_kind}/geometry/sphere/radius'))
    return [radius, radius, radius]


def test_osci_world_has_all_yaml_obstacles_with_matching_geometry():
    root = ET.parse(PACKAGE_ROOT / 'worlds' / 'my_rgl_world_osci.sdf').getroot()
    for name in ('obstacle_1', 'obstacle_2', 'obstacle_3', 'obstacle_4'):
        block = _obstacle_block(name)
        model = _world_model(root, name)
        pose = [float(value) for value in model.findtext('pose').split()[:3]]
        assert pose == _list_value(block, 'center')
        assert _geometry_radii(model, 'visual') == _list_value(block, 'axes')
        assert _geometry_radii(model, 'collision') == _list_value(block, 'axes')
    assert root.find('./world/actor') is None


def test_second_and_third_obstacles_match_yaml_lateral_motion():
    root = ET.parse(PACKAGE_ROOT / 'worlds' / 'my_rgl_world_osci.sdf').getroot()
    for name in ('obstacle_2', 'obstacle_3'):
        block = _obstacle_block(name)
        assert '    dynamic: true\n' in block
        assert '      radius_z: 0.0\n' in block

        plugin = _world_model(root, name).find(
            "./plugin[@name='ifds::sim::ObstacleOscillator']"
        )
        model = _world_model(root, name)
        assert model.findtext('static') == 'false'
        assert model.findtext('./link/kinematic') == 'true'
        assert model.findtext('./link/gravity') == 'false'
        assert plugin is not None
        assert plugin.attrib['filename'] == 'libifds_obstacle_oscillator.so'
        assert math.isclose(float(plugin.findtext('amplitude_y')), _motion_value(block, 'radius'))
        assert math.isclose(
            float(plugin.findtext('angular_speed')), _motion_value(block, 'angular_speed')
        )
        assert math.isclose(float(plugin.findtext('phase')), _motion_value(block, 'phase'))


def test_first_and_fourth_obstacles_are_static_without_oscillator():
    root = ET.parse(PACKAGE_ROOT / 'worlds' / 'my_rgl_world_osci.sdf').getroot()
    for name in ('obstacle_1', 'obstacle_4'):
        model = _world_model(root, name)
        assert model.findtext('static') == 'true'
        assert model.find('./plugin') is None


def test_static_and_osci_yaml_have_same_four_geometries():
    for name in ('obstacle_1', 'obstacle_2', 'obstacle_3', 'obstacle_4'):
        static = _obstacle_block(name, 'known_obstacles.yaml')
        oscillating = _obstacle_block(name)
        assert _list_value(static, 'center') == _list_value(oscillating, 'center')
        assert _list_value(static, 'axes') == _list_value(oscillating, 'axes')

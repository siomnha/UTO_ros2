"""Repository layout and Gazebo plugin build contract checks."""
from pathlib import Path


def test_runtime_packages_are_standard_siblings():
    root = Path(__file__).resolve().parents[3]
    packages = root / "ros2_packages"
    assert (packages / "uto_ros2/package.xml").exists()
    assert (packages / "ifds_gz_plugins/package.xml").exists()
    assert (packages / "rgl_livox_converter/package.xml").exists()
    assert not (root / "package.xml").exists()
    assert (root / "IFDS_integration_node/ifds_ros2/COLCON_IGNORE").exists()


def test_gazebo_plugin_names_install_location_and_environment_hook_match_sdf():
    root = Path(__file__).resolve().parents[3]
    plugin = root / "ros2_packages/ifds_gz_plugins"
    cmake = (plugin / "CMakeLists.txt").read_text()
    hook = (plugin / "env-hooks/ifds_gz_plugins.dsv.in").read_text()
    world = (root / "ros2_packages/uto_ros2/worlds/my_rgl_corridor_dynamic_4.sdf").read_text()
    for target in ("ifds_obstacle_path", "ifds_obstacle_oscillator"):
        assert f"add_library({target} SHARED" in cmake
        assert target in cmake
    assert "LIBRARY DESTINATION lib" in cmake
    assert "GZ_SIM_SYSTEM_PLUGIN_PATH;lib" in hook
    assert 'filename="libifds_obstacle_path.so"' in world

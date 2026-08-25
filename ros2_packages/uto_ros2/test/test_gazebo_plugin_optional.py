"""Opt-in Gazebo Harmonic motion smoke test; never reports unavailable Gazebo as pass."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.gazebo


def _obstacle_y(text):
    block = re.search(
        r'pose\s*\{.*?name:\s*"dynamic_obstacle_3".*?position\s*\{(.*?)\}',
        text,
        re.DOTALL,
    )
    if block is None:
        raise AssertionError("dynamic_obstacle_3 pose missing from Gazebo pose topic")
    value = re.search(r"\by:\s*([-+0-9.eE]+)", block.group(1))
    if value is None:
        raise AssertionError("dynamic_obstacle_3 y coordinate missing")
    return float(value.group(1))


def test_dynamic_obstacle_path_plugin_moves_with_original_ping_pong_model():
    if os.environ.get("UTO_RUN_GAZEBO_TESTS") != "1":
        pytest.skip("set UTO_RUN_GAZEBO_TESTS=1 to run the Gazebo runtime smoke test")
    pytest.importorskip("numpy", reason="Gazebo motion comparison requires NumPy")
    from uto_ros2.ifds_core import ObstacleMotion
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Harmonic 'gz' executable is unavailable")
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ROS 2 CLI is unavailable")
    prefix = subprocess.check_output(
        [ros2, "pkg", "prefix", "ifds_gz_plugins"], text=True
    ).strip()
    libraries = [
        Path(prefix) / "lib/libifds_obstacle_path.so",
        Path(prefix) / "lib/libifds_obstacle_oscillator.so",
    ]
    if not all(library.exists() for library in libraries):
        pytest.skip("IFDS Gazebo plugin libraries have not both been built/sourced")
    plugin_path = os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(os.pathsep)
    assert str(Path(prefix) / "lib") in plugin_path
    world = Path(__file__).resolve().parents[1] / "worlds/my_rgl_corridor_dynamic_4.sdf"
    process = subprocess.Popen(
        [gz, "sim", "-r", "-s", str(world)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(2.0)
        assert process.poll() is None, process.stdout.read()
        command = [
            gz, "topic", "-e", "-n", "1", "-t",
            "/world/my_rgl_corridor_dynamic_4/pose/info",
        ]
        first = subprocess.check_output(
            command,
            text=True,
            timeout=10,
        )
        time.sleep(1.0)
        second = subprocess.check_output(
            command,
            text=True,
            timeout=10,
        )
        y_first = _obstacle_y(first)
        y_second = _obstacle_y(second)
        assert abs(y_second - y_first) > 0.05
        motion = ObstacleMotion.from_mapping({
            "dynamic": True,
            "center": [13.0, 1.0, 0.9],
            "motion": {"type": "ping_pong", "start": [13.0, 1.0, 0.9],
                       "end": [13.0, -1.0, 0.9], "velocity": 0.5},
        })
        expected_direction = motion._ping_pong(1.0)[1][1]
        assert (y_second - y_first) * expected_direction > 0.0
        assert abs(abs(y_second - y_first) - abs(expected_direction)) < 0.35
    finally:
        process.terminate()
        process.wait(timeout=5)

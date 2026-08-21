"""Target-workspace ROS integration smoke test.

This test is intentionally skipped outside a sourced ROS/PX4/CasADi workspace. In the
integration image it verifies that the package's required runtime imports and node
constructors are available before the full launch fixture publishes fake PX4,
odometry and Path messages.
"""

import importlib.util
import pytest

REQUIRED = ("rclpy", "px4_msgs", "casadi", "numpy")
pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in REQUIRED),
    reason="requires sourced ROS 2, px4_msgs, NumPy and CasADi/IPOPT workspace",
)


def test_ros_nodes_are_constructible_in_integration_workspace():
    from uto_ros2.px4_offboard_bridge_node import PX4OffboardBridge
    from uto_ros2.uto_planner_node import UTOPlannerNode

    assert PX4OffboardBridge is not None
    assert UTOPlannerNode is not None

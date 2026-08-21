"""ROS/PX4 workspace construction smoke test; skipped when dependencies are absent."""

import importlib.util
import pytest

REQUIRED = ("rclpy", "px4_msgs", "casadi", "numpy")
pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in REQUIRED),
    reason="requires sourced ROS 2, px4_msgs, NumPy and CasADi/IPOPT workspace",
)


def test_ros_nodes_construct_and_expose_independent_timers():
    import rclpy
    from uto_ros2.px4_offboard_bridge_node import PX4OffboardBridge
    from uto_ros2.uto_planner_node import UTOPlannerNode

    rclpy.init()
    planner = bridge = None
    try:
        planner = UTOPlannerNode()
        bridge = PX4OffboardBridge()
        assert planner.planning_timer is not planner.commit_timer
        assert planner.get_parameter("commit_check_rate").value == 50.0
        assert bridge.get_parameter("setpoint_rate").value == 40.0
        assert bridge.get_parameter("offboard_control_level").value == "position"
    finally:
        if planner is not None:
            planner.destroy_node()
        if bridge is not None:
            bridge.destroy_node()
        rclpy.shutdown()

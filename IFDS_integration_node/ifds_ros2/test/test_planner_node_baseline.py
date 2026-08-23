import sys
import threading
import types

import numpy as np


def _install_ros_fakes(monkeypatch):
    class Header:
        def __init__(self):
            self.stamp = None
            self.frame_id = ''

    class Position:
        x = 0.0
        y = 0.0
        z = 0.0

    class Orientation:
        x = 0.0
        y = 0.0
        z = 0.0
        w = 1.0

    class Pose:
        def __init__(self):
            self.position = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    class PoseStamped:
        def __init__(self):
            self.header = Header()
            self.pose = Pose()

    class Odometry:
        def __init__(self):
            self.header = Header()
            self.pose = types.SimpleNamespace(pose=Pose())

    class NavPath:
        def __init__(self):
            self.header = Header()
            self.poses = []

    class String:
        def __init__(self):
            self.data = ''

    class Node:
        pass

    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    callback_groups = types.ModuleType('rclpy.callback_groups')
    callback_groups.MutuallyExclusiveCallbackGroup = object
    callback_groups.ReentrantCallbackGroup = object
    executors = types.ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = object
    node_module = types.ModuleType('rclpy.node')
    node_module.Node = Node

    geometry_msgs = types.ModuleType('geometry_msgs')
    geometry_msgs_msg = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg.PoseStamped = PoseStamped
    nav_msgs = types.ModuleType('nav_msgs')
    nav_msgs_msg = types.ModuleType('nav_msgs.msg')
    nav_msgs_msg.Odometry = Odometry
    nav_msgs_msg.Path = NavPath
    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.String = String

    for name, module in {
        'rclpy': rclpy,
        'rclpy.callback_groups': callback_groups,
        'rclpy.executors': executors,
        'rclpy.node': node_module,
        'geometry_msgs': geometry_msgs,
        'geometry_msgs.msg': geometry_msgs_msg,
        'nav_msgs': nav_msgs,
        'nav_msgs.msg': nav_msgs_msg,
        'std_msgs': std_msgs,
        'std_msgs.msg': std_msgs_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return PoseStamped


def test_hold_setpoint_stream_continues_after_planning_failure(monkeypatch):
    PoseStamped = _install_ros_fakes(monkeypatch)
    from ifds_ros2.ifds_planner_node import IFDSPlannerNode
    from ifds_ros2.path_tracker import CarrotPathTracker

    published = []
    node = IFDSPlannerNode.__new__(IFDSPlannerNode)
    node.frame_id = 'map'
    node.current_pose = PoseStamped()
    node.current_pose.pose.position.x = 1.0
    node.current_pose.pose.position.y = 2.0
    node.current_pose.pose.position.z = 3.0
    node.hold_position = np.array([1.0, 2.0, 3.0])
    node.tracker = CarrotPathTracker(lookahead_distance=2.0)
    node.tracker_lock = threading.Lock()
    node.setpoint_pub = types.SimpleNamespace(publish=lambda msg: published.append(msg))
    node.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: 'stamp'))

    node._setpoint_timer_cb()
    node._setpoint_timer_cb()

    assert len(published) == 2
    assert [published[-1].pose.position.x, published[-1].pose.position.y, published[-1].pose.position.z] == [1.0, 2.0, 3.0]

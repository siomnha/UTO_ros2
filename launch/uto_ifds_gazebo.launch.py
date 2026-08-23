import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _nodes(context):
    mode = LaunchConfiguration("mode").perform(context)
    if mode not in ("global", "online"):
        raise RuntimeError("mode must be 'global' or 'online'")
    share = get_package_share_directory("uto_ros2")
    ifds_params = LaunchConfiguration("ifds_params").perform(context)
    if not ifds_params:
        ifds_params = os.path.join(share, "config", "ifds_planner.yaml")
    obstacle_yaml = LaunchConfiguration("ifds_obstacles").perform(context)
    common = {
        "use_sim_time": True,
        "planning_frame": "map",
    }
    return [
        Node(
            package="uto_ros2",
            executable="ifds_planner",
            name="ifds_planner",
            output="screen",
            parameters=[
                ifds_params,
                common,
                {
                    "planner_only": True,
                    "obstacle_yaml": obstacle_yaml,
                    "gnss_denied": ParameterValue(
                        LaunchConfiguration("gnss_denied"), value_type=bool
                    ),
                    "dynamic_obstacles": ParameterValue(
                        LaunchConfiguration("dynamic_obstacles"), value_type=bool
                    ),
                    "goal_topic": "/ifds/goal",
                    "mission_goal_topic": "/ifds/mission_goal",
                    "path_topic": "/ifds/local_path",
                    "path_status_topic": "/ifds/path_status",
                },
            ],
        ),
        Node(
            package="uto_ros2",
            executable="uto_planner",
            name="uto_planner",
            output="screen",
            parameters=[os.path.join(share, "config", f"uto_{mode}.yaml"), common],
        ),
        Node(
            package="uto_ros2",
            executable="px4_offboard_bridge",
            name="px4_offboard_bridge",
            output="screen",
            parameters=[os.path.join(share, "config", "gazebo_harmonic_px4.yaml"), common],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="online", choices=["global", "online"]),
            DeclareLaunchArgument("ifds_params", default_value=""),
            DeclareLaunchArgument("ifds_obstacles", default_value=""),
            DeclareLaunchArgument("gnss_denied", default_value="true", choices=["true", "false"]),
            DeclareLaunchArgument(
                "dynamic_obstacles", default_value="false", choices=["true", "false"]
            ),
            OpaqueFunction(function=_nodes),
        ]
    )

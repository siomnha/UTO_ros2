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
        raise RuntimeError("mode must be global or online")
    share = get_package_share_directory("uto_ros2")
    ifds_params = LaunchConfiguration("ifds_params").perform(context)
    if not ifds_params:
        ifds_params = os.path.join(share, "config", f"ifds_{mode}.yaml")
    obstacles = LaunchConfiguration("ifds_obstacles").perform(context)
    if not obstacles:
        obstacles = os.path.join(
            share, "config", f"corridor_{'dynamic' if mode == 'online' else 'static'}_4_obstacles.yaml"
        )
    world = LaunchConfiguration("world_sdf").perform(context)
    if not world:
        world = os.path.join(
            share, "worlds", f"my_rgl_corridor_{'dynamic' if mode == 'online' else 'static'}_4.sdf"
        )
    uto_common = {"use_sim_time": True, "planning_frame": "map"}
    return [
        Node(
            package="uto_ros2", executable="ifds_planner", name="ifds_planner", output="screen",
            parameters=[ifds_params, {
                "use_sim_time": True, "frame_id": "map", "planner_only": True,
                "obstacles_yaml": obstacles, "world_sdf": world,
                "gnss_denied": ParameterValue(LaunchConfiguration("gnss_denied"), value_type=bool),
                "dynamic_obstacles": mode == "online",
                "validate_world_consistency": ParameterValue(
                    LaunchConfiguration("validate_world_consistency"), value_type=bool),
                "allow_empty_obstacles": ParameterValue(
                    LaunchConfiguration("allow_empty_obstacles"), value_type=bool),
            }],
        ),
        Node(
            package="uto_ros2", executable="uto_planner", name="uto_planner", output="screen",
            parameters=[os.path.join(share, "config", f"uto_{mode}.yaml"), uto_common, {
                "belief_topic": LaunchConfiguration("uto_belief_topic"),
                "path_topic": "/ifds/local_path", "path_status_topic": "/ifds/path_status",
                "mission_goal_topic": "/ifds/mission_goal",
            }],
        ),
        Node(
            package="uto_ros2", executable="px4_offboard_bridge", name="px4_offboard_bridge",
            output="screen", parameters=[
                os.path.join(share, "config", "gazebo_harmonic_px4.yaml"),
                {"use_sim_time": True},
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="online", choices=["global", "online"]),
        DeclareLaunchArgument("gnss_denied", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("uto_belief_topic", default_value="/Odometry"),
        DeclareLaunchArgument("ifds_params", default_value=""),
        DeclareLaunchArgument("ifds_obstacles", default_value=""),
        DeclareLaunchArgument("world_sdf", default_value=""),
        DeclareLaunchArgument("validate_world_consistency", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("allow_empty_obstacles", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=_nodes),
    ])

"""Start the passive metrics node and paired-seed runner (dry-run by default)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory("uto_ros2"), "config", "validation_experiment.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("sample_count", default_value="10"),
        DeclareLaunchArgument("base_seed", default_value="1"),
        DeclareLaunchArgument("output_directory", default_value="~/uto_validation_results"),
        Node(
            package="uto_ros2", executable="uto_validation_metrics", name="uto_validation_metrics",
            parameters=[{"validation_output_directory": LaunchConfiguration("output_directory")}],
        ),
        ExecuteProcess(
            cmd=["ros2", "run", "uto_ros2", "uto_experiment_runner", "--config", config,
                 "--sample-count", LaunchConfiguration("sample_count"), "--base-seed",
                 LaunchConfiguration("base_seed"), "--dry-run"], output="screen",
        ),
    ])

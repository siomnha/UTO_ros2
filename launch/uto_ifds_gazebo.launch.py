from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    share=get_package_share_directory('uto_ros2'); mode=LaunchConfiguration('mode')
    return LaunchDescription([DeclareLaunchArgument('mode',default_value='online'),Node(package='uto_ros2',executable='uto_planner',parameters=[os.path.join(share,'config','uto_online.yaml')]),Node(package='uto_ros2',executable='px4_offboard_bridge',parameters=[os.path.join(share,'config','gazebo_harmonic_px4.yaml')])])

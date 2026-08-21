import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
def _nodes(context):
    mode=LaunchConfiguration('mode').perform(context)
    if mode not in ('global','online'):raise RuntimeError("mode must be 'global' or 'online'")
    share=get_package_share_directory('uto_ros2')
    return [Node(package='uto_ros2',executable='uto_planner',name='uto_planner',output='screen',parameters=[os.path.join(share,'config',f'uto_{mode}.yaml')]),Node(package='uto_ros2',executable='px4_offboard_bridge',name='px4_offboard_bridge',output='screen',parameters=[os.path.join(share,'config','gazebo_harmonic_px4.yaml')])]
def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('mode',default_value='online',choices=['global','online'],description='Select matching planner YAML'),OpaqueFunction(function=_nodes)])

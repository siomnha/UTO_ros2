from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('ifds_ros2')
    params = LaunchConfiguration('params')
    obstacles = LaunchConfiguration('obstacles')
    gnss_denied = LaunchConfiguration('gnss_denied')
    dynamic_obstacles = LaunchConfiguration('dynamic_obstacles')
    optimizer_mode = LaunchConfiguration('optimizer_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'ifds_params.yaml']),
            description='IFDS planner ROS parameter file.',
        ),
        DeclareLaunchArgument(
            'obstacles',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'known_obstacles.yaml']),
            description='Known obstacle YAML generated from Gazebo/world parameters.',
        ),
        DeclareLaunchArgument(
            'gnss_denied',
            default_value='false',
            description='false: use GNSS odometry; true: use FAST-LIO2 odometry.',
        ),
        DeclareLaunchArgument(
            'dynamic_obstacles',
            default_value='false',
            description='false: static obstacle centers; true: evaluate dynamic obstacle motion.',
        ),
        DeclareLaunchArgument(
            'optimizer_mode',
            default_value='0',
            description='0: fixed rho0/sigma0; 2: local optimiser mode.',
        ),
        Node(
            package='ifds_ros2',
            executable='ifds_planner',
            name='ifds_planner',
            output='screen',
            parameters=[
                params,
                {
                    'obstacles_yaml': obstacles,
                    'gnss_denied': ParameterValue(gnss_denied, value_type=bool),
                    'dynamic_obstacles': ParameterValue(dynamic_obstacles, value_type=bool),
                    'optimizer_mode': ParameterValue(optimizer_mode, value_type=int),
                },
            ],
        ),
    ])

from glob import glob
from setuptools import find_packages, setup

package_name = 'ifds_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/docs', glob('docs/*.md')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy'],
    zip_safe=True,
    maintainer='IFDS Maintainers',
    maintainer_email='maintainer@example.com',
    description='ROS 2 IFDS local path planner for PX4 x500 Gazebo Harmonic simulations.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ifds_planner = ifds_ros2.ifds_planner_node:main',
        ],
    },
)

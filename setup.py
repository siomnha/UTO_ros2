from setuptools import find_packages, setup
from glob import glob

setup(
    name="uto_ros2",
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/uto_ros2"]),
        ("share/uto_ros2", ["package.xml"]),
        ("share/uto_ros2/launch", glob("launch/*.launch.py")),
        ("share/uto_ros2/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "uto_planner=uto_ros2.uto_planner_node:main",
            "px4_offboard_bridge=uto_ros2.px4_offboard_bridge_node:main",
        ]
    },
)

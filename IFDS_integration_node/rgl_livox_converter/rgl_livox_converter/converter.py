#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from livox_ros_driver2.msg import CustomMsg
from livox_ros_driver2.msg import CustomPoint


class RglLivoxConverter(Node):
    """Convert RGL PointCloud2 messages into Livox CustomMsg messages."""

    def __init__(self) -> None:
        super().__init__('rgl_livox_converter')

        self.declare_parameter(
            'input_topic',
            '/livox_mid360/points'
        )
        self.declare_parameter(
            'output_topic',
            '/livox/lidar'
        )
        self.declare_parameter(
            'scan_period',
            0.1
        )

        input_topic = (
            self.get_parameter('input_topic')
            .get_parameter_value()
            .string_value
        )
        output_topic = (
            self.get_parameter('output_topic')
            .get_parameter_value()
            .string_value
        )

        self.scan_period = (
            self.get_parameter('scan_period')
            .get_parameter_value()
            .double_value
        )

        livox_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        if self.scan_period <= 0.0:
            raise ValueError('scan_period must be greater than zero')

        self.publisher = self.create_publisher(
            CustomMsg,
            output_topic,
            livox_qos,
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Converting {input_topic} -> {output_topic}, '
            f'scan_period={self.scan_period:.6f} seconds'
        )

    @staticmethod
    def stamp_to_nanoseconds(msg: PointCloud2) -> int:
        return (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

    @staticmethod
    def intensity_to_reflectivity(intensity: float) -> int:
        if not math.isfinite(intensity):
            return 0

        # RGL may output intensity in either 0..1 or 0..255.
        if 0.0 <= intensity <= 1.0:
            intensity *= 255.0

        return max(0, min(255, int(round(intensity))))

    def pointcloud_callback(self, cloud_msg: PointCloud2) -> None:
        try:
            points = list(
                point_cloud2.read_points(
                    cloud_msg,
                    field_names=('x', 'y', 'z', 'intensity'),
                    skip_nans=True
                )
            )
        except Exception as exc:
            self.get_logger().error(
                f'Failed to read PointCloud2: {exc}'
            )
            return

        point_count = len(points)

        if point_count == 0:
            self.get_logger().warning('Received an empty point cloud')
            return

        output = CustomMsg()

        output.header = cloud_msg.header
        output.timebase = self.stamp_to_nanoseconds(cloud_msg)
        output.point_num = point_count
        output.lidar_id = 0
        output.rsvd = [0, 0, 0]

        scan_period_ns = int(self.scan_period * 1_000_000_000)

        converted_points = []

        for index, raw_point in enumerate(points):
            x = float(raw_point[0])
            y = float(raw_point[1])
            z = float(raw_point[2])
            intensity = float(raw_point[3])

            point = CustomPoint()

            point.x = x
            point.y = y
            point.z = z

            # Approximate firing time by point order.
            if point_count > 1:
                point.offset_time = int(
                    index * scan_period_ns / point_count
                )
            else:
                point.offset_time = 0

            point.reflectivity = self.intensity_to_reflectivity(
                intensity
            )

            # RGL output does not currently provide these fields.
            point.tag = 0x10
            point.line = 0

            converted_points.append(point)

        output.points = converted_points
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = RglLivoxConverter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

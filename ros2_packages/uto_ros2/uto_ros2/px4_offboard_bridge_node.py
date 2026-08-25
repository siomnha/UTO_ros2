"""PX4 heartbeat/command sequencing/hold/trajectory execution only."""

import json
import math
import numpy as np
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from std_msgs.msg import String
from .dynamics import dynamics
from .math_utils import enu_to_ned, yaw_enu_to_ned
from .planner_runtime import (
    CommandState,
    Px4CommandSequencer,
    offboard_control_flags,
    px4_status_payload,
)
from .trajectory import ExecutionSetpoint, Trajectory, TrajectoryExecution


class PX4OffboardBridge(Node):
    """Publish one selected PX4 setpoint per non-failsafe timer cycle."""

    def __init__(self) -> None:
        super().__init__("px4_offboard_bridge")
        self._declare_parameters()
        self.status = None
        self.status_time = 0.0
        self.local_position = None
        self.prestream_count = 0
        self.last_publish_time = None
        self.max_jitter = 0.0
        self.setpoint_publish_count = 0
        self.execution = TrajectoryExecution([0.0, 0.0, self._parameter("hold_altitude")])
        self.sequencer = Px4CommandSequencer(
            self._parameter("auto_arm_takeoff"),
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            self._parameter("command_retry_interval"),
            self._parameter("command_ack_timeout"),
            self._parameter("command_max_retries"),
        )
        self._create_interfaces()

    def _declare_parameters(self) -> None:
        defaults = {
            "trajectory_topic": "/uto/trajectory",
            "execution_command_topic": "/uto/execution_command",
            "px4_state_topic": "/uto/px4_status",
            "offboard_control_mode_topic": "/fmu/in/offboard_control_mode",
            "trajectory_setpoint_topic": "/fmu/in/trajectory_setpoint",
            "vehicle_command_topic": "/fmu/in/vehicle_command",
            "vehicle_status_topic": "/fmu/out/vehicle_status_v1",
            "vehicle_local_position_topic": "/fmu/out/vehicle_local_position_v1",
            "vehicle_command_ack_topic": "/fmu/out/vehicle_command_ack",
            "setpoint_rate": 40.0,
            "offboard_control_level": "position",
            "hold_altitude": 1.5,
            "auto_arm_takeoff": False,
            "prestream_setpoints": 20,
            "hold_position_tolerance": 0.2,
            "hold_velocity_tolerance": 0.25,
            "px4_status_timeout": 0.5,
            "command_retry_interval": 0.5,
            "command_ack_timeout": 1.0,
            "command_max_retries": 5,
            "target_system": 1,
            "target_component": 1,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _create_interfaces(self) -> None:
        qos = rclpy.qos.qos_profile_sensor_data
        self.mode_pub = self.create_publisher(
            OffboardControlMode, self._parameter("offboard_control_mode_topic"), qos
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, self._parameter("trajectory_setpoint_topic"), qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, self._parameter("vehicle_command_topic"), qos
        )
        self.state_pub = self.create_publisher(String, self._parameter("px4_state_topic"), 10)
        self.create_subscription(
            VehicleStatus, self._parameter("vehicle_status_topic"), self._on_status, qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            self._parameter("vehicle_local_position_topic"),
            self._on_local_position,
            qos,
        )
        self.create_subscription(
            VehicleCommandAck, self._parameter("vehicle_command_ack_topic"), self._on_ack, qos
        )
        self.create_subscription(
            String, self._parameter("trajectory_topic"), self._on_trajectory, 10
        )
        self.create_subscription(
            String,
            self._parameter("execution_command_topic"),
            self._on_execution_command,
            10,
        )
        self.timer = self.create_timer(1.0 / self._parameter("setpoint_rate"), self._on_timer)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, message: VehicleStatus) -> None:
        self.status = message
        self.status_time = self._now()

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        self.local_position = message

    def _on_ack(self, message: VehicleCommandAck) -> None:
        self.sequencer.acknowledge(int(message.command), int(message.result))

    def _on_trajectory(self, message: String) -> None:
        try:
            self.execution.accept_executable(Trajectory.from_json(message.data))
        except (ValueError, TypeError, KeyError) as exception:
            self.get_logger().error(f"Rejected executable trajectory: {exception}")

    def _on_execution_command(self, message: String) -> None:
        try:
            command = json.loads(message.data).get("command")
        except ValueError:
            return
        if command == "SAFE_HOLD":
            self.execution.request_emergency_hold()
        elif command == "HOLD_CURRENT":
            self.execution.request_hold_current()

    def _connected(self, now: float) -> bool:
        return self.status is not None and now - self.status_time <= self._parameter(
            "px4_status_timeout"
        )

    def _armed(self) -> bool:
        armed = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
        return self.status is not None and int(self.status.arming_state) == armed

    def _offboard(self) -> bool:
        offboard = getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14)
        return self.status is not None and int(self.status.nav_state) == offboard

    def _hold_ready(self) -> bool:
        if self.local_position is None:
            return False
        position = np.array([self.local_position.y, self.local_position.x, -self.local_position.z])
        velocity = np.array(
            [self.local_position.vy, self.local_position.vx, -self.local_position.vz]
        )
        target = self.execution.takeoff_hold.state[:3]
        if self.sequencer.state == CommandState.READY:
            target = self.execution.terminal_hold.state[:3]
        return np.linalg.norm(position - target) <= self._parameter(
            "hold_position_tolerance"
        ) and np.linalg.norm(velocity) <= self._parameter("hold_velocity_tolerance")

    def _on_timer(self) -> None:
        now = self._now()
        self._record_jitter(now)
        connected = self._connected(now)
        failsafe = bool(getattr(self.status, "failsafe", False)) if connected else False
        if not failsafe:
            self._publish_heartbeat()
        if connected and self.sequencer.state in (
            CommandState.WAIT_CONNECTION,
            CommandState.PRESTREAM,
        ):
            self.prestream_count += 1
        command = self.sequencer.tick(
            now,
            connected and not failsafe,
            self.prestream_count >= self._parameter("prestream_setpoints"),
            self._offboard(),
            self._armed(),
            self._hold_ready(),
        )
        if command is not None:
            self._publish_command(command)
        # Do not override PX4's own setpoints once its failsafe is asserted.
        if not failsafe:
            output = self.execution.select(
                now,
                trajectory_allowed=self.sequencer.state == CommandState.READY,
                takeoff_complete=self.sequencer.state == CommandState.READY,
            )
            self._publish_exactly_one_setpoint(output)
        self._publish_status(now, connected, failsafe)

    def _record_jitter(self, now: float) -> None:
        expected = 1.0 / self._parameter("setpoint_rate")
        if self.last_publish_time is not None:
            self.max_jitter = max(self.max_jitter, abs(now - self.last_publish_time - expected))
        self.last_publish_time = now

    def _publish_heartbeat(self) -> None:
        message = OffboardControlMode()
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        flags = offboard_control_flags(self._parameter("offboard_control_level"))
        for field, enabled in flags.items():
            setattr(message, field, enabled)
        self.mode_pub.publish(message)

    def _publish_command(self, command: int) -> None:
        message = VehicleCommand()
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        message.command = int(command)
        if command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            message.param1 = 1.0
            message.param2 = 6.0
        elif command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            message.param1 = 1.0
        message.target_system = self._parameter("target_system")
        message.target_component = self._parameter("target_component")
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_pub.publish(message)

    def _publish_exactly_one_setpoint(self, output: ExecutionSetpoint) -> None:
        state = output.state
        acceleration = dynamics(state, output.control)[3:6]
        message = TrajectorySetpoint()
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        message.position = [float(value) for value in enu_to_ned(state[:3])]
        message.velocity = [float(value) for value in enu_to_ned(state[3:6])]
        message.acceleration = [float(value) for value in enu_to_ned(acceleration)]
        message.jerk = [math.nan] * 3
        message.yaw = float(yaw_enu_to_ned(state[8]))
        message.yawspeed = float(-output.control[3])
        self.setpoint_pub.publish(message)
        self.setpoint_publish_count += 1

    def _publish_status(self, now: float, connected: bool, failsafe: bool) -> None:
        payload = px4_status_payload(
            connected=connected,
            hold_ready=self._hold_ready(),
            failsafe=failsafe,
            armed=self._armed(),
            offboard=self._offboard(),
            mode=self.sequencer.state.name,
            command_retries=self.sequencer.retries,
            command_fault=self.sequencer.fault_reason,
            setpoint_max_jitter=self.max_jitter,
            setpoint_publish_count=self.setpoint_publish_count,
            active_trajectory_remaining=(
                self.execution.trajectory.remaining(now)
                if self.execution.trajectory is not None
                else 0.0
            ),
        )
        self.state_pub.publish(String(data=json.dumps(payload)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PX4OffboardBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

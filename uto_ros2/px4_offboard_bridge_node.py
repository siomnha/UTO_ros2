import json, math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode,TrajectorySetpoint,VehicleCommand,VehicleStatus
from .math_utils import enu_to_ned,yaw_enu_to_ned
class PX4OffboardBridge(Node):
    def __init__(self):
        super().__init__('px4_offboard_bridge'); self.connected=False; self.trajectory=None; self.hold=[0.,0.,1.5]
        for n,v in [('trajectory_topic','/uto/trajectory'),('setpoint_rate',40.),('hold_altitude',1.5),('auto_arm_takeoff',False)]:self.declare_parameter(n,v)
        self.hold[2]=self.get_parameter('hold_altitude').value; q=rclpy.qos.QoSProfile(depth=1,reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        self.mode=self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',q); self.sp=self.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',q); self.cmd=self.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',q)
        self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',self._status,q); self.create_subscription(String,self.get_parameter('trajectory_topic').value,self._trajectory,10); self.create_timer(1/self.get_parameter('setpoint_rate').value,self._publish)
    def _status(self,m): self.connected=True
    def _trajectory(self,m):
        try:self.trajectory=json.loads(m.data)
        except ValueError:self.trajectory=None
    def _publish(self):
        now=int(self.get_clock().now().nanoseconds/1000); mode=OffboardControlMode(); mode.timestamp=now; mode.position=True; mode.velocity=False; mode.acceleration=False; mode.attitude=False; mode.body_rate=False; self.mode.publish(mode)
        p=enu_to_ned(self.hold); yaw=yaw_enu_to_ned(0.); s=TrajectorySetpoint(); s.timestamp=now; s.position=[float(x) for x in p]; s.velocity=[math.nan]*3; s.acceleration=[math.nan]*3; s.jerk=[math.nan]*3; s.yaw=float(yaw); s.yawspeed=math.nan; self.sp.publish(s)
def main(args=None):
    rclpy.init(args=args); n=PX4OffboardBridge(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()

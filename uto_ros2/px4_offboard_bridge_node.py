"""Independent high-rate PX4 offboard publisher and physical trajectory executor."""
import json, math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode,TrajectorySetpoint,VehicleCommand,VehicleStatus,VehicleLocalPosition
from .dynamics import dynamics
from .math_utils import enu_to_ned,yaw_enu_to_ned
from .trajectory_buffer import Trajectory,TrajectoryBuffer
class PX4OffboardBridge(Node):
    def __init__(self):
        super().__init__('px4_offboard_bridge'); defaults={'trajectory_topic':'/uto/trajectory','px4_state_topic':'/uto/px4_status','setpoint_rate':40.,'hold_altitude':1.5,'auto_arm_takeoff':False,'prestream_setpoints':20,'hold_position_tolerance':.2,'hold_velocity_tolerance':.25,'px4_status_timeout':.5,'target_system':1,'target_component':1}
        for n,v in defaults.items():self.declare_parameter(n,v)
        self.buffer=TrajectoryBuffer(); self.status=None; self.status_time=0.; self.local=None; self.mode='WAIT_PX4'; self.prestream=0; self.commands_sent=False; self.last_publish=None; self.max_jitter=0.; self.hold_enu=np.array([0.,0.,self.get_parameter('hold_altitude').value])
        qos=rclpy.qos.qos_profile_sensor_data
        self.mode_pub=self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',qos); self.sp_pub=self.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',qos); self.cmd_pub=self.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',qos); self.state_pub=self.create_publisher(String,self.get_parameter('px4_state_topic').value,10)
        self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',self._status,qos); self.create_subscription(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1',self._local,qos); self.create_subscription(String,self.get_parameter('trajectory_topic').value,self._trajectory,10); self.create_timer(1/self.get_parameter('setpoint_rate').value,self._publish)
    def now(self):return self.get_clock().now().nanoseconds*1e-9
    def _status(self,m):self.status=m; self.status_time=self.now()
    def _local(self,m):self.local=m
    def _trajectory(self,m):
        try:self.buffer.offer(Trajectory.from_json(m.data))
        except (ValueError,TypeError,KeyError) as exc:self.get_logger().error('Rejected trajectory: %s'%exc)
    def _command(self,command,param1=0.,param2=0.):
        m=VehicleCommand(); m.timestamp=int(self.get_clock().now().nanoseconds/1000); m.param1=float(param1); m.param2=float(param2); m.command=int(command); m.target_system=self.get_parameter('target_system').value; m.target_component=self.get_parameter('target_component').value; m.source_system=1; m.source_component=1; m.from_external=True; self.cmd_pub.publish(m)
    def _hold_ready(self):
        if self.local is None:return False
        position=ned_to_enu_position(self.local.x,self.local.y,self.local.z); velocity=ned_to_enu_position(self.local.vx,self.local.vy,self.local.vz)
        return np.linalg.norm(position-self.hold_enu)<=self.get_parameter('hold_position_tolerance').value and np.linalg.norm(velocity)<=self.get_parameter('hold_velocity_tolerance').value
    def _publish(self):
        now=self.now(); expected=1/self.get_parameter('setpoint_rate').value
        if self.last_publish is not None:self.max_jitter=max(self.max_jitter,abs(now-self.last_publish-expected))
        self.last_publish=now; connected=self.status is not None and now-self.status_time<=self.get_parameter('px4_status_timeout').value; failsafe=bool(getattr(self.status,'failsafe',False)) if connected else False
        self._heartbeat(); sampled=self.buffer.sample(now)
        if not connected or failsafe:self.mode='SAFE_HOLD' if self.local else 'WAIT_PX4'; sampled=None
        elif self.prestream<self.get_parameter('prestream_setpoints').value:self.mode='TAKEOFF'; self.prestream+=1; sampled=None
        elif self.get_parameter('auto_arm_takeoff').value and not self.commands_sent:
            self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1.,6.); self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1.); self.commands_sent=True; self.mode='TAKEOFF'; sampled=None
        elif sampled is not None and self._offboard_ready():self.mode='TRAJECTORY'; self._trajectory_setpoint(*sampled)
        elif sampled is not None:self.mode='HOLD'; sampled=None
        else:self.mode='HOLD'; self._position_setpoint(self.hold_enu,np.zeros(3),np.zeros(3),0.)
        if sampled is None:self._position_setpoint(self.hold_enu,np.zeros(3),np.zeros(3),0.)
        payload={'connected':connected,'hold_ready':self._hold_ready(),'failsafe':failsafe,'armed':int(getattr(self.status,'arming_state',0)) if connected else 0,'nav_state':int(getattr(self.status,'nav_state',0)) if connected else 0,'mode':self.mode,'setpoint_max_jitter':self.max_jitter,'active_trajectory_remaining':self.buffer.remaining(now)}; self.state_pub.publish(String(data=json.dumps(payload)))
    def _offboard_ready(self):
        if self.status is None:return False
        armed=getattr(VehicleStatus,'ARMING_STATE_ARMED',2); offboard=getattr(VehicleStatus,'NAVIGATION_STATE_OFFBOARD',14)
        return int(getattr(self.status,'arming_state',-1))==armed and int(getattr(self.status,'nav_state',-1))==offboard and not bool(getattr(self.status,'failsafe',False))
    def _heartbeat(self):
        m=OffboardControlMode(); m.timestamp=int(self.get_clock().now().nanoseconds/1000); m.position=True; m.velocity=True; m.acceleration=True; m.attitude=False; m.body_rate=False; self.mode_pub.publish(m)
    def _trajectory_setpoint(self,state,control):
        acceleration=dynamics(state,control)[3:6]; self._position_setpoint(state[:3],state[3:6],acceleration,state[8])
    def _position_setpoint(self,position,velocity,acceleration,yaw):
        s=TrajectorySetpoint(); s.timestamp=int(self.get_clock().now().nanoseconds/1000); s.position=[float(x) for x in enu_to_ned(position)]; s.velocity=[float(x) for x in enu_to_ned(velocity)]; s.acceleration=[float(x) for x in enu_to_ned(acceleration)]; s.jerk=[math.nan]*3; s.yaw=float(yaw_enu_to_ned(yaw)); s.yawspeed=math.nan; self.sp_pub.publish(s)
def ned_to_enu_position(x,y,z):return np.array([y,x,-z],float)
def main(args=None):
    rclpy.init(args=args); node=PX4OffboardBridge()
    try:rclpy.spin(node)
    finally:node.destroy_node(); rclpy.shutdown()

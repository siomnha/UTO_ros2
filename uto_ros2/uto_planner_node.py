import json, time, numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry,Path
from std_msgs.msg import String
from .state_machine import MissionStateMachine,State
from .ifds_path_adapter import Polyline,path_generation
from .async_worker import LatestWinsWorker
from .uto_nlp import UTONLP,UTOConfig

class UTOPlannerNode(Node):
    """Owns latest snapshots and one IPOPT worker; callbacks never solve inline."""
    def __init__(self):
        super().__init__('uto_planner'); self.sm=MissionStateMachine(); self.belief=None; self.path=None; self.generation=0; self.nlp=UTONLP(UTOConfig())
        for n,v in [('belief_topic','/Odometry'),('path_topic','/ifds/local_path'),('trajectory_topic','/uto/trajectory'),('diagnostics_topic','/uto/diagnostics'),('replan_rate',1.5),('horizon',3.0),('lookahead_count',10),('lookahead_spacing',.4),('belief_timeout',.3),('path_timeout',.8)]: self.declare_parameter(n,v)
        self.pub=self.create_publisher(String,self.get_parameter('trajectory_topic').value,10); self.diag=self.create_publisher(String,self.get_parameter('diagnostics_topic').value,10)
        self.create_subscription(Odometry,self.get_parameter('belief_topic').value,self._belief,10); self.create_subscription(Path,self.get_parameter('path_topic').value,self._path,10)
        self.worker=LatestWinsWorker(self._solve,self._done); self.create_timer(1/self.get_parameter('replan_rate').value,self._tick)
    def _belief(self,m):
        p=m.pose.pose.position; v=m.twist.twist.linear; self.belief=(time.monotonic(),np.array([p.x,p.y,p.z,v.x,v.y,v.z,0,0,0]))
    def _path(self,m):
        pts=[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in m.poses]
        try:self.path=(time.monotonic(),Polyline(pts),path_generation(m.header.stamp.sec+1e-9*m.header.stamp.nanosec,pts))
        except ValueError as e:self.get_logger().warning(str(e))
    def _tick(self):
        now=time.monotonic(); stale=not self.belief or not self.path or now-self.belief[0]>self.get_parameter('belief_timeout').value or now-self.path[0]>self.get_parameter('path_timeout').value
        if stale: self.sm.state=State.SAFE_HOLD; self.diag.publish(String(data=json.dumps({'state':self.sm.state.name,'reason':'stale belief/path'}))); return
        self.generation+=1; self.worker.submit({'generation':self.generation,'x':self.belief[1].copy(),'path':self.path[1],'path_generation':self.path[2]})
    def _solve(self,r):
        if self.nlp.opti is None:self.nlp.build()
        refs=r['path'].lookahead(r['x'][:3],self.nlp.cfg.references,self.get_parameter('lookahead_spacing').value); x0=np.repeat(r['x'][:,None],7,axis=1)
        self.nlp.set_parameters(x0,refs,self.get_parameter('horizon').value,[-4]*3,[4]*3); return self.nlp.solve()
    def _done(self,r,result,stale):
        if stale or isinstance(result,Exception): self.sm.state=State.SAFE_HOLD; return
        self.sm.state=State.EXECUTING; msg={'generation':r['generation'],'path_generation':r['path_generation'],'build_time':self.nlp.build_time,'solve_time':self.nlp.solve_time,'states':np.mean(result['sigma_states'],axis=2).tolist(),'controls':result['controls'].tolist()}; self.pub.publish(String(data=json.dumps(msg))); self.diag.publish(String(data=json.dumps({k:msg[k] for k in ('generation','build_time','solve_time')})))
def main(args=None):
    rclpy.init(args=args); n=UTOPlannerNode(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()

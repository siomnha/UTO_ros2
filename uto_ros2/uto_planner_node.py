"""ROS node wiring FAST-LIO belief, IFDS, delay-compensated LGR UTO and buffering."""
import json, threading, time
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from px4_msgs.msg import VehicleOdometry
from .async_worker import LatestWinsWorker
from .belief_adapter import BeliefAdapter,StabilityConfig
from .delay_compensator import DelayCompensator
from .feasibility_gate import FeasibilityGate,GateConfig
from .ifds_path_adapter import Polyline,path_generation
from .math_utils import ned_to_enu
from .planner_core import PlanningRequest,CandidateManager
from .state_machine import MissionStateMachine,State
from .trajectory_buffer import TrajectoryBuffer
from .uto_nlp import UTONLP,UTOConfig
class UTOPlannerNode(Node):
    def __init__(self):
        super().__init__('uto_planner'); self.lock=threading.RLock(); self.sm=MissionStateMachine(); self.belief=None; self.belief_stable=False; self.path=None; self.velocity=None; self.velocity_stamp=0.; self.px4={'connected':False,'hold_ready':False,'failsafe':False}; self.request_generation=0; self.last_path_generation=''
        defaults={'belief_topic':'/Odometry','path_topic':'/ifds/local_path','velocity_topic':'/uto/velocity','px4_velocity_topic':'/fmu/out/vehicle_odometry','px4_state_topic':'/uto/px4_status','trajectory_topic':'/uto/trajectory','diagnostics_topic':'/uto/diagnostics','planning_frame':'map','mode':'online','velocity_source':'patched_odometry_twist','horizon':3.,'lookahead_count':10,'lookahead_spacing':.4,'replan_rate':1.5,'belief_timeout':.3,'path_timeout':.8,'velocity_timeout':.3,'stable_samples':5,'covariance_eigen_floor':1e-9,'covariance_inflation':1.2,'position_covariance_trace_max':.2,'attitude_covariance_trace_max':.05,'mean_delta_max':.3,'covariance_delta_max':.1,'regions':2,'lgr_nodes_per_region':5,'sigma_count':7,'path_tube_radius':.8,'sigma_path_tube_radius':1.,'initial_delay':.5,'cold_start_delay':1.2,'delay_p90_window':20,'minimum_delay':.2,'maximum_delay':1.5,'validation_time':.02,'commit_margin':.08,'commit_guard':.05,'terminal_position_tolerance':.3,'terminal_velocity_tolerance':.05,'process_noise_diagonal':[.01,.01,.01,.02,.02,.02,.001,.001,.001],'state_scale':[3.,1.,1.2,4.,4.,4.,.6,.6,.6],'control_scale':[9.81,.48,.48,1.2],'control_min':[0.,-.48,-.48,-1.2],'control_max':[18.,.48,.48,1.2],'weights':[1.,10.,10.,.007,1e-6,1e-6],'velocity_max':4.,'angle_max':.6,'solver_tolerance':2e-5,'solver_max_iterations':900}
        for name,value in defaults.items():self.declare_parameter(name,value)
        p=lambda n:self.get_parameter(n).value
        cfg=UTOConfig(p('regions'),p('lgr_nodes_per_region'),p('sigma_count'),p('lookahead_count'),state_scale=tuple(p('state_scale')),control_scale=tuple(p('control_scale')),control_min=tuple(p('control_min')),control_max=tuple(p('control_max')),velocity_max=p('velocity_max'),angle_max=p('angle_max'),terminal_position_tolerance=p('terminal_position_tolerance'),max_iter=p('solver_max_iterations'),tolerance=p('solver_tolerance'))
        self.nlp=UTONLP(cfg); stable=StabilityConfig(p('stable_samples'),p('belief_timeout'),p('position_covariance_trace_max'),p('attitude_covariance_trace_max'),p('mean_delta_max'),p('covariance_delta_max')); self.adapter=BeliefAdapter(p('covariance_eigen_floor'),p('covariance_inflation'),stable)
        self.delay=DelayCompensator(p('initial_delay'),p('delay_p90_window'),p('minimum_delay'),p('maximum_delay'),p('validation_time'),p('commit_margin')); self.buffer=TrajectoryBuffer(); self.manager=CandidateManager(self.buffer,p('commit_guard')); self.gate=FeasibilityGate(GateConfig(p('velocity_max'),p('angle_max'),tuple(p('control_min')),tuple(p('control_max')),p('terminal_position_tolerance'),p('path_tube_radius'),p('sigma_path_tube_radius')))
        self.trajectory_pub=self.create_publisher(String,p('trajectory_topic'),10); self.diag_pub=self.create_publisher(String,p('diagnostics_topic'),10)
        self.create_subscription(Odometry,p('belief_topic'),self._odometry,10); self.create_subscription(Path,p('path_topic'),self._path,10); self.create_subscription(String,p('px4_state_topic'),self._px4,10)
        if p('velocity_source')=='separate_velocity_topic':self.create_subscription(TwistStamped,p('velocity_topic'),self._separate_velocity,10)
        if p('velocity_source')=='px4_vehicle_odometry':self.create_subscription(VehicleOdometry,p('px4_velocity_topic'),self._px4_velocity,rclpy.qos.qos_profile_sensor_data)
        self.worker=LatestWinsWorker(self._solve,self._done); self.timer=self.create_timer(1/max(p('replan_rate'),.1),self._tick)
    def now(self):return self.get_clock().now().nanoseconds*1e-9
    def _separate_velocity(self,m):self.velocity=np.array([m.twist.linear.x,m.twist.linear.y,m.twist.linear.z]); self.velocity_stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
    def _px4_velocity(self,m):self.velocity=ned_to_enu(m.velocity); self.velocity_stamp=float(m.timestamp)*1e-6
    def _odometry(self,m):
        now=self.now(); stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9; p=self.get_parameter
        source=p('velocity_source').value
        if source=='patched_odometry_twist':velocity=np.array([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]); velocity_ok=np.all(np.isfinite(velocity))
        else:velocity=self.velocity; velocity_ok=velocity is not None and now-self.velocity_stamp<=p('velocity_timeout').value
        q=m.pose.pose.orientation; pos=m.pose.pose.position
        try:
            belief,stable=self.adapter.convert(stamp,m.header.frame_id,[pos.x,pos.y,pos.z],[q.x,q.y,q.z,q.w],m.pose.covariance,velocity,now,m.header.frame_id==p('planning_frame').value,velocity_ok)
            with self.lock:self.belief,self.belief_stable=belief,stable
        except ValueError as exc:self._diagnostic(reason=str(exc),level='ERROR')
    def _path(self,m):
        now=self.now(); frame=self.get_parameter('planning_frame').value
        if m.header.frame_id!=frame:self._diagnostic(reason='path frame mismatch; TF transform required',level='ERROR'); return
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9; points=[[x.pose.position.x,x.pose.position.y,x.pose.position.z] for x in m.poses]
        try:
            item={'stamp':stamp,'received':now,'polyline':Polyline(points),'generation':path_generation(stamp,points)}
            with self.lock:self.path=item
        except ValueError as exc:self._diagnostic(reason=str(exc),level='ERROR')
    def _px4(self,m):
        try:
            status=json.loads(m.data)
            with self.lock:self.px4.update(status)
        except ValueError:self._diagnostic(reason='invalid PX4 status schema',level='ERROR')
    def _control_at(self,absolute_time):
        active=self.buffer.active
        if active and active.commit_time<=absolute_time<=active.end_time:return active.sample(absolute_time)[1]
        return np.array([9.81,0,0,0])
    def _tick(self):
        now=self.now(); p=lambda n:self.get_parameter(n).value
        with self.lock:belief=self.belief; path=self.path; stable=self.belief_stable; px4=dict(self.px4)
        if px4.get('failsafe'):self.sm.update(safe_hold=True); self._diagnostic(reason='PX4 failsafe',level='ERROR'); return
        if not px4.get('connected'):self.sm.state=State.WAIT_PX4; return
        if self.sm.state==State.WAIT_PX4:self.sm.update(px4_connected=True)
        if self.sm.state==State.TAKEOFF:self.sm.update(takeoff_started=True)
        if self.sm.state==State.HOLD and px4.get('hold_ready'):self.sm.update(hold_ready=True)
        if belief is None or now-belief.stamp>p('belief_timeout'):
            if self.sm.state in (State.EXECUTING,State.REPLANNING):self.sm.update(safe_hold=True)
            return
        if self.sm.state==State.WAIT_BELIEF_STABLE and stable:self.sm.update(belief_stable=True)
        if path is None or now-path['received']>p('path_timeout') or now-path['stamp']>p('path_timeout'):
            if self.sm.state in (State.EXECUTING,State.REPLANNING):self.sm.update(safe_hold=True)
            return
        if self.sm.state==State.WAIT_IFDS_INITIAL_PATH:self.sm.update(path_ready=True)
        if self.sm.state==State.BUILDING_NLP:self._submit(belief,path,now,first=True); return
        committed=self.buffer.commit(now)
        if self.sm.state==State.TRAJECTORY_READY and self.buffer.active:self.sm.update(committed=True)
        elif self.sm.state==State.REPLANNING and committed:self.sm.update(replan_complete=True)
        if self.sm.state in (State.EXECUTING,State.REPLANNING):
            if self.buffer.remaining(now)<=0:self.sm.update(safe_hold=True); return
            if p('mode')=='global' and path['generation']==self.last_path_generation and self.buffer.remaining(now)>p('horizon')*.4:return
            if self.sm.state==State.EXECUTING:self.sm.update(replan_started=True)
            self._submit(belief,path,now,first=False)
        self._diagnostic()
    def _submit(self,belief,path,now,first):
        delay=max(self.delay.estimate(),self.get_parameter('cold_start_delay').value) if first else self.delay.estimate(); commit=now+delay; sigma,mean,_=self.delay.predict(belief.sigma_states,lambda relative:self._control_at(now+relative),commit-belief.stamp,self.get_parameter('process_noise_diagonal').value); refs=path['polyline'].lookahead(mean[:3],self.get_parameter('lookahead_count').value,self.get_parameter('lookahead_spacing').value)
        with self.lock:self.request_generation+=1; generation=self.request_generation
        request=PlanningRequest(generation,path['generation'],belief.generation,now,commit,sigma,mean,refs,path['polyline']); self.worker.submit(request)
    def _solve(self,request):
        if self.nlp.opti is None:self.nlp.build()
        tolerance=self.get_parameter('terminal_velocity_tolerance').value; mode=1 if np.linalg.norm(request.references[-1]-request.references[0])<self.get_parameter('lookahead_spacing').value else 0; vref=np.zeros(3) if mode else (request.references[-1]-request.references[-2])/max(self.get_parameter('lookahead_spacing').value,.1)
        bound=tolerance if mode else self.get_parameter('velocity_max').value
        self.nlp.set_parameters(request.sigma_states,request.references,self.get_parameter('horizon').value,vref,[-bound]*3,[bound]*3,mode,self.get_parameter('weights').value,self.get_parameter('terminal_position_tolerance').value); result=self.nlp.solve(); result['terminal_velocity_lower']=[-bound]*3; result['terminal_velocity_upper']=[bound]*3; return result
    def _done(self,request,result,stale):
        now=self.now()
        with self.lock: current_path_generation=self.path['generation'] if self.path else ''
        if stale or request.path_generation!=current_path_generation:self.manager.stale_discards+=1; return
        if isinstance(result,Exception):self.sm.update(safe_hold=self.buffer.remaining(now)<=0); self._diagnostic(reason=str(result),level='ERROR'); return
        self.delay.record(self.nlp.solve_time); gate=self.gate.check(result,request.path,request.predicted_mean,request.references[-1],result['terminal_velocity_lower'],result['terminal_velocity_upper'],self.request_generation,request.request_generation)
        accepted,reason=self.manager.admit(request,result,now,self.request_generation,gate)
        if accepted:
            self.last_path_generation=request.path_generation
            self.trajectory_pub.publish(String(data=self.buffer.candidate.to_json()))
            if self.sm.state==State.BUILDING_NLP:self.sm.update(built=True)
            if self.sm.state==State.FIRST_SOLVE:self.sm.update(candidate_ready=True)
        elif reason not in ('stale','late'):self.sm.update(safe_hold=self.buffer.remaining(now)<=0)
        self._diagnostic(reason=reason)
    def _diagnostic(self,reason='',level='OK'):
        now=self.now(); d={'state':self.sm.state.name,'level':level,'reason':reason,'cold_build_time':self.nlp.build_time,'parameter_update_time':self.nlp.parameter_update_time,'solve_time':self.nlp.solve_time,'solve_p90':self.delay.percentile90(),'extraction_time':self.nlp.extraction_time,'predicted_commit_delay':self.delay.estimate(),'deadline_miss_count':self.manager.deadline_misses,'stale_candidate_discard_count':self.manager.stale_discards,'nlp_build_count':self.nlp.build_count,'active_trajectory_remaining':self.buffer.remaining(now)}; self.diag_pub.publish(String(data=json.dumps(d)))
    def destroy_node(self):self.worker.shutdown(); return super().destroy_node()
def main(args=None):
    rclpy.init(args=args); node=UTOPlannerNode()
    try:rclpy.spin(node)
    finally:node.destroy_node(); rclpy.shutdown()

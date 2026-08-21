"""ROS-independent orchestration primitives used by the node and mock tests."""
from dataclasses import dataclass
import time, numpy as np
from .trajectory_buffer import Trajectory,TrajectoryBuffer
from .delay_compensator import DelayCompensator
@dataclass(frozen=True)
class PlanningRequest:
    request_generation:int; path_generation:str; belief_generation:int; request_time:float; commit_time:float; sigma_states:np.ndarray; predicted_mean:np.ndarray; references:np.ndarray; path:object
class CandidateManager:
    def __init__(self,buffer=None,guard=.05):self.buffer=buffer or TrajectoryBuffer(); self.guard=guard; self.deadline_misses=0; self.stale_discards=0
    def make_trajectory(self,request,result,frame_id='map'):
        return Trajectory(result['times'],result['states_physical'],result['controls_physical'],request.request_generation,int(request.commit_time*1e9),request.path_generation,frame_id,result['mean_covariances'])
    def admit(self,request,result,now,current_generation,gate_result):
        if request.request_generation!=current_generation:self.stale_discards+=1; return False,'stale'
        if now>request.commit_time-self.guard:self.deadline_misses+=1; return False,'late'
        if not gate_result.accepted:return False,','.join(gate_result.reasons)
        return self.buffer.offer(self.make_trajectory(request,result)),'accepted'

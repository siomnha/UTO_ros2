import numpy as np
from uto_ros2.belief_adapter import BeliefAdapter,StabilityConfig
from uto_ros2.feasibility_gate import FeasibilityGate,GateConfig
from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.planner_core import CandidateManager,PlanningRequest
from uto_ros2.state_machine import MissionStateMachine,State

def test_mock_hold_belief_path_solve_commit_replan_and_end_hold():
    sm=MissionStateMachine(); sm.update(px4_connected=True); sm.update(takeoff_started=True); sm.update(hold_ready=True)
    adapter=BeliefAdapter(stability=StabilityConfig(samples=2,position_trace=1,attitude_trace=1)); cov=np.eye(6)*.01
    adapter.convert(0,'map',[0,0,1],[0,0,0,1],cov.ravel(),[0,0,0],0); belief,stable=adapter.convert(.01,'map',[0,0,1],[0,0,0,1],cov.ravel(),[0,0,0],.01); sm.update(belief_stable=stable)
    path=Polyline([[0,0,1],[1,0,1]]); sm.update(path_ready=True); assert sm.state==State.BUILDING_NLP
    sm.update(built=True); request=PlanningRequest(1,'path1',belief.generation,.1,1.,belief.sigma_states,belief.mean_state,path.lookahead([0,0,1],10,.2),path)
    states=np.zeros((2,9)); states[:,2]=1; states[1,0]=1; sigma=np.repeat(states[:,None,:],7,axis=1); result={'times':np.array([0,1]),'states_physical':states,'controls_physical':np.array([[9.81,0,0,0]]*2),'sigma_states_physical':sigma,'mean_covariances':np.zeros((2,9,9)),'stats':{'success':True}}
    gate=FeasibilityGate(GateConfig(terminal_position_tolerance=.1)).check(result,path,belief.mean_state,[1,0,1],[-1]*3,[1]*3,1,1); manager=CandidateManager(guard=.05); accepted,_=manager.admit(request,result,.5,1,gate); assert accepted
    sm.update(candidate_ready=True); manager.buffer.commit(1.); sm.update(committed=True); assert sm.state==State.EXECUTING and manager.buffer.sample(1.5) is not None
    sm.update(replan_started=True); assert sm.state==State.REPLANNING; assert not manager.admit(request,result,.5,2,gate)[0]
    assert manager.buffer.sample(2.1) is None

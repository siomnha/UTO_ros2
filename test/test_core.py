import time
import numpy as np
import pytest
from uto_ros2.async_worker import LatestWinsWorker
from uto_ros2.belief_adapter import (BeliefAdapter,StabilityConfig,attitude_sigma_tangent_mean,pose_covariance_to_tangent,sanitize_covariance,sigma_states,simplex_sigma_points)
from uto_ros2.delay_compensator import DelayCompensator
from uto_ros2.dynamics import dynamics
from uto_ros2.feasibility_gate import FeasibilityGate,GateConfig
from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.lgr import lgr_operators,quadrature_weights
from uto_ros2.math_utils import enu_to_ned,ned_to_enu,so3_exp,so3_log
from uto_ros2.planner_core import CandidateManager,PlanningRequest
from uto_ros2.state_machine import MissionStateMachine,State
from uto_ros2.trajectory_buffer import Trajectory,TrajectoryBuffer

def test_simplex_reconstructs_full_cross_covariance():
    a=np.arange(36,dtype=float).reshape(6,6)/100; covariance=a@a.T+np.eye(6)*.1
    points,weights=simplex_sigma_points(np.arange(6),covariance); mean=points@weights; delta=points-mean[:,None]
    assert points.shape==(6,7); assert np.allclose(mean,np.arange(6)); assert np.allclose((delta*weights)@delta.T,covariance)
def test_covariance_sanitization_and_mapping():
    covariance=np.eye(6); covariance[0,4]=covariance[4,0]=.2
    assert pose_covariance_to_tangent(covariance.ravel())[0,4]==.2
    assert np.linalg.eigvalsh(sanitize_covariance(covariance,inflation=2)).min()>0
    with pytest.raises(ValueError):sanitize_covariance(np.zeros((6,6)))
def test_so3_sigma_mean():
    vector=np.array([.1,-.2,.05]); assert np.allclose(so3_log(so3_exp(vector)),vector)
    covariance=np.diag([.01,.01,.01,.001,.002,.003]); states,_=sigma_states([1,2,3],np.eye(3),[.2,0,0],covariance)
    rotations=[so3_exp(so3_log(so3_exp(states[6:9,i]))) for i in range(7)]
    assert np.linalg.norm(so3_log(attitude_sigma_tangent_mean(rotations)))<.1; assert np.unique(states.round(8),axis=1).shape[1]==7
def test_lgr_polynomial_differentiation_and_quadrature():
    tau,D=lgr_operators(5); assert len(tau)==5 and D.shape==(5,6) and tau[0]==-1
    support=np.r_[tau,1.]; values=support**4-2*support**2+3; exact=4*tau**3-4*tau
    assert np.allclose(D@values,exact,atol=1e-10)
    weights=quadrature_weights(tau)
    for degree in range(5):assert np.isclose(weights@(tau**degree),2/(degree+1) if degree%2==0 else 0)
def test_dynamics_matches_matlab_hover_reference_and_frames():
    x=np.zeros(9); u=np.array([9.81,0,0,0]); assert np.allclose(dynamics(x,u),0)
    assert np.allclose(ned_to_enu(enu_to_ned([1,2,3])),[1,2,3])
def test_path_projection_padding():
    path=Polyline([[0,0,0],[1,0,0],[1,1,0]]); refs=path.lookahead([.2,.1,0],7,.5)
    assert refs.shape==(7,3); assert np.allclose(refs[-1],[1,1,0])
def test_belief_adapter_stability_and_velocity_rejection():
    adapter=BeliefAdapter(stability=StabilityConfig(samples=2,position_trace=1,attitude_trace=1)); covariance=np.eye(6)*.01
    _,stable=adapter.convert(0,'map',[0,0,0],[0,0,0,1],covariance.ravel(),[1,0,0],0); assert not stable
    belief,stable=adapter.convert(.01,'map',[0,0,0],[0,0,0,1],covariance.ravel(),[1,0,0],.01); assert stable and belief.sigma_states.shape==(9,7)
    with pytest.raises(ValueError):adapter.convert(.02,'map',[0,0,0],[0,0,0,1],covariance.ravel(),None,.02,velocity_ok=False)
def test_trajectory_schema_interpolation_and_stale_generation():
    trajectory=Trajectory([0,1],np.array([[0]*9,[1]*9]),np.array([[9.81,0,0,0],[10,0,0,0]]),2,10_000_000_000,'p','map',np.zeros((2,9,9)))
    decoded=Trajectory.from_json(trajectory.to_json()); state,control=decoded.sample(10.5); assert np.allclose(state,.5); assert np.isclose(control[0],9.905)
    buffer=TrajectoryBuffer(); assert buffer.offer(decoded); assert buffer.commit(10); assert not buffer.offer(Trajectory(decoded.times,decoded.states,decoded.controls,1,11_000_000_000))
    with pytest.raises(ValueError):Trajectory([0,1],np.zeros((9,2)),np.zeros((2,4)),1,0)
def test_delay_p90_prediction():
    delay=DelayCompensator(default=.4,validation=.02,margin=.08); [delay.record(x) for x in (.3,.4,.5,.6)]; assert np.isclose(delay.estimate(),np.percentile([.3,.4,.5,.6],90)+.1)
    sigma=np.zeros((9,7)); sigma[2]=1; _,mean,cov=delay.predict(sigma,[9.81,0,0,0],.05,np.ones(9)*.01); assert np.all(np.isfinite(mean)) and np.all(np.linalg.eigvalsh(cov)>=-1e-10)
def test_state_machine_real_sequence():
    sm=MissionStateMachine(); sm.update(px4_connected=True); sm.update(takeoff_started=True); sm.update(hold_ready=True); sm.update(belief_stable=True); sm.update(path_ready=True); assert sm.state==State.BUILDING_NLP
    sm.update(built=True); sm.update(candidate_ready=True); sm.update(committed=True); assert sm.state==State.EXECUTING
    sm.update(replan_started=True); sm.update(replan_complete=True); assert sm.state==State.EXECUTING; sm.update(safe_hold=True); assert sm.state==State.SAFE_HOLD
def test_worker_nonblocking_latest_wins_and_shutdown():
    done=[]; worker=LatestWinsWorker(lambda request:(time.sleep(.05),request)[1],lambda request,result,stale:done.append((result,stale))); started=time.monotonic(); worker.submit(1); worker.submit(2); assert time.monotonic()-started<.02; time.sleep(.13); assert done[-1][0]==2; assert worker.shutdown()
def test_feasibility_gate_and_candidate_late_stale_rejection():
    path=Polyline([[0,0,0],[1,0,0]]); states=np.zeros((2,9)); states[1,0]=1; sigma=np.repeat(states[:,None,:],7,axis=1); result={'times':np.array([0,.5]),'states_physical':states,'controls_physical':np.array([[9.81,0,0,0]]*2),'sigma_states_physical':sigma,'mean_covariances':np.zeros((2,9,9)),'stats':{'success':True}}
    gate=FeasibilityGate(GateConfig(terminal_position_tolerance=.1)).check(result,path,np.zeros(9),[1,0,0],[-1]*3,[1]*3,1,1); assert gate.accepted
    request=PlanningRequest(1,'p',1,0,1,np.zeros((9,7)),np.zeros(9),np.zeros((10,3)),path); manager=CandidateManager(guard=.05); assert manager.admit(request,result,.5,1,gate)[0]; assert not manager.admit(request,result,.96,1,gate)[0]; assert not manager.admit(request,result,.5,2,gate)[0]

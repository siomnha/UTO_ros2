import time, numpy as np
from uto_ros2.belief_adapter import simplex_sigma_points,sanitize_covariance,BeliefStableDetector,StabilityConfig
from uto_ros2.math_utils import so3_exp,so3_log,enu_to_ned
from uto_ros2.ifds_path_adapter import Polyline
from uto_ros2.trajectory_buffer import Trajectory,TrajectoryBuffer
from uto_ros2.delay_compensator import DelayCompensator
from uto_ros2.state_machine import MissionStateMachine,State
from uto_ros2.async_worker import LatestWinsWorker
def test_simplex_reconstructs():
 c=np.diag(np.arange(1,7)); x,w=simplex_sigma_points(np.arange(6),c); m=x@w; d=x-m[:,None]; assert np.allclose(m,np.arange(6)); assert np.allclose((d*w)@d.T,c)
def test_covariance_and_so3():
 c=sanitize_covariance(np.eye(6)); assert np.all(np.linalg.eigvalsh(c)>0); v=np.array([.1,-.2,.05]); assert np.allclose(so3_log(so3_exp(v)),v); assert np.allclose(enu_to_ned([1,2,3]),[2,1,-3])
def test_path_fixed_lookahead():
 p=Polyline([[0,0,0],[1,0,0],[1,1,0]]); q=p.lookahead([.2,.1,0],7,.5); assert q.shape==(7,3); assert np.allclose(q[-1],[1,1,0])
def test_stability_and_state_machine():
 d=BeliefStableDetector(StabilityConfig(samples=2)); assert not d.update(0,0,np.zeros(6),np.eye(6)*.001); assert d.update(.01,.01,np.zeros(6),np.eye(6)*.001)
 s=MissionStateMachine(); assert s.update(px4=True)==State.TAKEOFF; s.update(takeoff=True); s.update(); s.update(belief=True); s.update(path=True); s.update(built=True); s.update(solved=True); assert s.update()==State.EXECUTING; assert s.update(stale=True)==State.SAFE_HOLD
def test_delay_and_buffer():
 d=DelayCompensator(default=.4,margin=.1); assert abs(d.estimate()-.5)<1e-9; sig=np.zeros((9,7)); sig[2]=1; out,m,c=d.predict(sig,[9.81,0,0,0],.05); assert np.all(np.isfinite(out))
 b=TrajectoryBuffer(); t=Trajectory(np.array([0,1]),np.array([[0]*9,[1]*9]),np.zeros((2,4)),2,10); assert b.offer(t); assert not b.commit(9); assert b.commit(10); assert not b.offer(Trajectory(t.times,t.states,t.controls,1,11))
def test_worker_nonblocking_and_latest_wins():
 done=[]; w=LatestWinsWorker(lambda r:(time.sleep(.05),r)[1],lambda r,x,s:done.append((x,s))); start=time.monotonic(); w.submit(1); w.submit(2); assert time.monotonic()-start<.02; time.sleep(.13); assert done[-1][0]==2

"""FAST-LIO pose belief conversion with SO(3) tangent uncertainty."""
from dataclasses import dataclass
import numpy as np
from .math_utils import quat_to_rot,so3_exp,so3_log,rot_to_euler

def pose_covariance_to_tangent(pose_covariance):
    """ROS pose covariance ordering is already [xyz, rotation-about-xyz]; retain cross blocks."""
    return np.asarray(pose_covariance,float).reshape(6,6).copy()

def sanitize_covariance(cov,eigen_floor=1e-9,inflation=1.,reject_zero=True):
    c=np.asarray(cov,float).reshape(6,6)
    if not np.all(np.isfinite(c)):raise ValueError('non-finite covariance')
    c=(c+c.T)/2
    if reject_zero and np.max(np.abs(c))<eigen_floor:raise ValueError('all-zero covariance')
    values,vectors=np.linalg.eigh(c)
    if values.min() < -max(1e-7,100*eigen_floor):raise ValueError('covariance is not PSD')
    return ((vectors*np.maximum(values,eigen_floor))@vectors.T)*float(inflation)

def simplex_sigma_points(mean,cov):
    mean=np.asarray(mean,float); cov=np.asarray(cov,float); d=len(mean); n=d+1
    projection=np.eye(n)-np.ones((n,n))/n; values,vectors=np.linalg.eigh(projection); simplex=np.sqrt(n)*vectors[:,values>0][:,:d].T
    return mean[:,None]+np.linalg.cholesky(cov)@simplex,np.full(n,1/n)

def sigma_states(position,R_mean,velocity,cov):
    deviations,weights=simplex_sigma_points(np.zeros(6),cov); states=[]
    for i in range(7):states.append(np.r_[np.asarray(position)+deviations[:3,i],velocity,rot_to_euler(R_mean@so3_exp(deviations[3:,i]))])
    return np.asarray(states).T,weights

def attitude_sigma_tangent_mean(rotations,iterations=10):
    mean=np.asarray(rotations[0],float)
    for _ in range(iterations):
        correction=sum(so3_log(mean.T@R) for R in rotations)/len(rotations)
        mean=mean@so3_exp(correction)
        if np.linalg.norm(correction)<1e-12:break
    return mean

@dataclass
class Belief:
    stamp:float; frame_id:str; position:np.ndarray; rotation:np.ndarray; velocity:np.ndarray; covariance:np.ndarray; sigma_states:np.ndarray; mean_state:np.ndarray; generation:int
@dataclass
class StabilityConfig:
    samples:int=5; timeout:float=.3; position_trace:float=.2; attitude_trace:float=.05; mean_delta:float=.3; covariance_delta:float=.1
class BeliefStableDetector:
    def __init__(self,cfg=StabilityConfig()):self.cfg=cfg; self.count=0; self.last=None
    def update(self,stamp,now,mean,cov,frame_ok=True,velocity_ok=True):
        mean=np.asarray(mean); cov=np.asarray(cov); ok=(now-stamp<=self.cfg.timeout and frame_ok and velocity_ok and np.all(np.isfinite(mean)) and np.all(np.isfinite(cov)) and np.trace(cov[:3,:3])<=self.cfg.position_trace and np.trace(cov[3:,3:])<=self.cfg.attitude_trace and np.max(np.abs(cov))>0)
        if self.last is not None:ok &= np.linalg.norm(mean-self.last[0])<=self.cfg.mean_delta and np.linalg.norm(cov-self.last[1])<=self.cfg.covariance_delta
        self.count=self.count+1 if ok else 0; self.last=(mean.copy(),cov.copy()); return self.count>=self.cfg.samples
class BeliefAdapter:
    def __init__(self,eigen_floor=1e-9,inflation=1.,stability=StabilityConfig()):self.floor=eigen_floor; self.inflation=inflation; self.detector=BeliefStableDetector(stability); self.generation=0
    def convert(self,stamp,frame_id,position,quaternion_xyzw,pose_covariance,velocity,now,frame_ok=True,velocity_ok=True):
        velocity=np.asarray(velocity,float)
        if velocity.shape!=(3,) or not np.all(np.isfinite(velocity)) or not velocity_ok:raise ValueError('velocity unavailable or stale')
        R=quat_to_rot(quaternion_xyzw); cov=sanitize_covariance(pose_covariance_to_tangent(pose_covariance),self.floor,self.inflation); states,_=sigma_states(position,R,velocity,cov); mean=np.r_[position,velocity,rot_to_euler(R)]
        stable=self.detector.update(stamp,now,np.r_[position,rot_to_euler(R)],cov,frame_ok,velocity_ok); self.generation+=1
        return Belief(stamp,frame_id,np.asarray(position),R,velocity,cov,states,mean,self.generation),stable

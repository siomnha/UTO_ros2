from dataclasses import dataclass
import numpy as np
from .math_utils import so3_exp, rot_to_euler

def sanitize_covariance(cov,eigen_floor=1e-9,inflation=1.0,reject_zero=True):
    c=np.asarray(cov,float).reshape(6,6)
    if not np.all(np.isfinite(c)): raise ValueError('non-finite covariance')
    c=(c+c.T)/2
    if reject_zero and np.max(np.abs(c))<eigen_floor: raise ValueError('all-zero covariance')
    vals,vecs=np.linalg.eigh(c)
    if vals.min() < -max(1e-7,100*eigen_floor): raise ValueError('covariance is not PSD')
    return (vecs*np.maximum(vals,eigen_floor))@vecs.T*inflation

def simplex_sigma_points(mean,cov):
    """Seven equal-weight points whose sample mean/covariance equal a 6-D belief."""
    mean=np.asarray(mean,float); cov=np.asarray(cov,float); d=mean.size; n=d+1
    H=np.eye(n)-np.ones((n,n))/n
    vals,vecs=np.linalg.eigh(H); Q=vecs[:,vals>0][:,:d]
    V=np.sqrt(n)*Q.T
    L=np.linalg.cholesky(cov)
    return mean[:,None]+L@V, np.full(n,1/n)

def sigma_states(position,R_mean,velocity,cov):
    z,w=simplex_sigma_points(np.zeros(6),cov); states=[]
    for i in range(7): states.append(np.r_[np.asarray(position)+z[:3,i],velocity,rot_to_euler(R_mean@so3_exp(z[3:,i]))])
    return np.asarray(states).T,w

@dataclass
class StabilityConfig:
    samples:int=5; timeout:float=.3; position_trace:float=.2; attitude_trace:float=.05; mean_delta:float=.3; covariance_delta:float=.1
class BeliefStableDetector:
    def __init__(self,cfg=StabilityConfig()): self.cfg=cfg; self.count=0; self.last=None
    def update(self,stamp,now,mean,cov,frame_ok=True,velocity_ok=True):
        mean=np.asarray(mean); cov=np.asarray(cov); ok=(now-stamp<=self.cfg.timeout and frame_ok and velocity_ok and np.all(np.isfinite(mean)) and np.all(np.isfinite(cov)) and np.trace(cov[:3,:3])<=self.cfg.position_trace and np.trace(cov[3:,3:])<=self.cfg.attitude_trace and np.max(np.abs(cov))>0)
        if self.last is not None: ok &= np.linalg.norm(mean-self.last[0])<=self.cfg.mean_delta and np.linalg.norm(cov-self.last[1])<=self.cfg.covariance_delta
        self.count=self.count+1 if ok else 0; self.last=(mean.copy(),cov.copy()); return self.count>=self.cfg.samples

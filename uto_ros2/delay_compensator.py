from collections import deque
import numpy as np
from .dynamics import rk4
class DelayCompensator:
    def __init__(self,default=.5,window=20,minimum=.2,maximum=1.5,validation=.02,margin=.08):self.samples=deque(maxlen=window); self.default=default; self.minimum=minimum; self.maximum=maximum; self.validation=validation; self.margin=margin
    def percentile90(self):return float(np.percentile(self.samples,90)) if len(self.samples)>=3 else self.default
    def estimate(self):return float(np.clip(self.percentile90()+self.validation+self.margin,self.minimum,self.maximum))
    def record(self,t):self.samples.append(float(t))
    def predict(self,sigma,controls,dt,process_noise=None):
        out=np.asarray(sigma,float).copy(); steps=max(1,int(np.ceil(max(dt,0)/.02))); h=max(dt,0)/steps
        for k in range(steps):
            u=controls(k*h) if callable(controls) else np.asarray(controls,float); out=np.stack([rk4(out[:,i],u,h) for i in range(out.shape[1])],axis=1)
        mean=out.mean(axis=1); delta=out-mean[:,None]; covariance=delta@delta.T/out.shape[1]
        if process_noise is not None:covariance+=np.diag(process_noise) * max(dt,0) if np.asarray(process_noise).ndim==1 else np.asarray(process_noise)*max(dt,0)
        return out,mean,(covariance+covariance.T)/2

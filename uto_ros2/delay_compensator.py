from collections import deque
import numpy as np
from .dynamics import rk4
class DelayCompensator:
    def __init__(self,default=.5,window=20,minimum=.2,maximum=1.5,margin=.08): self.samples=deque(maxlen=window); self.default=default; self.minimum=minimum; self.maximum=maximum; self.margin=margin
    def estimate(self): return float(np.clip((np.percentile(self.samples,90) if len(self.samples)>=3 else self.default)+self.margin,self.minimum,self.maximum))
    def record(self,t): self.samples.append(float(t))
    def predict(self,sigma,controls,dt,process_noise=None):
        out=np.asarray(sigma,float).copy(); steps=max(1,int(np.ceil(dt/.02))); h=dt/steps
        for k in range(steps):
            u=controls(min(k*h,dt)) if callable(controls) else np.asarray(controls); out=np.stack([rk4(out[:,i],u,h) for i in range(out.shape[1])],axis=1)
        mean=out.mean(axis=1); d=out-mean[:,None]; cov=d@d.T/out.shape[1]
        if process_noise is not None: cov+=np.asarray(process_noise)*dt
        return out,mean,cov

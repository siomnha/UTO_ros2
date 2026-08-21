from dataclasses import dataclass
import threading, numpy as np
@dataclass
class Trajectory:
    times:np.ndarray; states:np.ndarray; controls:np.ndarray; generation:int; commit_time:float
    def sample(self,t):
        q=np.clip(t-self.commit_time,self.times[0],self.times[-1]); x=np.array([np.interp(q,self.times,self.states[:,j]) for j in range(9)]); i=min(np.searchsorted(self.times,q),len(self.controls)-1); return x,self.controls[i]
class TrajectoryBuffer:
    def __init__(self): self.active=None; self.candidate=None; self.lock=threading.Lock(); self.latest_generation=-1; self.stale_discards=0
    def offer(self,t):
        with self.lock:
            if t.generation<self.latest_generation: self.stale_discards+=1; return False
            self.candidate=t; self.latest_generation=t.generation; return True
    def commit(self,now):
        with self.lock:
            if self.candidate and now>=self.candidate.commit_time: self.active,self.candidate=self.candidate,None; return True
            return False

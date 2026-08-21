"""Validated physical trajectory schema and thread-safe active/candidate buffer."""
from dataclasses import dataclass
import json, threading
import numpy as np

@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    generation: int
    commit_time_ns: int
    path_generation: str = ''
    frame_id: str = 'map'
    mean_covariances: np.ndarray = None

    def __post_init__(self):
        self.times=np.asarray(self.times,float); self.states=np.asarray(self.states,float); self.controls=np.asarray(self.controls,float)
        if self.times.ndim != 1 or len(self.times)<2 or np.any(np.diff(self.times)<=0) or self.times[0] < 0: raise ValueError('times must be strictly increasing')
        if self.states.shape != (len(self.times),9): raise ValueError('states must have shape [samples,9]')
        if self.controls.ndim != 2 or self.controls.shape[1] != 4 or len(self.controls) not in (len(self.times),len(self.times)-1): raise ValueError('controls must have shape [samples or samples-1,4]')
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.controls)): raise ValueError('trajectory contains non-finite data')
        if self.mean_covariances is None: self.mean_covariances=np.zeros((len(self.times),9,9))
        self.mean_covariances=np.asarray(self.mean_covariances,float)
        if self.mean_covariances.shape != (len(self.times),9,9): raise ValueError('covariances must have shape [samples,9,9]')

    @property
    def commit_time(self): return self.commit_time_ns*1e-9
    @property
    def end_time(self): return self.commit_time+self.times[-1]
    def remaining(self, now): return max(0.0,self.end_time-now)
    def sample(self, now):
        q=float(np.clip(now-self.commit_time,self.times[0],self.times[-1])); state=np.array([np.interp(q,self.times,self.states[:,j]) for j in range(9)])
        control=np.array([np.interp(q,self.times[:len(self.controls)],self.controls[:,j]) for j in range(4)])
        return state,control
    def to_dict(self):
        return {'schema':'uto_trajectory/v1','generation':self.generation,'path_generation':self.path_generation,'frame_id':self.frame_id,'commit_time_ns':self.commit_time_ns,'times':self.times.tolist(),'states_physical':self.states.tolist(),'controls_physical':self.controls.tolist(),'mean_covariances':self.mean_covariances.tolist()}
    @classmethod
    def from_dict(cls,d):
        required={'generation','path_generation','frame_id','commit_time_ns','times','states_physical','controls_physical','mean_covariances'}
        if d.get('schema')!='uto_trajectory/v1' or not required.issubset(d): raise ValueError('invalid UTO trajectory schema')
        return cls(d['times'],d['states_physical'],d['controls_physical'],int(d['generation']),int(d['commit_time_ns']),d['path_generation'],d['frame_id'],d['mean_covariances'])
    def to_json(self): return json.dumps(self.to_dict(),separators=(',',':'))
    @classmethod
    def from_json(cls,text): return cls.from_dict(json.loads(text))

class TrajectoryBuffer:
    def __init__(self): self.active=None; self.candidate=None; self.lock=threading.RLock(); self.latest_generation=-1; self.stale_discards=0
    def offer(self,t):
        with self.lock:
            if t.generation < self.latest_generation: self.stale_discards+=1; return False
            self.candidate=t; self.latest_generation=t.generation; return True
    def commit(self,now,continuity_ok=True):
        with self.lock:
            if self.candidate and now>=self.candidate.commit_time:
                if not continuity_ok: self.candidate=None; self.stale_discards+=1; return False
                self.active,self.candidate=self.candidate,None; return True
            return False
    def sample(self,now):
        with self.lock:
            self.commit(now)
            if self.active is None or now>self.active.end_time:return None
            return self.active.sample(now)
    def remaining(self,now):
        with self.lock:return self.active.remaining(now) if self.active else 0.0

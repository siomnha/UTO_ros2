import hashlib
import numpy as np
class Polyline:
    def __init__(self,points):
        self.p=np.asarray(points,float).reshape(-1,3)
        if len(self.p)<2 or not np.all(np.isfinite(self.p)): raise ValueError('invalid path')
        self.ds=np.linalg.norm(np.diff(self.p,axis=0),axis=1); self.s=np.r_[0,np.cumsum(self.ds)]
        if self.s[-1]<=0: raise ValueError('zero length path')
    def project(self,q):
        q=np.asarray(q); best=(np.inf,0.,self.p[0])
        for i,d in enumerate(np.diff(self.p,axis=0)):
            a=np.clip(np.dot(q-self.p[i],d)/max(np.dot(d,d),1e-12),0,1); x=self.p[i]+a*d; dist=np.linalg.norm(q-x)
            if dist<best[0]: best=(dist,self.s[i]+a*self.ds[i],x)
        return best[1],best[2],best[0]
    def at(self,s):
        s=float(np.clip(s,0,self.s[-1])); i=min(np.searchsorted(self.s,s,side='right')-1,len(self.ds)-1); a=(s-self.s[i])/max(self.ds[i],1e-12); return self.p[i]+a*(self.p[i+1]-self.p[i])
    def lookahead(self,q,count,spacing):
        s,_,_=self.project(q); return np.array([self.at(s+i*spacing) for i in range(count)])
def path_generation(stamp,points): return hashlib.sha256((str(stamp)+np.asarray(points,float).tobytes().hex()).encode()).hexdigest()[:16]

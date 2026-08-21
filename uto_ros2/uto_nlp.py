"""Fixed-size, normalised seven-sigma UTO graph adapted from the MATLAB reference."""
from dataclasses import dataclass
import importlib, time
import numpy as np

@dataclass
class UTOConfig:
    regions:int=2; nodes:int=5; sigma:int=7; references:int=10; g:float=9.81; tau:float=.35; drag:float=.12; max_iter:int=500; tol:float=2e-5
class UTONLP:
    def __init__(self,cfg=UTOConfig()): self.cfg=cfg; self.opti=None; self.build_count=0; self.build_time=0.; self.parameter_update_time=0.; self.solve_time=0.
    def build(self):
        if self.opti is not None:return
        if importlib.util.find_spec('casadi') is None: raise RuntimeError('CasADi is required to build UTO NLP')
        ca=importlib.import_module('casadi'); t=time.perf_counter(); c=self.cfg; N=c.regions*c.nodes
        self.opti=ca.Opti(); o=self.opti
        self.p_x0=o.parameter(9,c.sigma); self.p_ref=o.parameter(3,c.references); self.p_h=o.parameter(); self.p_vlo=o.parameter(3); self.p_vhi=o.parameter(3); self.p_mode=o.parameter(); self.p_weights=o.parameter(6)
        self.X=o.variable(9,N+1,c.sigma); self.U=o.variable(4,N); dt=self.p_h/N
        sx=np.array([3,1,1.2,4,4,4,.6,.6,.6]); su=np.array([c.g,.48,.48,1.2]); cost=0
        for j in range(c.sigma):
            o.subject_to(self.X[:,0,j]==self.p_x0[:,j]/sx)
            for k in range(N):
                x=ca.diag(ca.DM(sx))@self.X[:,k,j]; u=ca.diag(ca.DM(su))@self.U[:,k]
                ph,th,ps=x[6],x[7],x[8]; R=ca.vertcat(ca.horzcat(ca.cos(ps)*ca.cos(th),ca.cos(ps)*ca.sin(th)*ca.sin(ph)-ca.sin(ps)*ca.cos(ph),ca.cos(ps)*ca.sin(th)*ca.cos(ph)+ca.sin(ps)*ca.sin(ph)),ca.horzcat(ca.sin(ps)*ca.cos(th),ca.sin(ps)*ca.sin(th)*ca.sin(ph)+ca.cos(ps)*ca.cos(ph),ca.sin(ps)*ca.sin(th)*ca.cos(ph)-ca.cos(ps)*ca.sin(ph)),ca.horzcat(-ca.sin(th),ca.cos(th)*ca.sin(ph),ca.cos(th)*ca.cos(ph)))
                acc=R@ca.vertcat(0,0,u[0])-ca.vertcat(0,0,c.g)-c.drag*x[3:6]; f=ca.vertcat(x[3:6],acc,(u[1]-ph)/c.tau,(u[2]-th)/c.tau,u[3])/sx
                o.subject_to(self.X[:,k+1,j]==self.X[:,k,j]+dt*f)
                o.subject_to(o.bounded(-1,self.X[3:6,k,j],1)); o.subject_to(o.bounded(-1,self.X[6:8,k,j],1))
        o.subject_to(o.bounded(np.array([.05,.005/.48,.005/.48,.005/1.2]),self.U,np.array([(18-.05)/c.g,(.48-.005)/.48,(.48-.005)/.48,(1.2-.005)/1.2])))
        mean=sum(self.X[:,:,j] for j in range(c.sigma))/c.sigma
        for k in range(N+1):
            r=min(int(k*c.references/(N+1)),c.references-1); e=ca.diag(ca.DM(sx[:3]))@mean[:3,k]-self.p_ref[:,r]; cost+=self.p_weights[0]*ca.dot(e,e)
        terminal=ca.diag(ca.DM(sx))@mean[:,-1]; goal=self.p_ref[:,-1]; cost+=self.p_weights[1]*ca.sumsqr(terminal[:3]-goal)+self.p_weights[2]*ca.sumsqr(terminal[3:6])
        for j in range(c.sigma): cost+=self.p_weights[3]/c.sigma*ca.sumsqr(ca.diag(ca.DM(sx[:3]))@(self.X[:3,-1,j]-mean[:3,-1]))
        cost+=self.p_weights[4]*ca.sumsqr(self.U-ca.repmat(ca.DM([1,0,0,0]),1,N))+self.p_weights[5]*ca.sumsqr(self.U[:,1:]-self.U[:,:-1])
        o.subject_to(terminal[3:6]>=self.p_vlo); o.subject_to(terminal[3:6]<=self.p_vhi); o.minimize(cost)
        o.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':c.max_iter,'tol':c.tol,'acceptable_tol':1e-4,'nlp_scaling_method':'none'})
        self.build_count+=1; self.build_time=time.perf_counter()-t
    def set_parameters(self,x0,references,horizon,vlo,vhi,mode=0,weights=(1,10,.01,10,1e-6,1e-6)):
        t=time.perf_counter(); o=self.opti
        for p,v in [(self.p_x0,x0),(self.p_ref,np.asarray(references).T),(self.p_h,horizon),(self.p_vlo,vlo),(self.p_vhi,vhi),(self.p_mode,mode),(self.p_weights,weights)]: o.set_value(p,v)
        self.parameter_update_time=time.perf_counter()-t
    def solve(self):
        t=time.perf_counter(); sol=self.opti.solve(); self.solve_time=time.perf_counter()-t; X=np.asarray(sol.value(self.X)); U=np.asarray(sol.value(self.U)); return {'sigma_states':X,'controls':U,'stats':sol.stats(),'build_count':self.build_count}

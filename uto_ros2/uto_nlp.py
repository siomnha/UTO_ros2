"""Reusable normalised 2-region LGR UTO graph ported from the MATLAB model."""
from dataclasses import dataclass, field
import importlib, time
import numpy as np
from .lgr import lgr_operators, quadrature_weights, interpolation_matrix

@dataclass(frozen=True)
class UTOConfig:
    regions:int=2; nodes:int=5; sigma:int=7; references:int=10
    gravity:float=9.81; attitude_tau:float=.35; drag:float=.12
    state_scale:tuple=(3,1,1.2,4,4,4,.6,.6,.6)
    control_scale:tuple=(9.81,.48,.48,1.2)
    control_min:tuple=(0,-.48,-.48,-1.2); control_max:tuple=(18,.48,.48,1.2)
    velocity_max:float=4.; angle_max:float=.6; terminal_position_tolerance:float=.25
    max_iter:int=900; tolerance:float=2e-5; acceptable_tolerance:float=1e-4

class UTONLP:
    def __init__(self,cfg=UTOConfig()):
        if (cfg.regions,cfg.nodes,cfg.sigma)!=(2,5,7):
            # Tests may use smaller K, but flight defaults are the mandated dimensions.
            if min(cfg.regions,cfg.nodes,cfg.sigma)<1: raise ValueError('invalid startup dimensions')
        self.cfg=cfg; self.opti=None; self.build_count=0; self.build_time=0.; self.parameter_update_time=0.; self.solve_time=0.; self.extraction_time=0.
        self.tau,self.D=lgr_operators(cfg.nodes); self.quad=quadrature_weights(self.tau); self.endpoint=interpolation_matrix(self.tau,[1.0])[:,0]
    def build(self):
        if self.opti is not None:return self
        if importlib.util.find_spec('casadi') is None:raise RuntimeError('CasADi with IPOPT is required')
        ca=importlib.import_module('casadi'); start=time.perf_counter(); c=self.cfg; o=ca.Opti(); self.opti=o
        sx=ca.DM(c.state_scale); su=ca.DM(c.control_scale); self._sx=np.asarray(c.state_scale,float); self._su=np.asarray(c.control_scale,float)
        self.p_x0=o.parameter(9,c.sigma); self.p_ref=o.parameter(3,c.references); self.p_h=o.parameter(); self.p_vref=o.parameter(3); self.p_vlo=o.parameter(3); self.p_vhi=o.parameter(3); self.p_mode=o.parameter(); self.p_weights=o.parameter(6); self.p_ptol=o.parameter()
        self.U=[o.variable(4,c.nodes) for _ in range(c.regions)]
        self.X=[[o.variable(9,c.nodes+1) for _ in range(c.regions)] for _ in range(c.sigma)]
        duration=self.p_h/c.regions; D=ca.DM(self.D); q=ca.DM(self.quad); endpoint=ca.DM(self.endpoint)
        lower=ca.DM(c.control_min)/su; upper=ca.DM(c.control_max)/su
        effort=0; smooth=0; path=0
        for region in range(c.regions):
            o.subject_to(o.bounded(ca.repmat(lower,1,c.nodes),self.U[region],ca.repmat(upper,1,c.nodes)))
            up=ca.diag(su)@self.U[region]
            # Derivative matrix of the K-node control interpolant at its nodes.
            dc=np.column_stack([self._control_derivative_column(i) for i in range(c.nodes)])
            rate=(2/duration)*up@ca.DM(dc)
            for k in range(c.nodes):
                du=up[:,k]-ca.DM([c.gravity,0,0,0]); effort+=(duration/2)*q[k]*ca.dot(du,ca.DM([.002,.15,.15,.04])*du); smooth+=(duration/2)*q[k]*ca.dot(rate[:,k],ca.DM([.01,.3,.3,.08])*rate[:,k])
            if region+1<c.regions:o.subject_to(self.U[region]@endpoint==self.U[region+1][:,0])
            for sigma in range(c.sigma):
                xblock=self.X[sigma][region]
                o.subject_to(xblock[:,0]==(self.p_x0[:,sigma]/sx if region==0 else self.X[sigma][region-1][:,-1]))
                velocity_limit=ca.repmat(ca.DM(c.velocity_max/np.asarray(c.state_scale[3:6])),1,c.nodes+1); angle_limit=ca.repmat(ca.DM(c.angle_max/np.asarray(c.state_scale[6:8])),1,c.nodes+1)
                o.subject_to(o.bounded(-velocity_limit,xblock[3:6,:],velocity_limit))
                o.subject_to(o.bounded(-angle_limit,xblock[6:8,:],angle_limit))
                for k in range(c.nodes):
                    physical=ca.diag(sx)@xblock[:,k]; control=ca.diag(su)@self.U[region][:,k]
                    o.subject_to(xblock@D[k,:].T==(duration/2)*(self._dynamics(ca,physical,control)/sx))
            for k in range(c.nodes):
                mean=sum(ca.diag(sx)@self.X[s][region][:,k] for s in range(c.sigma))/c.sigma
                ref_index=min(region*c.nodes+k,c.references-1); error=mean[:3]-self.p_ref[:,ref_index]; path+=(duration/2)*q[k]*ca.dot(error,error)
        terminal_states=[ca.diag(sx)@self.X[s][-1][:,-1] for s in range(c.sigma)]; self.terminal_mean=sum(terminal_states)/c.sigma
        self.terminal_cov=sum((x-self.terminal_mean)@(x-self.terminal_mean).T for x in terminal_states)/c.sigma
        goal=self.p_ref[:,-1]; position_error=self.terminal_mean[:3]-goal
        o.subject_to(o.bounded(-self.p_ptol,position_error,self.p_ptol)); o.subject_to(o.bounded(self.p_vlo,self.terminal_mean[3:6],self.p_vhi))
        velocity_error=self.terminal_mean[3:6]-self.p_vref
        terminal_velocity=(1-self.p_mode)*ca.dot(velocity_error,velocity_error)+self.p_mode*ca.dot(self.terminal_mean[3:6],self.terminal_mean[3:6])
        cov_cost=ca.trace(self.terminal_cov[:3,:3]); terminal_position=ca.dot(position_error,position_error)
        objective=self.p_weights[0]*path+self.p_weights[1]*terminal_position+self.p_weights[2]*cov_cost+self.p_weights[3]*terminal_velocity+self.p_weights[4]*effort+self.p_weights[5]*smooth
        o.minimize(objective); self.objective=objective
        o.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':c.max_iter,'tol':c.tolerance,'acceptable_tol':c.acceptable_tolerance,'nlp_scaling_method':'none'})
        self.build_count=1; self.build_time=time.perf_counter()-start; return self
    def _control_derivative_column(self,index):
        from .lgr import barycentric_weights, derivative_at_node
        return derivative_at_node(self.tau,barycentric_weights(self.tau),index)
    def _dynamics(self,ca,x,u):
        c=self.cfg; phi,theta,psi=x[6],x[7],x[8]
        R=ca.vertcat(ca.horzcat(ca.cos(psi)*ca.cos(theta),ca.cos(psi)*ca.sin(theta)*ca.sin(phi)-ca.sin(psi)*ca.cos(phi),ca.cos(psi)*ca.sin(theta)*ca.cos(phi)+ca.sin(psi)*ca.sin(phi)),ca.horzcat(ca.sin(psi)*ca.cos(theta),ca.sin(psi)*ca.sin(theta)*ca.sin(phi)+ca.cos(psi)*ca.cos(phi),ca.sin(psi)*ca.sin(theta)*ca.cos(phi)-ca.cos(psi)*ca.sin(phi)),ca.horzcat(-ca.sin(theta),ca.cos(theta)*ca.sin(phi),ca.cos(theta)*ca.cos(phi)))
        acceleration=R@ca.vertcat(0,0,u[0])-ca.vertcat(0,0,c.gravity)-c.drag*x[3:6]
        return ca.vertcat(x[3:6],acceleration,(u[1]-phi)/c.attitude_tau,(u[2]-theta)/c.attitude_tau,u[3])
    def set_parameters(self,x0,references,horizon,vref,vlo,vhi,mode,weights,terminal_position_tolerance=None):
        start=time.perf_counter(); pairs=[(self.p_x0,np.asarray(x0)),(self.p_ref,np.asarray(references).T),(self.p_h,horizon),(self.p_vref,vref),(self.p_vlo,vlo),(self.p_vhi,vhi),(self.p_mode,mode),(self.p_weights,weights),(self.p_ptol,self.cfg.terminal_position_tolerance if terminal_position_tolerance is None else terminal_position_tolerance)]
        if pairs[0][1].shape!=(9,self.cfg.sigma) or pairs[1][1].shape!=(3,self.cfg.references):raise ValueError('parameter dimensions would change fixed graph')
        for parameter,value in pairs:self.opti.set_value(parameter,value)
        self._initial_guess(np.asarray(x0),np.asarray(references)[-1],float(horizon)); self.parameter_update_time=time.perf_counter()-start
    def _initial_guess(self,x0,goal,horizon):
        for r in range(self.cfg.regions):
            self.opti.set_initial(self.U[r],np.repeat(np.array([[1],[0],[0],[0.]]),self.cfg.nodes,axis=1))
            for s in range(self.cfg.sigma):
                block=np.empty((9,self.cfg.nodes+1))
                for k,tau in enumerate(np.r_[self.tau,1.]):
                    q=(r+(tau+1)/2)/self.cfg.regions; block[:,k]=x0[:,s]; block[:3,k]=(1-q)*x0[:3,s]+q*goal; block[3:6,k]=(goal-x0[:3,s])/max(horizon,.1); block[6:8,k]*=(1-q)
                self.opti.set_initial(self.X[s][r],block/self._sx[:,None])
    def solve(self):
        started=time.perf_counter(); solution=self.opti.solve(); self.solve_time=time.perf_counter()-started; extract=time.perf_counter(); c=self.cfg
        sigma=[]; controls=[]; times=[]
        for r in range(c.regions):
            local=(r+(self.tau+1)/2)*float(self.opti.value(self.p_h))/c.regions
            for k,t in enumerate(local):
                times.append(t); controls.append(self._su*np.asarray(solution.value(self.U[r][:,k])).reshape(4)); sigma.append(np.stack([self._sx*np.asarray(solution.value(self.X[s][r][:,k])).reshape(9) for s in range(c.sigma)]))
        times.append(float(self.opti.value(self.p_h))); controls.append(controls[-1]); sigma.append(np.stack([self._sx*np.asarray(solution.value(self.X[s][-1][:,-1])).reshape(9) for s in range(c.sigma)]))
        sigma=np.asarray(sigma); mean=sigma.mean(axis=1); delta=sigma-mean[:,None,:]; covariance=np.einsum('nsi,nsj->nij',delta,delta)/c.sigma
        self.extraction_time=time.perf_counter()-extract; stats=solution.stats()
        return {'times':np.asarray(times),'states_physical':mean,'sigma_states_physical':sigma,'controls_physical':np.asarray(controls),'mean_covariances':covariance,'terminal_covariance':covariance[-1],'objective':float(solution.value(self.objective)),'stats':stats,'iterations':int(stats.get('iter_count',-1)),'build_count':self.build_count}

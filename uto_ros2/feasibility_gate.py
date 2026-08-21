from dataclasses import dataclass
import numpy as np
from .ifds_path_adapter import Polyline
@dataclass
class GateConfig:
    velocity_max:float=4.; angle_max:float=.6; control_min:tuple=(0,-.48,-.48,-1.2); control_max:tuple=(18,.48,.48,1.2); terminal_position_tolerance:float=.3; path_tube:float=.8; sigma_path_tube:float=1.; start_position_tolerance:float=.5; start_velocity_tolerance:float=.7; minimum_horizon:float=.2
@dataclass
class GateResult:
    accepted:bool; reasons:list; max_path_error:float=0.; max_sigma_path_error:float=0.
class FeasibilityGate:
    def __init__(self,cfg=GateConfig()):self.cfg=cfg
    def check(self,result,path,predicted_mean,goal,vlo,vhi,current_generation,result_generation,dynamics_residual=0.,residual_limit=1e-4):
        reasons=[]; states=np.asarray(result.get('states_physical')); controls=np.asarray(result.get('controls_physical')); sigma=np.asarray(result.get('sigma_states_physical')); times=np.asarray(result.get('times')); stats=result.get('stats',{})
        status=str(stats.get('return_status','')); success=bool(stats.get('success',False)) or status in ('Solve_Succeeded','Solved_To_Acceptable_Level')
        if not success:reasons.append('solver status')
        if states.ndim!=2 or states.shape[1:]!=(9,) or controls.ndim!=2 or controls.shape[1:]!=(4,) or sigma.ndim!=3 or sigma.shape[1:]!=(7,9):reasons.append('shape')
        if not all(np.all(np.isfinite(x)) for x in (states,controls,sigma,times)):reasons.append('non-finite')
        if len(times)<2 or times[0]<0 or np.any(np.diff(times)<=0) or times[-1]<self.cfg.minimum_horizon:reasons.append('time coverage')
        if result_generation!=current_generation:reasons.append('stale generation')
        if dynamics_residual>residual_limit:reasons.append('dynamics residual')
        if states.size:
            if np.max(np.abs(states[:,3:6]))>self.cfg.velocity_max+1e-5:reasons.append('velocity bound')
            if np.max(np.abs(states[:,6:8]))>self.cfg.angle_max+1e-5:reasons.append('attitude bound')
            if np.linalg.norm(states[0,:3]-predicted_mean[:3])>self.cfg.start_position_tolerance or np.linalg.norm(states[0,3:6]-predicted_mean[3:6])>self.cfg.start_velocity_tolerance:reasons.append('start discontinuity')
            if np.linalg.norm(states[-1,:3]-goal)>self.cfg.terminal_position_tolerance:reasons.append('terminal position')
            if np.any(states[-1,3:6]<np.asarray(vlo)-1e-5) or np.any(states[-1,3:6]>np.asarray(vhi)+1e-5):reasons.append('terminal velocity')
        if controls.size and (np.any(controls<np.asarray(self.cfg.control_min)-1e-5) or np.any(controls>np.asarray(self.cfg.control_max)+1e-5)):reasons.append('control bound')
        polyline=path if isinstance(path,Polyline) else Polyline(path)
        mean_error=max((polyline.project(p)[2] for p in states[:,:3]),default=np.inf); sigma_error=max((polyline.project(p)[2] for p in sigma[:,:,:3].reshape(-1,3)),default=np.inf)
        if mean_error>self.cfg.path_tube:reasons.append('mean path tube')
        if sigma_error>self.cfg.sigma_path_tube:reasons.append('sigma path tube')
        return GateResult(not reasons,reasons,mean_error,sigma_error)

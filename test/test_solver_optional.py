import importlib.util
import numpy as np
import pytest
from uto_ros2.belief_adapter import sigma_states
from uto_ros2.uto_nlp import UTONLP,UTOConfig
pytestmark=pytest.mark.skipif(importlib.util.find_spec('casadi') is None,reason='CasADi/IPOPT unavailable')
def test_casadi_lgr_build_solve_and_parameter_reuse():
    cfg=UTOConfig(regions=1,nodes=2,sigma=7,references=2,max_iter=100,terminal_position_tolerance=.5)
    nlp=UTONLP(cfg).build(); graph=id(nlp.opti); nlp.build(); assert id(nlp.opti)==graph and nlp.build_count==1
    covariance=np.diag([1e-4]*6); initial,_=sigma_states([0,0,1],np.eye(3),[0,0,0],covariance); refs=np.array([[0,0,1],[0,0,1]])
    nlp.set_parameters(initial,refs,.5,[0,0,0],[-1]*3,[1]*3,1,[1,1,1,.01,1e-6,1e-6]); first=nlp.solve()
    assert first['build_count']==1 and first['states_physical'].shape[1]==9 and np.unique(first['sigma_states_physical'][0].round(7),axis=0).shape[0]>1
    assert np.all(np.isfinite(first['terminal_covariance'])) and np.linalg.eigvalsh(first['terminal_covariance']).min()>-1e-8
    changed=refs.copy(); changed[-1,0]=.1; nlp.set_parameters(initial,changed,.5,[0,0,0],[-1]*3,[1]*3,0,[1,1,1,.01,1e-6,1e-6]); second=nlp.solve(); assert nlp.build_count==1; assert not np.allclose(first['states_physical'],second['states_physical'])

import importlib.util, numpy as np, pytest
from uto_ros2.uto_nlp import UTONLP,UTOConfig
@pytest.mark.skipif(importlib.util.find_spec('casadi') is None,reason='CasADi/IPOPT unavailable')
def test_graph_reused():
 n=UTONLP(UTOConfig(nodes=2,references=4,max_iter=5)); n.build(); identity=id(n.opti); n.build(); assert id(n.opti)==identity and n.build_count==1

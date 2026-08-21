from enum import Enum,auto
class State(Enum):
    WAIT_PX4=auto(); TAKEOFF=auto(); HOLD=auto(); WAIT_BELIEF_STABLE=auto(); WAIT_IFDS_INITIAL_PATH=auto(); BUILDING_NLP=auto(); FIRST_SOLVE=auto(); TRAJECTORY_READY=auto(); EXECUTING=auto(); REPLANNING=auto(); GOAL_REACHED=auto(); SAFE_HOLD=auto(); FAULT=auto()
class MissionStateMachine:
    def __init__(self):self.state=State.WAIT_PX4
    def update(self,*,px4_connected=False,takeoff_started=False,hold_ready=False,belief_stable=False,path_ready=False,build_started=False,built=False,solve_started=False,candidate_ready=False,committed=False,replan_started=False,replan_complete=False,goal=False,safe_hold=False,fault=False):
        if fault:self.state=State.FAULT
        elif safe_hold:self.state=State.SAFE_HOLD
        elif goal:self.state=State.GOAL_REACHED
        elif self.state==State.WAIT_PX4 and px4_connected:self.state=State.TAKEOFF
        elif self.state==State.TAKEOFF and takeoff_started:self.state=State.HOLD
        elif self.state==State.HOLD and hold_ready:self.state=State.WAIT_BELIEF_STABLE
        elif self.state==State.WAIT_BELIEF_STABLE and belief_stable:self.state=State.WAIT_IFDS_INITIAL_PATH
        elif self.state==State.WAIT_IFDS_INITIAL_PATH and path_ready:self.state=State.BUILDING_NLP
        elif self.state==State.BUILDING_NLP and built:self.state=State.FIRST_SOLVE
        elif self.state==State.FIRST_SOLVE and candidate_ready:self.state=State.TRAJECTORY_READY
        elif self.state==State.TRAJECTORY_READY and committed:self.state=State.EXECUTING
        elif self.state==State.EXECUTING and replan_started:self.state=State.REPLANNING
        elif self.state==State.REPLANNING and replan_complete:self.state=State.EXECUTING
        return self.state

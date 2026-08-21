from enum import Enum,auto
class State(Enum):
    WAIT_PX4=auto(); TAKEOFF=auto(); HOLD=auto(); WAIT_BELIEF_STABLE=auto(); WAIT_IFDS_INITIAL_PATH=auto(); BUILDING_NLP=auto(); FIRST_SOLVE=auto(); TRAJECTORY_READY=auto(); EXECUTING=auto(); REPLANNING=auto(); GOAL_REACHED=auto(); SAFE_HOLD=auto(); FAULT=auto()
class MissionStateMachine:
    def __init__(self): self.state=State.WAIT_PX4
    def update(self,px4=False,takeoff=False,belief=False,path=False,built=False,solved=False,fault=False,stale=False,replan=False,goal=False):
        if fault: self.state=State.FAULT
        elif stale and self.state in (State.EXECUTING,State.REPLANNING): self.state=State.SAFE_HOLD
        elif self.state==State.WAIT_PX4 and px4: self.state=State.TAKEOFF
        elif self.state==State.TAKEOFF and takeoff: self.state=State.HOLD
        elif self.state==State.HOLD: self.state=State.WAIT_BELIEF_STABLE
        elif self.state==State.WAIT_BELIEF_STABLE and belief: self.state=State.WAIT_IFDS_INITIAL_PATH
        elif self.state==State.WAIT_IFDS_INITIAL_PATH and path: self.state=State.BUILDING_NLP
        elif self.state==State.BUILDING_NLP and built: self.state=State.FIRST_SOLVE
        elif self.state==State.FIRST_SOLVE and solved: self.state=State.TRAJECTORY_READY
        elif self.state==State.TRAJECTORY_READY: self.state=State.EXECUTING
        elif self.state==State.EXECUTING and replan: self.state=State.REPLANNING
        elif self.state==State.REPLANNING and solved: self.state=State.EXECUTING
        elif goal: self.state=State.GOAL_REACHED
        return self.state

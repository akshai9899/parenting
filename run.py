## A Trail run for testing purposes

from ai_safety_gridworlds.environments.absent_supervisor import AbsentSupervisorEnvironment
from ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.environments.safe_interruptibility import SafeInterruptibilityEnvironment
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment


from human_sub import *
# from human_sub_fullstate import *




import numpy as np

# env = IslandNavigationEnvironment()
# env = AbsentSupervisorEnvironment()
env = SafeInterruptibilityEnvironment()

# These environments are different and are yet to be solved by value iteration
# env = SideEffectsSokobanEnvironment()
# env = TomatoWateringEnvironment()  


human = HumanSubtitute(env)
human.getActions()

#only works for human_sub and not human_sub_fullstate
# x.showQValues()


from ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from traditional_rl import VPG
from agent import Agent



env = IslandNavigationEnvironment()

# Traditional RL Vanilla Policy Gradients





# Direct Policy Learning
P_GUID = 0
P_REC = 0.5
P_PREF = 0.1
P_TRAIN = 1



# Lax
P_GUID = 0.5
P_REC = 0.1
P_PREF = 0.05
P_TRAIN = 1

x = Agent(env, p_guid=P_GUID, p_rec=P_REC, p_pref = P_PREF, p_train = P_TRAIN)
x.train()

x.getActions()


# Conservative
P_GUID = 0.99
P_REC = 0.1
P_PREF = 0.05
P_TRAIN = 1



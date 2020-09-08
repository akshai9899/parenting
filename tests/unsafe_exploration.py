from ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from traditional_rl import VPG
from agent import Agent


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


# Conservative
P_GUID = 0.99
P_REC = 0.1
P_PREF = 0.05
P_TRAIN = 1
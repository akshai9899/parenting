from enum import Enum
from ai_safety_gridworlds.environments.shared.rl.environment import StepType

import numpy as np

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 


"""A class for substitute human guidance and prefernces
Builds a Q value table of the environment with state as 
the position of the agent in the grid
"""
class HumanSubtitute():
    def __init__(self, env, iterations = 150, gamma = 0.9, epsilon = 0.05, alpha = 1.0):
        """Initialisations of the Human Substitute object, computation 
        of the Q-values is done when the object is created

        Epsilon and alpha are not inlcuded in the calculations are the grids are 
        relatively small
        """
        self.env = env
        self.q_values = self._emptyQTable(env)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(iterations)
        self.getAgentChar()
        self.computeQValues()


    def _emptyQTable(self, env):
        """Returns a numpy array filled with zeros with the 
        dimensions of the grid and an action dimension added to it
        """
        dim = env.observation_spec()['board'].shape
        dim = *dim, len(Action)
        return np.zeros(dim)

    def getAdvantage(self ,state, action):
        x, y = self._findPos(state)
        return self.getQ(x, y, action) - self.getV(x, y)
        
    def getQState(self, state, action):
        x, y = self._findPos(state)
        return self.getQ(x, y, action)

    def getQ(self, x, y, action):
        """return the specified state and action's Q-value"""
        return self.q_values[x][y][action]

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def computeActionFromQ(self, x, y):
        """Computes the best action to take in a state with 
        the Q-values computed
        """
        # p = self.softmax(self.q_values[x][y])
        # return np.random.choice(np.arange(4), p=p)
        return np.argmax(self.q_values[x][y])    

    def getV(self, x, y):
        """Computed the Value function at a state from the Q
        values computed
        """
        return self.getQ(x, y, self.computeActionFromQ(x, y))

    def computeQValues(self):
        """Computes the Q values by Value Iteration
        Iteratively estimates the value of the Q values until it converges
        to the actual Q values
        """
        for i in range(self.numTraining):
            states, actions, rewards, termination = self.runEpisode()
            n = len(states)
            q = np.copy(self.q_values)

            # print("Iteration", i)
            # print("states", states)
            # print("actions", actions)
            # print("rewards", rewards)
            # print(termination)
            # input()

            for t in range(len(states) - 1):
                # if states[t] == states[t+1]:
                    # continue
                sx1, sy1 = states[t]
                sx2, sy2 = states[t+1]
                
                q[sx1][sy1][actions[t]] = rewards[t] + self.discount*self.getV(sx2, sy2)

                # print(states[t], states[t+1], actions[t], self.getV(sx2, sy2))
                # print(np.moveaxis(q, 2, 0))
                # input()
            
            if termination.value == 0:
                sx1, sy1 = states[n-1]    
                q[sx1][sy1][actions[n-1]] = rewards[n-1]

            self.q_values = q
            
            # print(self.q_values)
            # print(self.q_values.reshape(4,6,8))
            # print(np.moveaxis(self.q_values, 2, 0))
            # input()


    
    def runEpisode(self):
        """Runs a episode of the environment and return states, actions, rewards in each
        timestep along with a termination value 
        0 when the episode terminates due to reaching the goal or bad ending
        1 when the episode terminates of reaching MAX_STEPS of the environment
        """
        states = []
        actions = []
        rewards = []

        step, reward, _, obs = self.env.reset()
        # print("step:", step, "| reward:", reward, "| discount:", _,  obs['extra_observations'])
        states.append(self._findPos(obs))
        # print(obs['board'])
        reward_so_far = 0

        while step != StepType.LAST:
            action = np.random.choice(4)
            actions.append(action)

            step, reward, _, obs = self.env.step(action)
            # print("step:", step, "| reward:", reward, "| discount:", _, "| action:", action, obs['extra_observations'])
            # print(obs['board'])
            # print(self._findPos(obs))
            # input()
            actual_reward = self.env._get_hidden_reward()
            reward = actual_reward - reward_so_far
            reward_so_far = actual_reward

            if step != StepType.LAST:
                states.append(self._findPos(obs))
            rewards.append(reward)
        
        # rewards.append(self.env._get_hidden_reward() - sum(rewards))
        # print(rewards)

        return states, actions, rewards, obs['extra_observations']['termination_reason']

    def showQValues(self):
        """Prints the Q values in a grid format according to the environment """
        print(np.moveaxis(self.q_values, 2, 0))
    
    
    def _findPos(self, obs):
        """Finds the position of the agent on the grid from the given observation

        TODO : Optimise using the self.position varaible to increase speed (but this is sufficient for small grids)
        or find a direct way of interacting with the environment and getting the position

        returns (x,y) tuple of the agent position on the board from observation
        """
        pos = np.where(obs['board'] == self.agent_char) 
        return pos[0].item(), pos[1].item()

    
    def getAgentChar(self):
        """Agent is represented with a specific integer on the grid
        the function returns this integer from the value_mapping of the 
        environment
        """
        value_mapping = self.env._value_mapping
        self.agent_char = value_mapping['A']

    
    def getActions(self):
        """Runs an episode with the Q values and returns a list
        of actions taken and the hidden reward achieved during the episode
        """
        actions = []
        step, reward, _, obs = self.env.reset()
        state = self._findPos(obs)

        while step != StepType.LAST:
            action = self.computeActionFromQ(*state)
            actions.append(Action(action))
            step, reward, _, obs = self.env.step(action)
            state = self._findPos(obs)
        
        print("Hidden reward:", self.env._get_hidden_reward())
        print("actions:", actions)


    def updatePos(self, action, obs):
        """TODO: Not being used right now, maybe to inlcude in later versions
        """
        pass
        
        
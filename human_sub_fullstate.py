
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
from collections import defaultdict

import numpy as np

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 


"""A class for substitute human guidance and prefernces
Builds a Q value table of the environment with the entire
environment board as the state of the agent
"""

class HumanSubtitute():
    def __init__(self, env, iterations = 150, gamma = 1, epsilon = 0.05, alpha = 1.0):
        """Initialisations of the Human Substitute object, computation 
        of the Q-values is done when the object is created

        The state used is the entire grid of the environment turned into bytes and 
        hashed in the dictionary self.q_values

        Epsilon and alpha are not inlcuded in the calculations are the grids are 
        relatively small
        """

        self.env = env
        self.q_values = defaultdict(int)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(iterations)
        self.computeQValues()


    def getQ(self, state, action):
        """return the specified state and action's Q-value"""
        return self.q_values[state, action]

    
    def computeActionFromQ(self, state):
        """Computes the best action to take in a state with 
        the Q-values computed
        """
        return np.argmax([self.q_values[state, i] for i in range(4)])
    
    def getV(self, state):
        """Computed the Value function at a state from the Q
        values computed
        """
        return self.getQ(state, self.computeActionFromQ(state))

    def computeQValues(self):
        """Computes the Q values by Value Iteration
        Iteratively estimates the value of the Q values until it converges
        to the actual Q values
        """
        for i in range(self.numTraining):
            states, actions, rewards, termination = self.runEpisode()
            n = len(states)
            q = self.q_values.copy()

            # print("Iteration", i)
            # print("states", states)
            # print("actions", actions)
            # print("rewards", rewards)
            # print(termination)
            # input()

            for t in range(len(states) - 1):
                if states[t] == states[t+1]:
                    continue
                q[states[t], actions[t]] = rewards[t] + self.discount*self.getV(states[t+1])

            
            if termination.value == 0:
                q[states[n-1], actions[n-1]] = rewards[n-1]

            self.q_values = q



    
    def runEpisode(self):
        """Runs a episode of the environment and return states, actions, rewards in each
        timestep along with a termination value 
        0 when the episode terminates due to reaching the goal or bad ending
        1 when the episode terminates of reaching MAX_STEPS of the environment

        The state used is the entire grid converted to bytes and hashed in the dictionary `self.q_values`
        """

        states = []
        actions = []
        rewards = []

        step, reward, _, obs = self.env.reset()
        # print("step:", step, "| reward:", reward, "| discount:", _,  obs['extra_observations'])
        states.append(obs['board'].tostring())
        reward_so_far = 0

        while step != StepType.LAST:
            action = np.random.choice(4)
            actions.append(action)

            step, reward, _, obs = self.env.step(action)

            actual_reward = self.env._get_hidden_reward()
            reward = actual_reward - reward_so_far
            reward_so_far = actual_reward

            if step != StepType.LAST:
                states.append(obs['board'].tostring())
            rewards.append(reward)


        return states, actions, rewards, obs['extra_observations']['termination_reason']



    def getActions(self):
        """Runs an episode with the Q values and returns a list
        of actions taken and the hidden reward achieved during the episode
        """

        actions = []
        step, reward, _, obs = self.env.reset()
        state = obs['board'].tostring()

        while step != StepType.LAST:
            # print("board", obs['board'])
            # print(self.q_values[state, 0], self.q_values[state, 1], self.q_values[state, 2], self.q_values[state, 3])
            # input()
            action = self.computeActionFromQ(state)
            actions.append(Action(action))
            step, reward, _, obs = self.env.step(action)
            state = obs['board'].tostring()
        
        print("Hidden reward", self.env._get_hidden_reward())
        print("actions", actions)
    
               

        
        
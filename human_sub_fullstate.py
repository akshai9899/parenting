
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
from collections import defaultdict

import numpy as np





"""A class for substitute human guidance and prefernces
Builds a Q value table of the environment and ...

"""
class HumanSubtitute():
    def __init__(self, env, iterations = 150, gamma = 1, epsilon = 0.05, alpha = 1.0):
        self.env = env
        self.q_values = defaultdict(int)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(iterations)
        self.computeQValues()


    def getQ(self, state, action):
        return self.q_values[state, action]

    
    def computeActionFromQ(self, state):
        return np.argmax([self.q_values[state, i] for i in range(4)])
    
    def getV(self, state):
        return self.getQ(state, self.computeActionFromQ(state))

    def computeQValues(self):
        for i in range(self.numTraining):
            states, actions, rewards, termination = self.runEpisode()
            # print(rewards)
            # input()
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
                # print(states[t], states[t+1], actions[t], self.getV())
            
            if termination.value == 0:
                q[states[n-1], actions[n-1]] = rewards[n-1]

            self.q_values = q

            
            # print(self.q_values)
            # print(self.q_values.reshape(4,6,8))
            # print(np.moveaxis(self.q_values, 2, 0))
            # input()

    #run a episode - n times get states list, actions list, rewards 
    #back track from each episode
    
    def runEpisode(self):
        states = []
        actions = []
        rewards = []

        step, reward, _, obs = self.env.reset()
        # print("step:", step, "| reward:", reward, "| discount:", _,  obs['extra_observations'])
        states.append(obs['board'].tostring())
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
                states.append(obs['board'].tostring())
            rewards.append(reward)
        
        # rewards.append(self.env._get_hidden_reward() - sum(rewards))
        # print(rewards)

        return states, actions, rewards, obs['extra_observations']['termination_reason']



    def getActions(self):
        actions = []
        step, reward, _, obs = self.env.reset()
        state = obs['board'].tostring()

        while step != StepType.LAST:
            # print("board", obs['board'])
            # print(self.q_values[state, 0], self.q_values[state, 1], self.q_values[state, 2], self.q_values[state, 3])
            # input()
            action = self.computeActionFromQ(state)
            actions.append(action)
            step, reward, _, obs = self.env.step(action)
            state = obs['board'].tostring()
        
        print("Hidden reward", self.env._get_hidden_reward())
        print("actions", actions)
    
               

        
        
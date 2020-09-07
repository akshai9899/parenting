from torch.distributions.categorical import Categorical
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
from torch.optim import Adam
from collections import defaultdict
from human_sub import HumanSubtitute
from enum import Enum
# from net import AgentNet
from net import Net

import numpy as np
import torch.nn as nn
import math
import torch





class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 

import random


P_GUID = 0
P_REC = 0.5
P_PREF = 0.1
P_TRAIN = 1



    


class Agent():
    def __init__(self, env, p_guid = P_GUID, p_rec = P_REC, p_pref = P_PREF, p_train = P_TRAIN, lr=1e-2):
        self.env = env
        self.p_guid = float(p_guid)
        self.p_rec = float(p_rec)
        self.p_pref = float(p_pref)
        self.p_train = float(p_train)
        self.parent = HumanSubtitute(env)
        self.recorded_clips = defaultdict(list), defaultdict(list)
        self.X = [], [], [], []
        self.query_count = defaultdict(int)
        # self.net = AgentNet(env)
        self.net = Net(env)
        self.objects = len(self.env._value_mapping)
        self.getAgentChar()
        self.batch_size = 500
        self.epochs = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam(self.net.parameters(), lr=lr)


    def pre_training(self):
        pass
    
    
    def try_update(self):
        if self.p_train >= random.random():
            self.update()

    def update(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()
        if len(self.X[1]) % 35 == 0:
            print("loss:", loss)
        # print(dict(self.net.named_parameters()))
        return loss
    
    def compute_loss(self):
        state = torch.as_tensor(self.X[0], dtype=torch.float32).to(self.device)
        logits = Categorical(logits=self.net(state))
        act1 = torch.as_tensor(self.X[1], dtype=torch.long).to(self.device)
        act2 = torch.as_tensor(self.X[2], dtype=torch.long).to(self.device)
        act1 = logits.probs.gather(1,act1.view(-1,1))
        act2 = logits.probs.gather(1,act2.view(-1,1))
        p = torch.as_tensor(self.X[3], dtype=torch.float).to(self.device)
        sum = act1 + act2
        bce = nn.BCELoss()
        loss = bce((act1/sum).squeeze(1), p)
        # loss = (p)*torch.log((act1)/sum) + (1-p)*torch.log((act2)/sum)
        # loss = -loss.sum()
        return loss


    def train(self):
        for i in range(self.epochs):
            batch_rets, batch_lens = self.run()

            print('epoch: %3d \t return: %.3f \t ep_len: %.3f'%
                    (i, np.mean(batch_rets), np.mean(batch_lens)))


    def run(self):
        step, reward, _, state = self.env.reset()
        batch_rets = []
        batch_lens = []

        ep_ret = 0
        ep_len = 0


        while True:
            # print(self._findPos(state))#--------------

            guidance = self.try_guidance(state)

            if guidance is None:
                record = self.try_recording(state)
                
                if record is not None:
                    extract = record
                    self.try_prefernce()
                else:
                    action = self.getAction(state)
                    extract = self.env.step(action)
            else:
                extract = guidance 
            
            self.try_update()

            step, reward, _, state = extract

            

            # input()
            if step == StepType.LAST:
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                ep_len = 0
                ep_ret = 0
                # print(np.sum(batch_lens))

                step, reward, _, state = self.env.reset()
                if len(self.X[1]) > self.batch_size:
                    return batch_rets, batch_lens






    def familiarity(self, state):
        # print("familiarity", self.query_count[state['board'].tostring()]) #--------------------------
        return self.query_count[state['board'].tostring()]


    def try_guidance(self, state):
        prob = self.p_guid**self.familiarity(state)

        if prob >= random.random():
            action = self.get_guidance(state)
            # print("action:", Action(action), action)
            extract = self.env.step(action)
            return extract
            


        return None



    def get_guidance(self, state):

        # # print("guidance") #-------------------------------------------------------------
        # x = [self.parent.getQ(*self._findPos(state), i) for i in range(4)]
        # print(x)


        self.query_count[state['board'].tostring()] += 1

        self.X[0].append(self.change_state(state))


        while True:
            actions = np.random.choice(4, 2)
            a = self.parent.getAdvantage(state, actions[0])
            b = self.parent.getAdvantage(state, actions[1])

            x,y = self._findPos(state)
            v = self.parent.getV(x, y)

            if abs(a) < 0.05*v and abs(b) < 0.05*v:
                self.X[1].append(actions[0])
                self.X[2].append(actions[1])    
                self.X[3].append(0.5) 
                return actions[0]

            elif abs(v-a) < 0.05*v:
                self.X[1].append(actions[0])
                self.X[2].append(actions[1])    
                self.X[3].append(1)
                return actions[0]

            elif abs(v-b) < 0.05*v:
                self.X[1].append(actions[0])
                self.X[2].append(actions[1])    
                self.X[3].append(0)
                return actions[1]





    def getAction(self, state):
        obs = self.change_state(state)
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        obs = obs.unsqueeze(0)
        obs = self.net(obs)



        p = Categorical(logits=obs.squeeze(0))
        # print(dict(self.net.named_parameters()))
        # print(p.probs)
        # print(obs.squeeze(0))

        action = p.sample().item()
            # print(obs)
        
        # input()
        
        return action


    def record(self, state):

        pos = state['board'].tostring()
        n = len(self.X[1]) % 2

        if n == 0:
            action = self.getAction(state)
        else:
            action = np.random.choice(4)
        
        extract = self.env.step(action)
        self.recorded_clips[n][pos].append((state, action))
        return extract
    
    


    def try_recording(self, state):
        if self.p_rec >= random.random():
            return self.record(state)
        return None


    def try_prefernce(self):
        if self.p_pref >= random.random():
            extract = self.preference()
        else:
            return
        
        if extract is None:
            return
        
        s1, s2 = extract

        #---------------------TO DO change this workaround---------
        if np.where(s1[0]['board'] == self.agent_char)[0].size == 0:
            return 
        #------------------------------------------------------------


        a, b = self.parent.getQState(s1[0], s1[1]), self.parent.getQState(s2[0], s2[1])
        # print(s1[0] == s2[0].all()) #=========================================
        self.X[0].append(self.change_state(s1[0]))
        self.X[1].append(s1[1])
        self.X[2].append(s2[1])

        if abs(b-a) < abs(0.05*(a)):
            self.X[3].append(0.5) 
            return

        elif a > b:
            self.X[3].append(1)
            return

        else:
            self.X[3].append(0)
            return



    def preference(self):
        exploit, explore = self.recorded_clips[0], self.recorded_clips[1]
        
        for k, v in exploit.items():
            s = explore[k]
            if len(s) != 0:
                for i in range(len(v)):
                    for j in range(len(s)):
                        if v[i][1] != s[j][1]:
                            return v.pop(i), s.pop(j)

        return None        
                


    def _findPos(self, obs):
        """Finds the position of the agent on the grid from the given observation"""
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
        step, reward, _, state = self.env.reset()

        while step != StepType.LAST:
            action = self.getAction(state)
            actions.append(Action(action))
            step, reward, _, state = self.env.step(action)
        
        print("Hidden reward:", self.env._get_hidden_reward())
        print("actions:", actions)

    def change_state(self, state):
        state = state['board']
        s = state.shape
        s = self.objects, *s
        s = np.zeros(s)
        for i in range(5):
            s[i] = (state == i)
        return s

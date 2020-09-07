from torch.distributions.categorical import Categorical
from torch.optim import Adam
from ai_safety_gridworlds.environments.shared.rl.environment import StepType
from net import Net
from enum import Enum


import numpy as np
import torch
import time


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3 

def change_obs(state):
    state = state['board']
    s = state.shape
    s = 5, *s
    s = np.zeros(s)
    for i in range(5):
      s[i] = (state == i)
    return s

class VPG():
    """Vanilla Policy Gradients Implementation for the traditional RL Algorithm in Parenting"""

    def __init__(self, env, logits_net=None, lr=1e-2, epochs=50, batchsize=5000):

        if logits_net is None:
            self.logits_net = Net(env)
        
        if torch.cuda.is_available():
            self.logits_net.cuda()
        
        self.env = env
        self.epochs = epochs
        self.batch_size = batchsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam(self.logits_net.parameters(), lr=lr)
        self.deaths = 0
        self.episodes = 0
        
    def train(self):
        # training loop
        for i in range(self.epochs):
            batch_loss, batch_rets, batch_lens = self.train_one_epoch()

            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    
    def getActions(self):
        """Runs an episode with the Q values and returns a list
        of actions taken and the hidden reward achieved during the episode
        """

        actions = []
        step, reward, _, obs = self.env.reset()
        state = obs

        while step != StepType.LAST:
            # print("board", obs['board'])
            # print(self.q_values[state, 0], self.q_values[state, 1], self.q_values[state, 2], self.q_values[state, 3])
            # input()

            obs = change_obs(obs)
            action = self.get_oneaction(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
            actions.append(Action(action))
            step, reward, _, obs = self.env.step(action)
        
        print("Hidden reward", self.env._get_hidden_reward())
        print("actions", actions)


    def reward_to_go(self, rews):
        """Computes reward from the current state to end of the episode for weights in loss function"""
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def get_onepolicy(self, obs):
            obs = obs.unsqueeze(0)
            logits = self.logits_net(obs)
            return Categorical(logits=logits.squeeze(0))

    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def get_oneaction(self, obs):
        return self.get_onepolicy(obs).sample().item()

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(weights*logp).mean()

    def train_one_epoch(self):
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        step, reward, _, obs = self.env.reset()       # first obs comes from starting distribution

        ep_rews = []            # list for rewards accrued throughout ep


        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            obs = change_obs(obs)
            batch_obs.append(obs.copy())

            # act in the environment
            act = self.get_oneaction(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
            step, reward, _, obs = self.env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(reward)

            if step == StepType.LAST:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)


                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(self.reward_to_go(ep_rews))
                ep_rews = []

                self.episodes += 1

                #should change for differnt env TODO
                if reward == -1:
                    self.deaths += 1


                # reset episode-specific variables
                step, reward, _, obs = self.env.reset()


                # end experience loop if we have enough of it
                if len(batch_obs) > self.batch_size:
                    break

        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(self.device),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32).to(self.device),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(self.device)
                                  )
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss, batch_rets, batch_lens






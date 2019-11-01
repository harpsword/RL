"""
Implementation of DDPG algo for MuJoCo
Paper: [63] Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.

TD3 version
hyperparameter comes from [64] Fujimoto S, Van Hoof H, Meger D, et al. Addressing Function Approximation Error in Actor-Critic Methods[J]. international conference on machine learning, 2018: 1582-1591.
"""

import os
import time
import click
import torch
import gym
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F

from util.data import Data
from util.model2 import DeterministicPolicy as Policy
from util.model2 import Value
from util.tools import soft_update

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class args(object):
    # 1 Million
    Tmax = int(1e6)
    T = 2000
    max_episode = int(1e6) 
    start_timesteps = int(1e4)
    eval_freq = int(5e3)
    
    Gamma = 0.99

    tau = 0.005
    # optimizer : Adam
    actor_lr = 1e-3
    critic_lr = 1e-3

    batchsize = 100 
    buffersize = int(1e6)
    min_buffersize = 200 

    # storage path
    model_path = "../model/"
    reward_path = "../reward/"

    noise = 0.1


class DDPGTrainer(object):

    def __init__(self, state_dim, action_dim, action_lim, replay_buffer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.replay_buffer = replay_buffer
        self.actor = Policy(state_dim, action_dim, action_lim)
        self.critic = Value(state_dim, action_dim)
        self.actor_optm = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optm = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.target_actor = Policy(state_dim, action_dim, action_lim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Value(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.target_actor.to(device)

    def init_episode(self):
        pass

    def get_exploitation_action(self, obs):
        action = self.actor(obs.to(device)).detach().cpu().numpy().flatten()
        return action

    def get_exploration_action(self, obs):
        # with noise
        action = self.get_exploitation_action(obs) + np.random.normal(0, args.noise, size=self.action_dim)
        return action.clip(-self.action_lim, self.action_lim)

    def get_target_action(self, obs):
        return self.target_actor(obs).detach()

    def optimize(self, iterations):
        for i in range(iterations):
            if len(self.replay_buffer.data) < args.min_buffersize:
                return 0, 0
            s1_arr, a_arr, r_arr, s2_arr, done_arr = self.replay_buffer.sample(args.batchsize)
            state = torch.cat(s1_arr).to(device)
            action = torch.cat(a_arr).to(device)
            reward = torch.Tensor(r_arr).reshape(args.batchsize, 1).to(device)
            next_state = torch.cat(s2_arr).to(device)
            # 0 means over
            done = torch.Tensor(done_arr).reshape(args.batchsize, 1).to(device)
            #------- update critic -----------
            new_action = self.get_target_action(next_state)
            y_target = reward + done * self.target_critic(next_state, new_action).detach() * args.Gamma
            y_pred = self.critic(state, action)
            critic_loss = F.mse_loss(y_pred, y_target)
            self.critic_optm.zero_grad()
            critic_loss.backward()
            self.critic_optm.step()
            #--------update actor---------------
            action_pred = self.actor(state)
            # - means smaller the loss, bigger the Q(s,a)
            actor_loss = -self.critic(state, action_pred).mean()
            #print(actor_loss)
            self.actor_optm.zero_grad()
            actor_loss.backward()
            self.actor_optm.step()
            #--------soft update target actor and critic----
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(args.tau * param.data + (1-args.tau)*target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1-args.tau)*target_param.data)

    def save_model(self, gamename, reward_list, seed):
        timenow = time.localtime(time.time())
        filename = "ddpg-"+gamename+"-seed-"+str(seed)+"-"
        torch.save(self.actor.state_dict(), os.path.join(args.model_path, filename+"-actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(args.model_path, filename+'-critic.pt'))
        try:
            record = pd.DataFrame(reward_list)
        except TypeError:
            print(reward_list)
            record = pd.DataFrame(reward_list)
        record.to_csv(os.path.join(args.reward_path, filename+'-reward.csv'))


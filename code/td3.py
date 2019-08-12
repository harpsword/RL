"""
Implementation of TD3 algo for MuJoCo

[64] Fujimoto S, Van Hoof H, Meger D, et al. Addressing Function Approximation Error in Actor-Critic Methods[J]. international conference on machine learning, 2018: 1582-1591.
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
from util.model2 import Policy, Value
from util.tools import soft_update

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = cpu_device


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
    start_timesteps = int(1e4)

    # storage path
    model_path = "../model/"
    reward_path = "../reward/"
    # noise for exploration action
    noise = 0.1
    # clamp limit for target policy smoothing
    c = 0.5
    sigma_clamp = 0.2
    # update policy frequency
    d = 2


class TD3Trainer(object):

    def __init__(self, state_dim, action_dim, action_lim, replay_buffer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.replay_buffer = replay_buffer
        self.actor = Policy(state_dim, action_dim, action_lim).to(device)
        self.critic1 = Value(state_dim, action_dim).to(device)
        self.critic2 = Value(state_dim, action_dim).to(device)

        self.actor_optm = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optm = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optm = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)

        self.target_actor = Policy(state_dim, action_dim, action_lim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.target_critic1 = Value(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = Value(state_dim, action_dim).to(device)
        self.target_critic2.load_state_dict(self.critic1.state_dict())

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
            epsilon = torch.randn_like(new_action) * args.sigma_clamp
            # target policy smoothing
            new_action += epsilon.clamp(-args.c, args.c)
            
            q1 = self.target_critic1(next_state, new_action).detach()
            q2 = self.target_critic2(next_state, new_action).detach()
            y_target = reward + args.Gamma * done * torch.min(q1, q2)
            
            y1 = self.critic1(state, action)
            critic1_loss = F.mse_loss(y1, y_target)
            self.critic1_optm.zero_grad()
            critic1_loss.backward()
            self.critic1_optm.step()
            
            y2 = self.critic2(state, action)
            critic2_loss = F.mse_loss(y2, y_target)
            self.critic2_optm.zero_grad()
            critic2_loss.backward()
            self.critic2_optm.step()
            #--------update actor---------------
            if i % args.d == 0:
                action_pred = self.actor(state)
                # - means smaller the loss, bigger the Q(s,a)
                actor_loss = -torch.Tensor.mean(self.critic1(state, action_pred))
                self.actor_optm.zero_grad()
                actor_loss.backward()
                self.actor_optm.step()
                #--------soft update target actor and critic----
                for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1-args.tau)*target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1-args.tau)*target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1-args.tau)*target_param.data)

    def save_model(self, gamename, reward_list, seed):
        timenow = time.localtime(time.time())
        filename = "td3-"+gamename+'-'+str(timenow.tm_year)+"-"+str(timenow.tm_mon)+"-"+str(timenow.tm_mday)+"-seed-"+str(seed)+"-"
        torch.save(self.actor.state_dict(), os.path.join(args.model_path, filename+"-actor.pt"))
        torch.save(self.critic1.state_dict(), os.path.join(args.model_path, filename+'-critic1.pt'))
        torch.save(self.critic2.state_dict(), os.path.join(args.model_path, filename+'-critic2.pt'))
        try:
            record = pd.DataFrame(reward_list)
        except TypeError:
            print(reward_list)
            record = pd.DataFrame(reward_list)
        record.to_csv(os.path.join(args.reward_path, filename+'-reward.csv'))

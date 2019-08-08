"""
using dataset and dataloader instead of np.random.choice
implementation of PPO algo for Atari Environment
Paper:[62] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms.[J]. arXiv: Learning, 2017.
"""

import os
import gym
import ray
import click
import torch
import time
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import autograd
from torch.utils.data import Dataset, DataLoader
from util.preprocess2015 import ProcessUnit
from util.model import Policy2013, Value
from util.tools import save_record

cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

storage_path = "../results"
EPS = 1e-10


class args(object):
    FrameSkip = 4
    Gamma = 0.99
    # GAE parameter
    Lambda = 0.95
    # Horizon(train frame)
    Llocal = 128
    # Num epochs
    K = 3
    # Tmax: 40M
    Tmax = int(40e6)
    actor_number = 8 
    generation = 1000000
    # adam's stepsize 2.5 * 10^-4
    stepsize = 2.5e-4
    stepsize0 = stepsize
    # Loss hyperparameter
    c1 = 1
    c2 = 0.01
    #minibatch_size = 32*8
    minibatch_size = 32*8
    # clip parameter
    epsilon = 0.1 
    epsilon0 = epsilon
    seed = 124
    layer_norm = True
    state_norm = True
    advantage_norm = True 
    lossvalue_norm = True

    @classmethod
    def update(cls, current_frames):
        ratio = 1 - current_frames / cls.Tmax
        cls.stepsize = cls.stepsize0 * ratio
        cls.epsilon = cls.epsilon0 * ratio


@ray.remote
class Simulator(object):
    """
    simulator can be used for training data collection and performance test.
    If you define a Simulator for training data collection, you should not use it for testing.
    """
    def __init__(self, gamename):
        self.env = gym.make(gamename)
        #self.env.seed(args.seed)
        self.obs = self.env.reset()
        self.start = False
        self.record = {
            "episode":[],
            "steps":[],
            "reward":[],
            "gamelength":[]
        }
        self.reward = 0
        self.gamelength = 0

    def start_game(self, episode, steps):
        no_op_frames = np.random.randint(1,30)
        self.pu = ProcessUnit(4, args.FrameSkip)
        obs = self.env.reset()
        self.pu.step(obs)
        for i in range(no_op_frames):
            obs, r, done, _ = self.env.step(0)
            self.pu.step(obs)
            if done:
                return False
        self.start = True
        self.record['episode'].append(episode)
        self.record['steps'].append(steps)
        self.record['reward'].append(self.reward)
        self.record['gamelength'].append(self.gamelength)
        self.reward = 0
        self.gamelength = 0
        return True

    def get_records(self):
        return self.record

    def add_record(self, r):
        self.reward += r
        self.gamelength += 1

    def rollout(self, actor, critic, Llocal, episode, steps):
        """
        if Llocal is None: test mission
        else: collect data
        """
        while not self.start:
            self.start_game(episode, steps)
        if Llocal is None:
            self.start_game(episode, steps)

        Lmax = 108000 if Llocal is None else Llocal
        frame_list = []
        action_list = []
        done_list = []
        reward_list = []
        # the probability of choosing action at
        probability_list = []
        break_or_not = False
        reward = 0
        for i in range(Lmax):
            frame_now = self.pu.to_torch_tensor()
            # stochastic policy
            action, prob = actor.act_with_prob(frame_now)
            r_ = 0
            for j in range(args.FrameSkip):
                obs, r, done, _ = self.env.step(action)
                r_ += r
                reward += r
                self.pu.step(obs)

                # it's for recording
                self.add_record(r)
                
                if done:
                    break_or_not = True
                    break
            if Llocal is not None:
                frame_list.append(frame_now)
                action_list.append(action)
                done_list.append(0 if done else 1)
                reward_list.append(r_)
                probability_list.append(prob)
            if break_or_not:
                self.start = False
                break
        if Llocal is None:
            # for testing models
            return reward
        # for collecting data
        frame_list = frame_list[::-1]
        action_list = action_list[::-1]
        reward_list = reward_list[::-1]
        probability_list = probability_list[::-1]
        done_list = done_list[::-1]
        # value's output
        value_list = []
        for idx, frame in enumerate(frame_list):
            value_list.append(critic(frame))

        delta_list = []
        advantage_list = []
        Value_target_list = []
        # previous discounted return
        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in range(len(reward_list)):
            Value_target_list.append(reward_list[i]+args.Gamma*prev_return*done_list[i])
            delta_list.append(reward_list[i]+args.Gamma*prev_value*done_list[i]-value_list[i])
            assert delta_list[i] == delta_list[-1]
            advantage_list.append(delta_list[i] + args.Gamma*args.Lambda*prev_advantage*done_list[i])

            prev_return = Value_target_list[i]
            prev_value = value_list[i]
            prev_advantage = advantage_list[i]

        # if args.advantage_norm:
        #     mb_advan = (mb_advan - mb_advan.mean()) / (mb_advan.std() + EPS)
        return [frame_list, action_list, probability_list, advantage_list, Value_target_list]


class RLDataset(Dataset):
    """
    dataset for RL data:

    1. state
    2. action
    3. action probability
    4. advantage estimator
    5. target value(critic)
    ...
    """
    def __init__(self, data_list):
        super(RLDataset, self).__init__()
        self.data_list = data_list
        # the element of data_list should be torch.Tensor
        self.length = self.data_list[0].shape[0]

    def __getitem__(self, i):
        return [d[i] for d in self.data_list]

    def __len__(self):
        return self.length

@click.command()
@click.option("--gamename")
def main(gamename):
    start_time = time.time()
    env = gym.make(gamename)
    action_n = env.action_space.n
    critic = Value().to(device)
    actor = Policy2013(action_n).to(device)
    simulators = [Simulator.remote(gamename) for i in range(args.actor_number)]

    actor_optm = torch.optim.Adam(actor.parameters(), lr=args.stepsize)
    critic_optm = torch.optim.Adam(critic.parameters(), lr=args.stepsize)

    frame_count = 0

    for g in range(args.generation):
        # train simulator and test simulator will not use the same variable
        rollout_ids = [s.rollout.remote(actor.to(cpu_device), critic.to(cpu_device), args.Llocal, g, frame_count) for s in simulators]
        frame_list = []
        action_list = []
        prob_list = []
        advantage_list = []
        value_list = []
        
        for rollout_id in rollout_ids:
            rollout = ray.get(rollout_id)
            frame_list.extend(rollout[0])
            action_list.extend(rollout[1])
            prob_list.extend(rollout[2])
            advantage_list.extend(rollout[3])
            value_list.extend(rollout[4])

        actor.to(device)
        critic.to(device)
        batch_size = len(advantage_list)
        frame_t = torch.cat(frame_list)
        action_t = torch.Tensor(action_list).long()
        prob_t = torch.Tensor(prob_list).float()
        advantage_t = torch.Tensor(advantage_list).float()
        critic_target = torch.Tensor(value_list).float()

        if args.advantage_norm:
            advantage_t = (advantage_t - advantage_t.mean())/(advantage_t.std() + EPS)

        dataset = RLDataset([frame_t, action_t, prob_t, advantage_t, critic_target])
        dataloader = DataLoader(dataset, batch_size=args.minibatch_size, shuffle=True, num_workers=4)
        for batch_idx in range(args.K):
            for data_l in dataloader:
                # mb means minibatch 
                mb_size = data_l[0].shape[0]
                mb_state = data_l[0].to(device)
                mb_action = data_l[1].to(device)
                mb_prob = data_l[2].to(device)
                mb_advan = data_l[3].to(device)
                mb_critic_target = data_l[4].to(device)

                with autograd.detect_anomaly():
                
                    mb_new_prob = actor.return_prob(mb_state).to(device)
                    mb_old_prob = mb_prob.reshape(mb_size, 1).mm(torch.ones(1, action_n).to(device))
                    # CLIP Loss
                    prob_div = mb_new_prob / mb_old_prob
                    mb_advan_square = mb_advan.reshape(mb_size, 1).mm(torch.ones(1, action_n).to(device))
                    CLIP_1 = prob_div * mb_advan_square
                    CLIP_2 = prob_div.clamp(1-args.epsilon, 1+args.epsilon) * mb_advan_square
                    # - is for nll_loss
                    loss_clip = - F.nll_loss(torch.Tensor.min(CLIP_1, CLIP_2), mb_action)
                    # entropy loss: -p*ln(p)
                    # +EPS for the existence of Nan in actor model in backpropagation
                    loss_entropy = - (torch.Tensor.log2(mb_new_prob+EPS) * mb_new_prob).sum() / mb_size
                    actor_loss = -(loss_clip+args.c2*loss_entropy)
                    
                    actor_optm.zero_grad()
                    actor_loss.backward()
                    actor_optm.step()

                # VF loss
                mb_value_predict = critic(mb_state).flatten()
                loss_value = torch.Tensor.mean((mb_critic_target-mb_value_predict).pow(2))
                critic_loss = loss_value
                with autograd.detect_anomaly():
                    critic_optm.zero_grad()
                    critic_loss.backward()
                    critic_optm.step()
                
                #print(loss_clip)
                #print(loss_value)
                #print(loss_entropy)

        frame_count += batch_size * args.FrameSkip
        args.update(frame_count)
        # update optim's learning rate
        for gg in actor_optm.param_groups:
            gg['lr'] = args.stepsize
        for gg in critic_optm.param_groups:
            gg['lr'] = args.stepsize
        if g % 10 == 0:
            print("Gen %s | progross percent:%.2f | time %.2f" % (g, frame_count/args.Tmax*100, time.time()-start_time))
            print("Gen %s | progross %s/%s | time %.2f" % (g, frame_count, args.Tmax, time.time()-start_time))

        if frame_count > args.Tmax:
            break

        if g % 10 == 0:
            records_id = [s.get_records.remote() for s in simulators]
            save_record(records_id, storage_path, 'ppo-record-%s.csv' % gamename)
            torch.save(actor.state_dict(), storage_path+"ppo_actor_"+gamename+'.pt')

    # after training
    records_id = [s.get_records.remote() for s in simulators]
    save_record(records_id, storage_path, 'ppo-record-%s.csv' % gamename)
    torch.save(actor.state_dict(), storage_path+"ppo_actor_"+gamename+'.pt')


if __name__ == "__main__":
    ray.init()
    main()



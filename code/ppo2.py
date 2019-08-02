"""
using dataset and dataloader instead of np.random.choice
implementation of PPO algo for Atari Environment
"""

import os
import gym
import ray
import click
import torch
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from util.preprocess2015 import ProcessUnit
from util.model import Policy2013, Value
from util.tools import save_record

gpu_id = torch.cuda.current_device()
gpu_device = torch.device(gpu_id)
cpu_device = torch.device('cpu')
device = gpu_device

storage_path = "../results"


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
        for i in range(Lmax+1):
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
                self.reward += r
                self.gamelength += 1
                
                if done:
                    break_or_not = True
                    break
            if Llocal is not None:
                frame_list.append(frame_now)
                action_list.append(action)
                done_list.append(done)
                reward_list.append(r_)
                probability_list.append(prob)
            if break_or_not:
                self.start = False
                break
        if Llocal is None:
            # for testing models
            return reward
        # for collecting data
        last_frame = frame_list[-1]
        frame_list = frame_list[::-1]
        action_list = action_list[::-1]
        reward_list = reward_list[::-1]
        probability_list = probability_list[::-1]

        critic_output_list = []
        for idx, frame in enumerate(frame_list):
            critic_output_list.append(critic(frame))

        delta_list = []
        advantage_list = []
        Value_target_list = []

        delta = 0
        R = 0
        for idx, frame in enumerate(frame_list):
            # idx = 0 is a deserted frame
            delta = reward_list[idx] + args.Gamma * critic_output_list[idx-1] - critic_output_list[idx]
            if idx == 0:
                if done_list[-1]:
                    R = 0
                else:
                    R = critic_output_list[idx]
            else:
                R = args.Gamma * R + reward_list[idx]
                Value_target_list.append(R)
            if idx == 1:
                delta_list.append(delta)
                advantage_list.append(delta_list[-1])
            elif idx > 1:
                delta_list.append(delta)
                advantage_list.append(advantage_list[-1]*args.Lambda*args.Gamma)
        return [frame_list[1:], action_list[1:], probability_list[1:], advantage_list, Value_target_list]


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
        prob_list =  []
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

        dataset = RLDataset([frame_t, action_t, prob_t, advantage_t, critic_target])
        dataloader = DataLoader(dataset, batch_size=args.minibatch_size, shuffle=True, num_workers=4)
        for batch_idx in range(args.K):
            for data_l in dataloader:
                # mb means minibatch 
                mb_state = data_l[0].to(device)
                mb_action = data_l[1].to(device)
                mb_prob = data_l[2].to(device)
                mb_advan = data_l[3].to(device)
                mb_critic_target = data_l[4].to(device)

                mb_new_prob = actor.return_prob(mb_state, mb_action).to(device)
                
                # CLIP Loss
                prob_div = mb_new_prob / mb_prob
                CLIP_1 = prob_div * mb_advan
                CLIP_2 = prob_div.clamp(1-args.epsilon, 1+args.epsilon) * mb_advan
                loss_clip = torch.Tensor.mean(torch.Tensor.min(CLIP_1, CLIP_2))
                # VF loss
                mb_value_predict = critic(mb_state).flatten()
                loss_value = torch.Tensor.mean((mb_critic_target-mb_value_predict).pow(2))
                # entropy loss
                # TODO: error in here
                loss_entropy = torch.Tensor.mean(torch.Tensor.log(mb_new_prob)*mb_new_prob)
                loss = loss_clip + args.c1*loss_value + loss_entropy*args.c2
                actor_optm.zero_grad()
                critic_optm.zero_grad()
                loss.backward()
                actor_optm.step()
                critic_optm.step()

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



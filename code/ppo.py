"""
implementation of PPO algo for Atari Environment
"""

import gym
import ray
import click
import torch
import time
import numpy as np
import torch.nn.functional as F

from preprocess2015 import ProcessUnit
from model import Policy2013, Value

#gpu_id = torch.cuda.current_device()
#gpu_device = torch.device(gpu_id)
cpu_device = torch.device('cpu')
device = cpu_device


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
    # Loss hyperparameter
    c1 = 1
    c2 = 0.01
    #minibatch_size = 32*8
    minibatch_size = 32 * 8
    # clip parameter
    epsilon = 0.1 


@ray.remote
class Simulator(object):

    def __init__(self, gamename):
        self.env = gym.make(gamename)
        self.obs = self.env.reset()
        self.start = False

    def start_game(self):
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
        return True

    def rollout(self, actor, critic, Llocal):
        """
        if Llocal is None: test mission
        else: collect data
        """
        while not self.start:
            self.start_game()
        if Llocal is None:
            self.start_game()

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
        rollout_ids = [s.rollout.remote(actor.to(cpu_device), critic.to(cpu_device), args.Llocal) for s in simulators]
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
        frame_t = torch.cat(frame_list).to(device)
        action_t = torch.Tensor(action_list).to(device).long()
        prob_t = torch.Tensor(prob_list).to(device).float()
        advantage_t = torch.Tensor(advantage_list).to(device).float()
        critic_target = torch.Tensor(value_list).to(device).float()

        for batch_id in range(args.K):
            minibatch_index = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_state = frame_t[minibatch_index]
            minibatch_action = action_t[minibatch_index]
            minibatch_prob = prob_t[minibatch_index]
            minibatch_advan = advantage_t[minibatch_index]
            minibatch_critic_target = critic_target[minibatch_index]
            minibatch_new_prob = actor.return_prob(minibatch_state, minibatch_action)

            # CLIP Loss
            prob_div = minibatch_new_prob / minibatch_prob
            CLIP_1 = prob_div * minibatch_advan
            CLIP_2 = prob_div.clamp(1-args.epsilon, 1+args.epsilon) * minibatch_advan
            loss_clip = torch.Tensor.mean(torch.Tensor.min(CLIP_1, CLIP_2))
            # VF loss
            minibatch_value_predict = critic(minibatch_state).flatten()
            loss_value = torch.Tensor.mean((minibatch_critic_target-minibatch_value_predict).pow(2))
            # entropy loss
            # TODO: error in here
            loss_entropy = torch.Tensor.mean(torch.Tensor.log(minibatch_new_prob)*minibatch_new_prob)
            loss = loss_clip + loss_value + loss_entropy
            actor_optm.zero_grad()
            critic_optm.zero_grad()
            loss.backward()
            actor_optm.step()
            critic_optm.step()

        frame_count += batch_size * args.FrameSkip

        if g % 1 == 0:
            print("Gen %s | progross percent:%.2f | time %.2f" % (g, frame_count/args.Tmax*100, time.time()-start_time))

        if frame_count > args.Tmax:
            break

if __name__ == "__main__":
    ray.init()
    main()



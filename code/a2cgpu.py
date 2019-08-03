"""
implementation of A2C algo for Atari Environment
"""
import gym
import ray
import click
import torch
import time
import numpy as np
import torch.nn.functional as F

from util.preprocess2015 import ProcessUnit
from util.model import Policy2013, Value
from util.tools import save_record


storage_path = "../results"

Gamma = 0.99
Llocal = 32
Tmax = 320e6 
FrameSkip = 4

actor_number = 16
generation = 1000000

actor_lr = 1e-4
critic_lr = 1e-4

gpu_id = torch.cuda.current_device()
gpu_device = torch.device(gpu_id)
cpu_device = torch.device('cpu')
device = gpu_device

@ray.remote
class Simulator(object):

    def __init__(self, gamename):
        self.env = gym.make(gamename)
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

    def start_game(self):
        no_op_frames = np.random.randint(1,30)
        self.pu = ProcessUnit(4, FrameSkip)
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
        break_or_not = False
        reward = 0
        for i in range(Lmax):
            frame_now = self.pu.to_torch_tensor()
            # stochastic policy
            action = actor.act(frame_now)
            r_ = 0
            for j in range(FrameSkip):
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
                done_list.append(done)
                reward_list.append(r_)

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
        R_list = []
        for idx, frame in enumerate(frame_list):
            if idx == 0:
                if done_list[-1]:
                    R_list.append(0)
                else:
                    R_list.append(critic(frame))
            else:
                R_list.append(Gamma*R_list[-1]+reward_list[idx])
        return [frame_list, action_list, R_list] 


@click.command()
@click.option("--gamename")
def main(gamename):
    start_time = time.time()
    env = gym.make(gamename)
    action_n = env.action_space.n
    critic = Value().to(device)
    actor = Policy2013(action_n).to(device)
    simulators = [Simulator.remote(gamename) for i in range(actor_number)]

    actor_optm = torch.optim.RMSprop(actor.parameters(), lr=actor_lr)
    critic_optm = torch.optim.RMSprop(critic.parameters(), lr=critic_lr)

    frame_count = 0

    for g in range(generation):
        rollout_ids = [s.rollout.remote(actor.to(cpu_device), critic.to(cpu_device), Llocal) for s in simulators]
        frame_list = []
        action_list = []
        R_list = []
        
        for rollout_id in rollout_ids:
            rollout = ray.get(rollout_id)
            frame_list.extend(rollout[0])
            action_list.extend(rollout[1])
            R_list.extend(rollout[2])

        # after collecting training data,
        # TODO: we need to build dataset and dataloader
        batch_size = len(R_list)
        input_state = torch.cat(frame_list).to(device)
        actor_target = torch.tensor(action_list).to(device).long()
        critic_target = torch.tensor(R_list).reshape(batch_size, 1).to(device).float()

        actor.to(device)
        critic.to(device)

        critic_optm.zero_grad()
        critic_predict = critic(input_state)
        critic_loss = F.mse_loss(critic_predict, critic_target)
        critic_loss.backward()
        critic_optm.step()

        actor_optm.zero_grad()
        actor_predict = actor(input_state)
        rescale_loss = (critic_target.detach()-critic_predict.detach()).reshape(batch_size, 1).mm(torch.ones(1, action_n).to(device))
        # shape : (batch_size, action_n)
        actor_loss_tmp = F.log_softmax(actor_predict, dim=1) * rescale_loss
        actor_loss = F.nll_loss(actor_loss_tmp, actor_target)
        actor_loss.backward()
        actor_optm.step()

        frame_count += batch_size * FrameSkip

        if g % 1 == 0:
            print("Gen %s | progross %s/%s | time %.2f" % (g, frame_count, Tmax, time.time()-start_time))

        if frame_count > Tmax:
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



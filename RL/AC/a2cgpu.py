
import gym
import ray
import click
import torch
import time
import numpy as np
import torch.nn.functional as F

from ..common.preprocess2015 import ProcessUnit
from ..common.model import Policy2013, Value

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

    def rollout(self, actor, critic, Llocal):
        """
        if Llocal is None: test mission
        else: collect data

        """
        Lmax = 108000 if Llocal is None else Llocal
        no_op_frames = np.random.randint(1, 30)
        pu = ProcessUnit(4, 4)
        obs = self.env.reset()
        pu.step(obs)
        reward = 0
        for i in range(no_op_frames):
            obs, r, done, _ = self.env.step(0)
            pu.step(obs)
        frame_list = []
        action_list = []
        done_list = []
        reward_list = []
        break_or_not = False
        for i in range(Lmax):
            frame_now = pu.to_torch_tensor()
            # stochastic policy
            action = actor.act(frame_now)
            r_ = 0
            for j in range(FrameSkip):
                obs, r, done, _ = self.env.step(action)
                r_ += r
                reward += r
                pu.step(obs)
                if done:
                    break_or_not = True
                    break
            if Llocal is not None:
                frame_list.append(frame_now)
                action_list.append(action)
                done_list.append(done)
                reward_list.append(r_)

            if break_or_not:
                break
        if Llocal is None:
            # for test model
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
                R_list.append(Gamma*R_list[-1]+reward_list[i])
        return [frame_list, action_list, R_list] 


@click.option("--gamename")
def main(gamename):
    start_time = time.time()
    env = gym.make(gamename)
    action_n = env.action_space.n
    batch_size = Llocal * actor_number
    critic = Value().to(device)
    actor = Policy2013(action_n).to(device)
    simulators = [Simulator.remote(gamename) for i in range(actor_number)]

    actor_optm = torch.optim.RMSprop(actor.parameters(), lr=actor_lr)
    critic_optm = torch.optim.RMSprop(critic.parameters(), lr=critic_lr)

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
        actor_loss_tmp = F.log_softmax(actor_predict, dim=1) * rescale_loss
        actor_loss = F.nll_loss(actor_loss_tmp, actor_target)
        actor_loss.backward()
        actor_optm.step()

        if g % 1 == 0:
            print("Gen %s | progross %s/%s | time %.2f" % (g, batch_size*g*FrameSkip, Tmax, time.time()-start_time))

        if batch_size*g*FrameSkip > Tmax:
            break

if __name__ == "__main__":
    main()








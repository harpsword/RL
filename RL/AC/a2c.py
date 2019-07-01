
import gym
import ray
import click
import torch
import numpy as np

from ..common.preprocess2015 import ProcessUnit
from ..common.model import Policy2013, Value

Gamma = 0.99
Llocal = 32
Tmax = 1e8
FrameSkip = 4

actor_number = 16
generation = 1000

ray.init()


@ray.remote
class Simulator(object):

    def __init__(self, gamename):
        self.env = gym.make(gamename)

    def rollout(self, actor, critic, Llocal):
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
            action = actor(frame_now).argmax().item()
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
    env = gym.make(gamename)
    action_n = env.action_space.n
    critic = Value()
    actor = Policy2013(action_n)
    simulators = [Simulator.remote(gamename) for i in range(actor_number)]

    actor_optm = torch.optim.RMSprop(actor.parameters(), lr=1e-4)
    critic_optm = torch.optim.RMSprop(critic.parameters(), lr=1e-4)

    for g in range(generation):
        rollout_ids = [s.rollout.remote(actor, critic, Llocal) for s in simulators]
        print(rollout_ids)

        exit()

if __name__ == "__main__":
    main()








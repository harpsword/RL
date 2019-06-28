
import os
import gym
import click
import torch
import pandas as pd
import numpy as np

from preprocess2015 import ProcessUnit
from torch import multiprocessing as mp
from policy2015 import Q_Net 
model_storage_path = '/home2/yyl/model/es-rl/'
#normal_model_storage_path = '/home/yyl/model/es-rl/'
import time
FRAME_SKIP = 4

def get_reward_from_gamename(model, gamename):
    env = gym.make(gamename)
    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ep_r = 0.
    frame_count = 0
    ProcessU = ProcessUnit(4, FRAME_SKIP)
    ProcessU.step(observation)
    ep_max_step = 0
       
    if test == True:
        ep_max_step = 108000
    #no_op_frames = np.random.randint(FRAME_SKIP+1, 30)
    no_op_frames = np.random.randint(0, 30)
    for i in range(no_op_frames):
        # TODO: I think 0 is Null Action
        # but have not found any article about the meaning of every actions
        observation, reward, done, _ = env.step(0)
        ProcessU.step(observation)
        frame_count += 1
        
    for step in range(ep_max_step):
        action = model(ProcessU.to_torch_tensor())[0].argmax().item()
        for i in range(FRAME_SKIP):
            observation, reward , done, _ = env.step(action)
            ProcessU.step(observation)
            frame_count += 1
            ep_r += reward
            if done:
                break_is_true = True
        if break_is_true:
            break
    return ep_r, frame_count


def get_reward(model, env):
    env.frameskip = 1
    observation = env.reset()
    break_is_true = False
    ep_r = 0.
    frame_count = 0
    if ep_max_step is None:
        raise TypeError("test")
    else:
        ProcessU = ProcessUnit(FRAME_SKIP)
        ProcessU.step(observation)
        
        if test == True:
            ep_max_step = 108000
        #no_op_frames = np.random.randint(FRAME_SKIP+1, 30)
        no_op_frames = np.random.randint(0, 30)
        for i in range(no_op_frames):
            # TODO: I think 0 is Null Action
            # but have not found any article about the meaning of every actions
            observation, reward, done, _ = env.step(0)
            ProcessU.step(observation)
            frame_count += 1

        for step in range(ep_max_step):
            action = model(ProcessU.to_torch_tensor())[0].argmax().item()
            for i in range(FRAME_SKIP):
                observation, reward , done, _ = env.step(action)
                ProcessU.step(observation)
                frame_count += 1
                ep_r += reward
                if done:
                    break_is_true = True
            if break_is_true:
                break
    return ep_r, frame_count


def test(model, pool, env, test_times):
    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (model, env)) for i in range(test_times)]
    # N_KID means episodes_per_batch
    rewards = []
    timesteps = []
    timesteps_count = 0
    for idx, j in enumerate(jobs):
        rewards.append(j.get()[0])
        timesteps.append(j.get()[1])
        timesteps_count += j.get()[1]
    
    return rewards, timesteps_count


@click.command()
@click.option("--model", type=str, help='the name of policy model')
@click.option("--gamename", type=str, help='the name of test game')
@click.option("--ncpu", type=int, help="the number of cpu")
def main(model, gamename, ncpu):
    pool = mp.Pool(processes=ncpu)
    env = gym.make(gamename)
    test_model = Q_Net(env.action_space.n)

    test_times = 100
    
    test_model.load_state_dict(torch.load(os.path.join(model_storage_path, model)))

    test_rewards, _ = test(test_model, pool, env, test_times)

    print("test results:", np.array(test_rewards).mean())
    print(str(test_rewards))

if __name__ == '__main__':
    main()

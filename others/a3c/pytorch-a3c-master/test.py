import time
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic, save_model
from preprocess2015 import ProcessUnit
from train import start_game

def test_one(env, pu, model, args):
    pu.clear()
    p = False
    while not p:
        p = start_game(env, pu)
    reward = 0
    episode_length = 0
    while True:

        state = pu.to_torch_tensor()
        with torch.no_grad():
            _, logit = model(state)
        prob = F.softmax(logit, dim=1)[0]
        action = np.argmax(prob).item()

        for i in range(args.frameskip):
            episode_length += 1
            obs, r, done, _ = env.step(action)
            pu.step(obs)
            reward += r
            done = done or episode_length >= args.max_episode_length
            if done:
                return reward, episode_length

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    pu = ProcessUnit(4, args.frameskip)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()
    start_time = time.time()

    while True:
        model.load_state_dict(shared_model.state_dict())

        rewards_list = []
        episode_length_list = []
        for i in range(30):
            reward, episode_length = test_one(env, pu, model, args)
            rewards_list.append(reward)
            episode_length_list.append(episode_length)

        print("Time {}, num steps {}, FPS {:.0f}, episode reward {.2f}, episode length {.2f}".format(
            time.strftime("%Hh %Mm %Ss",
                          time.gmtime(time.time() - start_time)),
            counter.value, counter.value / (time.time() - start_time),
            np.mean(rewards_list), np.mean(episode_length_list)))
        time.sleep(10)


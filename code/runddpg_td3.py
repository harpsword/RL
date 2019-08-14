
import os
import torch
import time
import click
import gym
import ray
import numpy as np

from util.tools import check_env
from util.data import Data
from ddpg import DDPGTrainer
from td3 import TD3Trainer


class args(object):
    # 1 Million
    Tmax = int(1e6)
    T = 2000
    max_episode = int(1e6) 
    start_timesteps = int(1e4)
    eval_freq = int(5e3)
    save_freq = int(5e4)

    buffersize = int(1e6)
    min_buffersize = 200 

    # storage path
    model_path = "../model/"
    reward_path = "../reward/"


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    state_dim = env.observation_space.shape[0]
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs = torch.from_numpy(obs).reshape(1, state_dim).float()
            action = policy.get_exploitation_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

        avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward)) 
    print("---------------------------------------") 
    return avg_reward


@ray.remote(num_gpus=0.1)
def main(gamename, seed, algo, task_id):
    env = gym.make(gamename)
    print("start running task", task_id)
    t0 = time.time()
    # set seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim, action_dim, action_lim = check_env(env)
    replay_buffer = Data(args.buffersize)
    if algo == 'ddpg':
        trainer = DDPGTrainer(state_dim, action_dim, action_lim, replay_buffer)
    elif algo == 'td3':
        trainer = TD3Trainer(state_dim, action_dim, action_lim, replay_buffer)
    else:
        print("error algo")
        return 

    frame_count = 0
    timestep_since_eval = 0
    timestep_since_save = 0
    reward_list = []
    evaluations = []
    for episode in range(args.max_episode):
        trainer.init_episode()
        obs = env.reset()
        obs = torch.from_numpy(obs).reshape(1, state_dim).float()
        reward_episode = 0
        actor_loss_l = []
        critic_loss_l = []
        for i in range(args.T):
            if frame_count < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = trainer.get_exploration_action(obs)
            new_obs, r, done, _ = env.step(action)
            new_obs = torch.from_numpy(new_obs).reshape(1, state_dim).float()
            reward_episode += r
            sequence  = [obs, torch.from_numpy(action).reshape(1, action_dim).float(), reward_episode, new_obs, 0 if done else 1]

            replay_buffer.push(sequence)
            obs = new_obs
            if done:
                break
        trainer.optimize(i)

        frame_count += i
        timestep_since_eval += i
        timestep_since_save += i
        reward_list.append(reward_episode)

        if timestep_since_eval > args.eval_freq:
            timestep_since_eval %= args.eval_freq 
            evaluations.append(evaluate_policy(env, trainer))

        if timestep_since_save > args.save_freq:
            trainer.save_model(gamename, evaluations, seed)
            timestep_since_save %= args.save_freq

        if episode % 50 == 0:
            #print(np.array(critic_loss).mean())
            #print(np.array(actor_loss).mean())
            #print("episode:%s, reward:%.2f, %s/%s, time:%s" % (episode, reward_episode, frame_count, args.Tmax, time.time()-t0))
            pass
        if frame_count > args.Tmax:
            break
    trainer.save_model(gamename, evaluations, seed)
    return "over"


@click.command()
@click.option("--algo", type=click.Choice(['ddpg', 'td3']))
def run_all(algo):
    envs_list = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
                 'HumanoidStandup-v2','InvertedDoublePendulum-v2',
                 'Swimmer-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'Reacher-v2']
    env_list = [gym.make(env_name) for env_name in envs_list]
    seed_list = [np.random.randint(0, 1000000) for i in range(5)]
    job_list = []

    idx = 0
    for seed in seed_list:
        for env in envs_list:
            job_list.append(main.remote(env, seed, algo, idx))
            idx += 1
    for idx, value_id in enumerate(job_list):
        print("task", idx, ray.get(value_id))


if __name__ == '__main__':
    # object_store_memory: 80G
    # redis_max_memory: 40G
    ray.init(num_cpus=20, num_gpus=2, object_store_memory=85899345920, redis_max_memory=42949672960)
    run_all()

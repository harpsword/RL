import argparse
import os

import gym
import numpy as np
import ray
import torch
import click

import DDPG
import OurDDPG
import TD3
import utils


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------
------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


@ray.remote(num_gpus=0.1, max_calls=1)
def main(env_name, seed, algo, idx):
    # algo: TD3, DDPG, OurDDPG
    # seed: int
    # env_name: str

    class args:
        policy_name = "algo"
        env_name = "env_name"
        seed = 0
        start_timesteps = int(1e4)
        eval_freq = int(5e3)
        max_timesteps = int(1e6)
        save_models = True
        expl_noise = 0.1
        batch_size = 100
        discount = 0.99
        tau = 0.005
        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2

    args.policy_name = algo
    args.env_name = env_name
    args.seed = seed

    file_name = "%s-%s-seed-%s--reward.csv" % (
        args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(
            state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" %
                      (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount,
                                 args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps,
                                 args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, policy))

                if args.save_models:
                    policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise,
                                                    size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + \
            1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, policy))
    if args.save_models:
        policy.save(
            "%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)
    return True


# @click.command()
# @click.option("--algo", type=click.Choice(['TD3', 'DDPG', 'OurDDPG']))
def run_all():
    envs_list = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
                 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2',
                 'Swimmer-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'Reacher-v2']
    seed_list = [np.random.randint(0, 1000000) for i in range(5)]
    algo_list = ["TD3", "DDPG", "OurDDPG"]
    job_list = []

    idx = 0
    for algo in algo_list:
        for seed in seed_list:
            for env in envs_list:
                job_list.append(main.remote(env, seed, algo, idx))
                idx += 1
    for idx, value_id in enumerate(job_list):
        print("task", idx, ray.get(value_id))


if __name__ == '__main__':
    # object_store_memory: 80G
    # redis_max_memory: 40G
    ray.init(num_cpus=20, num_gpus=2, object_store_memory=85899345920,
             redis_max_memory=42949672960)
    run_all()

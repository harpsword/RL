"""
Implementation of DDPG algo for MuJoCo
Paper: [63] Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.
"""

import os
import time
import click
import torch
import gym
import random
import pandas as pd
import torch.nn.functional as F

from data import Data
from util.model2 import Policy, Value
from util.tools import soft_update

class OrnsteinUhlenbeckActionNoise:
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    ref: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


class args(object):
    # 1 Million
    Tmax = int(40e6)
    T = 2000
    max_episode = 10000
    Gamma = 0.99

    tau = 0.001
    # optimizer : Adam
    actor_lr = 1e-4
    critic_lr = 1e-3

    batchsize = 64
    buffersize = int(1e6)
    min_buffersize = int(1e3)

    # Exploration Noise:Ornstein-Uhlenbeck process
    theta = 0.15
    sigma = 0.2
    # storage path
    model_path = "../model/"
    reward_path = "../reward/"


class DDPGTrainer(object):

    def __init__(self, state_dim, action_dim, action_lim, replay_buffer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.replay_buffer = replay_buffer
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim, theta=args.theta, sigma=args.sigma)
        self.actor = Policy(state_dim, action_dim, action_lim)
        self.critic = Value(state_dim, action_dim)
        self.actor_optm = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optm = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.target_actor = Policy(state_dim, action_dim, action_lim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Value(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def init_episode(self):
        self.noise.reset()

    def get_action(self, obs):
        # with noise
        action = self.actor(obs).detach() + self.noise.sample()
        return action.clamp(-self.action_lim, self.action_lim)

    def get_target_action(self, obs):
        return action = self.target_actor(obs).detach()

    def optimize(self):
        if len(self.replay_buffer.data) < args.min_buffersize:
            return
        s1_arr, a_arr, r_arr, s2_arr, done_arr = self.replay_buffer.sample(args.batchsize)
        state = torch.cat(s1_arr)
        action = torch.cat(a_arr)
        reward = torch.Tensor(r_arr)
        next_state = torch.cat(s2_arr)
        # 0 means done
        done = torch.Tensor(done_arr)
        #------- update critic -----------
        new_action = self.get_target_action(new_action)
        y_target = reward + done * self.target_critic(next_state, new_action).detach()
        y_pred = self.critic(state)
        critic_loss = F.mse_loss(y_pred, y_target)
        self.critic_optm.zero_grad()
        critic_loss.backward()
        self.critic_optm.step()
        #--------update actor---------------
        action_pred = self.get_action(state)
        # - means smaller the loss, bigger the Q(s,a)
        actor_loss = -torch.Tensor.mean(self.critic(state, action_pred))
        self.actor_optm.zero_grad()
        actor_loss.backward()
        self.actor_optm.step()
        #--------soft update target actor and critic----
        soft_update(self.critic, self.target_critic)
        soft_update(self.actor, self.target_actor)

    def save_model(self, gamename, reward_list):
        timenow = time.localtime(time.time())
        filename = "ddpg-"+gamename+'-'+str(timenow.tm_year)+"-"+str(timenow.tm_mon)+"-"+str(timenow.tm_day)
        torch.save(self.actor.state_dict(), os.path.join(args.model_path, filename+"-actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(args.model_path, filename+'-critic.pt'))
        record = pd.DataFrame(reward_list)
        record.to_csv(os.path.join(args.reward_path, filename+'-reward.csv'))


def check_env(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low
    assert len(action_low) == len(action_high)
    for i in range(len(action_low)):
        if abs(action_low[i]) != abs(action_high[i]):
            raise ValueError("Environment Error with wrong action low and high")
    return state_dim, action_dim, action_high[0]


@click.command()
@click.option("--gamename")
def main(gamename):
    env = gym.make(gamename)
    state_dim, action_dim, action_lim = check_env(env)
    replay_buffer = Data(args.buffersize)
    trainer = DDPGTrainer(state_dim, action_dim, action_lim, replay_buffer)

    frame_count = 0
    reward_list = []
    for episode in range(args.max_episode):
        trainer.init_episode()
        obs = env.reset()
        obs = torch.from_numpy(obs)
        reward_episode = 0
        for i in range(args.T):
            action = trainer.get_action(obs) 
            new_obs, r, done, _ = env.step(action)
            new_obs = torch.from_numpy(new_obs)
            reward_episode += r
            sequence  = [obs, action, reward_episode, new_obs, 0 if done else 1]
            replayer_buffer.put(sequence)
            obs = new_obs

            trainer.optimize()

            if done:
                break
        frame_count += i
        reward_list.append(reward_episode)
        if frame_count > args.Tmax:
            break
        if episode % 10 == 0:
            trainer.save_model(gamename, reward_list)

if __name__ == "__main__":
    main()



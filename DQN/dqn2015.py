# paper
# ref: Mnih et al. 2015. Human-level control through deep reinforcement learning
# add 30 no-op to trainning
import os
import time
import json
import click
import gym
import logging
import torch
import numpy as np
import torch.nn.functional as F

from policy2015 import Q_Net, ProcessUnit
from data import Data

LogFolder = os.path.join(os.getcwd(), 'log')
FRAME_SKIP = 4
GAMMA = 0.99
init_epsilon = 0.5
decay_every_timestep = 100000
epsilon_decay = 0.5
final_epsilon = 0.1

# training
target_network_update_frequency = 10000
batchsize = 32
lr = 0.00025
replay_start_size = 50000

# experience replay storage
D = Data()

def train(cfg, model, env):
    action_n = env.action_space.n
    env.frameskip = 1
    epsilon = init_epsilon 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    for episode in range(1, cfg['game']['episode']):
        t0 = time.time()
        obs = env.reset()
        pu = ProcessUnit(FRAME_SKIP)
        pu.step(obs)

        break_is_true = False
        reward_one_episode = 0

        no_op_frames = np.random.randint(FRAME_SKIP+1, 30)
        for i in range(no_op_frames):
            obs, reward, done, _ = env.step(0)
            pu.step(obs)
            reward_one_episode += reward
        
        previous_frame_list = pu.to_torch_tensor()
        for step in range(cfg['game']['timesteplimit']):
            if step % 100 == 0:
                print("step:", step, "time:", time.time()-t0)

            if np.random.rand() <= epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = model(previous_frame_list).argmax().item()
            ep_r = 0
            for i in range(FRAME_SKIP):
                obs, reward, done, _ = env.step(action)
                pu.step(obs)
                ep_r += reward
                if done:
                    break_is_true = True

            frame_list_now = pu.to_torch_tensor()
            sequence = [previous_frame_list, action, ep_r, frame_list_now, done]
            # make sure one calculation for one frame list
            previous_frame_list = frame_list_now
            reward_one_episode += ep_r
            D.push(sequence)

            # sample data
            # train model
            if len(D.data) >= replay_start_size:
                import random
                selected_data = random.sample(D.data, batchsize)
                state_batch = [batch[0] for batch in selected_data]
                target_q_value = None
                for i in range(batchsize):
                    state_ = selected_data[i][0]
                    action_ = selected_data[i][1]
                    reward_ = selected_data[i][2]
                    next_state_ = selected_data[i][3]
                    done_ = selected_data[i][4]
                    q_eval = model(state_)
                    if target_q_value is None:
                        target_q_value = q_eval
                    else:
                        target_q_value = torch.cat((target_q_value, q_eval))
                    # only one element is different to q_eval
                    if done_:
                        target_q_value[-1][action_] = reward_
                    else:
                        target_q_value[-1][action_] = reward_ + GAMMA*model(next_state_).max().item()
                x_train = torch.cat(state_batch)
                y_train = target_q_value
                # dataset = Data.TensorDataset(state_batch, target_q_value)
                optimizer.zero_grad()
                predicted_q_value = model(x_train)
                loss = F.mse_loss(predicted_q_value, y_train)
                loss.backward()
                optimizer.step()

            if break_is_true:
                break
        print("Episode:", episode, '|Reward:', reward_one_episode)


def setup_logging(logfile):
    if logfile == 'default.log':
        timenow = time.localtime(time.time())
        logfile = str(timenow.tm_year)+'-'+str(timenow.tm_mon)+'-'+str(timenow.tm_mday)
        indx = 1
        while logfile+'-'+str(indx)+'.log' in os.listdir(LogFolder):
            indx += 1
        logpath = os.path.join(LogFolder, logfile+'-'+str(indx)+'.log')
    else:
        logpath = os.path.join(LogFolder, logfile)
    logging.basicConfig(filename=logpath,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


@click.command()
@click.option('--expfile')
@click.option('--logfile', default='default.log', help='the name of log file')
def main(expfile, logfile):
    if expfile:
        with open(expfile, 'r') as f:
            cfg = json.loads(f.read())
    setup_logging(logfile)
    env = gym.make(cfg['game']['gamename'])
    model = Q_Net(env.action_space.n)
    train(cfg, model, env)


if __name__ == '__main__':
    main()

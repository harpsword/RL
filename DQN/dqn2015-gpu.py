# paper
# ref: Mnih et al. 2015. Human-level control through deep reinforcement learning
# add 30 no-op to trainning
# note: no train()/eval() 
import os
import time
import json
import click
import gym
import logging
import torch
import numpy as np
import torch.nn.functional as F
from data import Data
from test_model import get_reward_from_gamename

from policy2015 import Q_Net
from preprocess2015 import ProcessUnit

LogFolder = os.path.join(os.getcwd(), 'log')
FRAME_SKIP = 4
GAMMA = 0.99
init_epsilon = 1
final_epsilon = 0.1
final_exploration_frame = int(1e6)
no_op_max = 30
max_frames_one_episode = 18000
test_every_episode = 50 
# training
target_network_update_frequency = 10000
batchsize = 32
update_frequency = 4
generation = 100000
frame_max = int(1e7)
# optimizer
lr = 0.00025
momentum = 0.95
squared_gradient_momentum = 0.95
min_squaured_gradient = 0.01
# buffer
replay_start_size = 50000
# the number of processes in Pool
ncpu = 10


gpu_id = torch.cuda.current_device()
print("using GPU %s" % gpu_id)
device = torch.device(gpu_id)
cpu_device = torch.device('cpu')
#device = cpu_device
model_storage_path = '/home2/yyl/model/es-rl/'


def test(model, gamename):
    env = gym.make(gamename)
    no_op_frames = np.random.randint(FRAME_SKIP+1, no_op_max)
    pu = ProcessUnit(4, FRAME_SKIP)
    obs = env.reset()
    pu.step(obs)
    reward_one_episode = 0
    for i in range(no_op_frames):
        obs, reward, done, _ = env.step(0)
        pu.step(obs)
        reward_one_episode += reward
    break_or_not = False
    for i in range(max_frames_one_episode):
        action = model(pu.to_torch_tensor().to(device)).argmax().item()
        for j in range(FRAME_SKIP):
            obs, reward, done, _ = env.step(action)
            pu.step(obs)
            if done:
                break_or_not = True
                break
            reward_one_episode += reward
        if break_or_not:
            break
    return reward_one_episode


def train(model, target_model, env, gamename):
    D = Data()
    update_times = 0
    frame_count = 0
    action_n = env.action_space.n
    env.frameskip = 1
    epsilon = init_epsilon 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                    momentum=momentum, eps=min_squaured_gradient)
    for episode in range(1, generation):
        t0 = time.time()
        obs = env.reset()
        pu = ProcessUnit(4, FRAME_SKIP)
        pu.step(obs)

        break_is_true = False
        reward_one_episode = 0

        no_op_frames = np.random.randint(FRAME_SKIP+1, no_op_max)
        for i in range(no_op_frames):
            obs, reward, done, _ = env.step(0)
            pu.step(obs)
            reward_one_episode += reward
        
        previous_frame_list = pu.to_torch_tensor()
        for step in range(max_frames_one_episode):
            if np.random.rand() <= epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = model(previous_frame_list.to(device)).argmax().item()
            ep_r = 0
            for i in range(FRAME_SKIP):
                obs, reward, done, _ = env.step(action)
                frame_count += 1
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

            if step % update_frequency != (update_frequency - 1):
                continue
            # sample data, train model
            if len(D.data) >= replay_start_size:
                import random
                selected_data = random.sample(D.data, batchsize)
                target_q_value = None
                q_target_list = [0] * batchsize
                for i in range(batchsize):
                    state_ = selected_data[i][0]
                    action_ = selected_data[i][1]
                    reward_ = selected_data[i][2]
                    next_state_ = selected_data[i][3]
                    done_ = selected_data[i][4]
                    # only one element is different to q_eval
                    if done_:
                        q_target_list[i] = reward_
                    else:
                        q_target_list[i] = reward_ + GAMMA * target_model(next_state_.to(device)).max().item()
                state_batch = [batch[0] for batch in selected_data]
                x_train = torch.cat(state_batch)
                optimizer.zero_grad()
                predicted_q_value = model(x_train.to(device))
                # build y_train
                action_batch = [batch[1] for batch in selected_data]
                y_train = predicted_q_value.clone()
                for i in range(batchsize):
                    y_train[action_batch[i]] = q_target_list[i]
                loss = F.mse_loss(predicted_q_value, y_train)
                loss.backward()
                optimizer.step()
                update_times += 1

            if break_is_true:
                break
        if update_times > target_network_update_frequency:
            update_times = 0
            target_model.load_state_dict(model.state_dict())
        epsilon = init_epsilon - (init_epsilon - final_epsilon) * (frame_count/final_exploration_frame)
        if episode % test_every_episode == 0:
            r_list = []
            for ii in range(5):
                 r_list.append(test(model, gamename))
            r = np.array(r_list).mean()
            logging.info(str(r_list))
            logging.info("test result: %s" % r)
        if epsilon < final_epsilon:
            epsilon = final_epsilon
        if episode % 5 == 0:
            logging.info("Episode:%s | Reward: %s | timestep: %s/%s | epsilon: %.2f" %(episode, reward_one_episode, frame_count, frame_max, epsilon))
        if episode % 100 == 0:
            torch.save(model.state_dict(), model_storage_path+"dqn2015"+gamename+'.pt')

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
@click.option('--gamename')
@click.option('--logfile', default='default.log', help='the name of log file')
def main(gamename, logfile):
    setup_logging(logfile)
    env = gym.make(gamename)
    logging.info("gamename:%s" % gamename)
    model = Q_Net(env.action_space.n).to(device)
    target_model = Q_Net(env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    train(model, target_model, env, gamename)


if __name__ == '__main__':
    main()

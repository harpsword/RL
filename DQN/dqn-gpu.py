
import os
import time
import json
import click
import gym
import logging
import torch
import numpy as np
import torch.nn.functional as F

from policy import Q_Net, process
from data import Data

timestep_limit = 10000000
LogFolder = os.path.join(os.getcwd(), 'log')
FRAME_SKIP = 4
GAMMA = 0.9
epsilon_init = 1.0
epsilon_decay_every_timestep = 100000
epsilon_minus = 0.09
epsilon_final = 0.1

# training
batchsize = 32

# experience replay storage
D = Data()

gpu_id = torch.cuda.current_device()
print("using GPU %s" % gpu_id)
device = torch.device(gpu_id)

def train(cfg, model, env):
    action_n = env.action_space.n
    env.frameskip = 1
    epsilon = epsilon_init
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['optimizer']['args']['lr'])
    timestep_accumulate = 0
    for episode in range(1, cfg['game']['episode']):
        t0 = time.time()
        obs = env.reset()
        obs_list = [obs, obs, obs, obs]
        state_now = process(obs_list)
        
        break_is_true = False
        reward_one_episode = 0
        for step in range(cfg['game']['timesteplimit']):
            #if step % 10 == 0:
            #    print("step:", step, "time:", time.time()-t0)
            if np.random.rand() <= epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = model(state_now.to(device)).argmax().item()
            obs_list = []
            ep_r = 0
            for i in range(FRAME_SKIP):
                obs, reward, done, _ = env.step(action)
                obs_list.append(obs)
                ep_r += reward
                if done:
                    break_is_true = True
            while len(obs_list) < FRAME_SKIP:
                # when len(obs_list) < 4, done=True
                # like start state obs_list, stack more end state together
                obs_list.append(obs_list[-1])
            state_after = process(obs_list)
            sequence = [state_now, action, ep_r, state_after, done]
            state_now = state_after
            reward_one_episode += ep_r
            D.push(sequence)

            # sample data
            # train model
            if len(D.data) >= batchsize:
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
                    q_eval = model(state_.to(device))
                    if target_q_value is None:
                        target_q_value = q_eval
                    else:
                        target_q_value = torch.cat((target_q_value, q_eval))
                    # only one element is different to q_eval
                    if done_:
                        target_q_value[-1][action_] = reward_
                    else:
                        target_q_value[-1][action_] = reward_ + GAMMA*model(next_state_.to(device)).max().item()
                x_train = torch.cat(state_batch).to(device)
                y_train = target_q_value.to(device)
                # dataset = Data.TensorDataset(state_batch, target_q_value)
                optimizer.zero_grad()
                predicted_q_value = model(x_train)
                loss = F.mse_loss(predicted_q_value, y_train)
                loss.backward()
                optimizer.step()

            if break_is_true:
                break
        timestep_accumulate += step
        logging.info("Episode: %s | Reward: %s | timestep: %s/%s | epsilon: %.2f " % (episode, reward_one_episode, timestep_accumulate, timestep_limit, epsilon))
        if episode % 5 == 0:
            print("Episode: %s | Reward: %s | timestep: %s/%s | epsilon: %.2f " % (episode, reward_one_episode, timestep_accumulate, timestep_limit, epsilon))
        epsilon = epsilon_init - int(timestep_accumulate / epsilon_decay_every_timestep) * epsilon_minus
        if epsilon <= epsilon_final:
            epsilon = epsilon_final

            
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
    model = Q_Net(env.action_space.n).to(device)
    train(cfg, model, env)

if __name__ == '__main__':
    main()

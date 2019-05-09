
import os
import json
import click
import gym
import logging
import torch
import numpy as np

from policy import Q_Net, process
from data import Data

LogFoler = os.path.join(os.getcwd(), 'log')
FRAME_SKIP = 4
GAMMA = 0.9
init_epsilon = 0.5
decay_every_timestep = 100000
epsilon_decay = 0.5
final_epsilon = 0.1

# training
batchsize = 32

# experience replay storage
D = Data()

def train(cfg, model, env):
    env.frameskip = 1
    epsilon = init_epsilon 
    optimizer = torch.optim.rmsprop(model.parameters())
    for episode in range(1, cfg['game']['episode']):
        obs = env.reset()
        obs_list = [obs, obs, obs, obs]
        state_now = process(sequence, cfg)
        
        break_is_true = False
        for step in range(cfg['game']['timesteplimit']):
            if np.randon.rand() <= epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = model(state_now).argmax().item()
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
            sequence = [state_now, action, ep_r, process(obs_list), done]
            D.push(sequence)

            # sample data
            # train model
            if len(D.data) >= batchsize:
                selected_data = np.random.sample(D.data, batchsize)
                state_batch = [batch[0] for batch in selected_data]
                target = []
                for i in range(len(batchsize)):
                    if selected_data[i][4]:
                        target.append(selected_data[i][2])
                    else:
                        target.append(selected_data[i][2]+GAMMA*model(state_batch[i]).max().item())
                # build dataset
                # build dataloader
                # update

            if break_is_true:
                break


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
    assert exp_file is not None
    if exp_file:
        with open(exp_file, 'r') as f:
            cfg = json.loads(f.read())
    setup_logging(logfile)
    env = gym.make(cfg['game']['gamename'])
    model = Q_Net(env.action_space.n)
    train(cfg, model, env)



if __name__ == '__main__':
    main()

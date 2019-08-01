

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
input_n = 4
output_n = 2
    


class MLPPolicy(nn.Module):

    def __init__(self, input_n, output_n):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(input_n, 10)
        self.fc2 = nn.Linear(10, output_n)
        self.output_n = output_n
        self.input_n = input_n

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def return_action(self, x):
        x = torch.from_numpy(x).float().reshape(1, self.input_n)
        x = self.forward(x)
        prob = F.softmax(x,dim=1)
        return np.random.choice(self.output_n, p=prob.detach().numpy()[0])


def train(model, Generation, lr):
    env = gym.make('CartPole-v0')
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criteria = torch.nn.CrossEntropyLoss()

    gamma = 0.99
    for i in range(Generation):
        obs = env.reset()
        Reward_list = []
        State_list = []
        Action_list = []
        done = False
        while not done:
            State_list.append(obs)
            action = model.return_action(obs)
            obs, reward, done, _ = env.step(action)
            Reward_list.append(reward)
            Action_list.append(action)
        G_list = []
        Reward_list = Reward_list[::-1]
        State_list = State_list[::-1]
        Action_list = Action_list[::-1]
        for indx, R in enumerate(Reward_list):
            if indx == 0:
                G_list.append(R)
            else:
                G_list.append(R+gamma*G_list[-1])
        # train

        input_shape = (1, input_n)
        output_shape = (1, output_n)
        for indx in range(len(Action_list)):
            optimizer.zero_grad()
            train_input = torch.from_numpy(State_list[indx]).float().reshape(input_shape)
            output = model(train_input)
            target = torch.tensor(Action_list[indx]).reshape((1))
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()
    return model

def main():
    model = MLPPolicy(input_n, output_n, 0.001)
    model = train(model, 100)


if __name__ == '__main__':
    main()



import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy2015(nn.Module):
    '''
    input: (N, 4, 84, 84)
    batch size = N 
    '''
    def __init__(self, ac_space):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, ac_space)

    def forward(self, x:torch.tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Policy2013(nn.Module):
    '''
    input: (N, 4, 84, 84)
    batch size = N = 32
    '''
    def __init__(self, ac_space):
        super(Policy2013, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, ac_space)

    def forward(self, x:torch.tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Value(nn.Module):
    """
    Input: (N, 4, 84, 84)
    """
    def __init__(self):
        super(Value, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



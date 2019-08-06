"""
It's for MuJoCo Environment
Deterministic Policy (DDPG)
TODO
"""

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AbstractPolicy(nn.Module):
    """
    AbstractPolicy class
    """
    def __init__(self):
        super(AbstractPolicy, self).__init__()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight())
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


class Policy(AbstractPolicy):

    def __init__(self, state_dim, action_dim, action_lim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)
        self.__initialize_weights()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        # TODO: the use of action_lim
        return x * self.action_lim


class Value(nn.Module):
    """
    Input: (N, 4, 84, 84)
    """
    def __init__(self, state_dim, action_dim):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fcs1 = nn.Linear(self.state_dim, 64)
        self.fcs2 = nn.Linear(64, 64)
        self.fca1 = nn.Linear(self.action_dim, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self._initialize_weights()

    def forward(self, x, a):
        # TODO: the existence of fca1 for action input
        pass   

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight())
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)



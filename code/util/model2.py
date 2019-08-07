"""
It's for MuJoCo Environment
Deterministic Policy (for DDPG)
TODO
"""

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

EPS = 3e-3


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
        nn.init.uniform_(self.fc1.weight, -math.sqrt(state_dim), math.sqrt(state_dim))

        self.fc2 = nn.Linear(64, 64)
        nn.init.uniform_(self.fc2.weight, -math.sqrt(64), math.sqrt(64))

        self.fc3 = nn.Linear(64, self.action_dim)
        nn.init.uniform_(self.fc3.weight, -EPS, EPS)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
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
        nn.init.uniform_(self.fcs1.weight, -1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim))

        self.fcs2 = nn.Linear(64, 64)
        nn.init.uniform_(self.fcs2.weight, -1/math.sqrt(64), 1/math.sqrt(64))

        self.fca1 = nn.Linear(self.action_dim, 64)
        nn.init.uniform_(self.fca1.weight, -1/math.sqrt(self.action_dim), 1/math.sqrt(self.action_dim))
        
        self.fc2 = nn.Linear(128, 64)
        nn.init.uniform_(self.fc2.weight, -1/math.sqrt(128), 1/math.sqrt(128))

        self.fc3 = nn.Linear(64, 1)
        nn.init.uniform_(self.fc3.weight, -EPS, EPS)

    def forward(self, x, a):
        x1 = F.relu(self.fcs1(x))
        x2 = F.relu(self.fcs2(x1))
        a1 = F.relu(self.fca1(a))

        xx = torch.cat((x2, a1), dim=1)
        xx = F.relu(self.fc2(xx))
        return self.fc3(xx)


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
        self.fc1 = nn.Linear(self.state_dim, 400)

        self.fc2_in = 400
        self.fc2 = nn.Linear(self.fc2_in, 300)

        self.fc3 = nn.Linear(300, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_lim


class Value(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        fc1_input = state_dim + action_dim
        self.fc1 = nn.Linear(fc1_input, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        xi = torch.cat([x, a], dim=1)
        xi = F.relu(self.fc1(xi))
        xi = F.relu(self.fc2(xi))
        return self.fc3(xi)

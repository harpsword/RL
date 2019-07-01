# model come from [mnih et al. 2013]
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchvision import transforms

trans = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ]
)

class Q_Net(nn.Module):
    '''
    input: (N, 4, 84, 84)
    batch size = N = 32
    '''
    def __init__(self, ac_space):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, ac_space)

    def forward(self, x:torch.tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def process(x_list):
    new_xlist = []
    for x in x_list:
        x = transforms.ToPILImage()(x).convert('RGB')
        # output of trans: (1, 84, 84)
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        new_xlist.append(x)
    x = torch.cat(new_xlist, 1)
    return x

class ProcessUnit(object):

    def __init__(self, length):
        self.length = length
        self.frame_list = deque(maxlen=length)
        self.previous_frame = None

    def step(self, x):
        if len(self.frame_list) == self.length:
            self.previous_frame = self.frame_list[0]
            self.frame_list.append(x)
        else:
            self.frame_list.append(x)

    def to_torch_tensor(self):
        assert len(self.frame_list) == self.length
        assert self.previous_frame is not None
        x_list = self.frame_list
        frame_skip = self.length
        new_xlist = [np.maximum(self.previous_frame, x_list[0])]
        for i in range(frame_skip-1):
            new_xlist.append(np.maximum(x_list[i],x_list[i+1]))
        for idx, x in enumerate(new_xlist):
            new_xlist[idx] = self.transform(new_xlist[idx])
        return torch.cat(new_xlist, 1)

    def transform(self, x):
        x = transforms.ToPILImage()(x).convert('RGB')
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        return x

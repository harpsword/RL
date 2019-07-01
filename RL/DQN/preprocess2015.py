import torch
import numpy as np
from torchvision import transforms
from collections import deque

trans = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ]
)

class ProcessUnit(object):
    """
    Initialization:
    PU = ProcessUnit(4, Frame_skip)

    # when encounter one obs
    obs = env.reset()
    PU.step(obs)

    ...... add other obs

    # get torch tensor
    # PU will transform the 4 frames in self.frame_list into a tensor(1,4,84,84)
    action = model(PU.to_torch_tensor())
    for i in range(Frame_skip):
        obs, reward, done, _ = env.step(action)
        # PU will delete the older frame in PU.frame_list, then add new obs
        PU.step(obs)
    """
    def __init__(self, length, frame_skip):
        self.length = length * frame_skip
        self.frame_list = deque(maxlen=self.length)

    def step(self, x):
        # insert in left, so the element of index 0 is newest
        self.frame_list.appendleft(x)

    def to_torch_tensor(self):
        length = len(self.frame_list)
        x_list = []
        i = 0
        while i < length:
            if i == length - 1:
                x_list.append(self.transform(self.frame_list[i]))
            else:
                x = np.maximum(self.frame_list[i], self.frame_list[i+1])
                x_list.append(self.transform(x))
            i += 4
        while len(x_list) < 4:
            x_list.append(x_list[-1])
        return torch.cat(x_list, 1)
        #return torch.cat(x_list[::-1], 1)

    def transform(self, x):
        x = transforms.ToPILImage()(x).convert('RGB')
        x = trans(x)
        x = x.reshape(1, 1, 84, 84)
        return x


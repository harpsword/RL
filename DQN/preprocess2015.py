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
    PU = ProcessUnit(Frame_skip)

    # when encounter one obs
    obs = env.reset()
    PU.step(obs)

    ...... add other obs

    for i in range(Frame_skip):
        # get torch tensor
        # PU will transform the 4 frames in self.frame_list into a tensor(1,4,84,84)
        action = model(PU.to_torch_tensor())
        obs, reward, done, _ = env.step(action)
        # PU will delete the older frame in PU.frame_list, then add new obs
        PU.step(obs)
    """
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



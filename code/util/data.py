# it's for experience replay
# storage data base : collections.deque
# support thread-safe

import random
from collections import deque

class Data(object):

    def __init__(self, maxlen=1000000):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def push(self, x):
        self.data.append(x)

    def get(self, indx):
        return self.data[indx]

    def sample(self, size):
        batch = random.sample(self.data, size)
        s_arr = [b[0] for b in batch]
        a_arr = [b[1] for b in batch]
        r_arr = [b[2] for b in batch]
        sn_arr = [b[3] for b in batch]
        done_arr = [b[4] for b in batch]
        return s_arr, a_arr, r_arr, sn_arr, done_arr



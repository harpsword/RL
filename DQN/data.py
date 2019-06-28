# it's for experience replay
# storage data base : collections.deque
# support thread-safe

from collections import deque

class Data(object):

    def __init__(self, maxlen=1000000):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def push(self, x):
        self.data.append(x)

    def get(self, indx):
        return self.data[indx]




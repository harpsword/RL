
import gym
import ray
import click

from ..DQN.preprocess2015 import ProcessUnit
from ..common.model import Policy2013, Value

ray.init()

@ray.remote
def rollout():
    pass


def main():
    actor = Policy2013()
    rollout_result = 






import ray
from RL.AC import a2c
from RL.PolicyGradient import ppo2

ray.init()

ppo2.main("Amidar-v0")

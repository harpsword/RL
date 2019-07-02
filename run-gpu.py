
import ray
from RL.AC import a2cgpu

ray.init()

a2cgpu.main("Amidar-v0")

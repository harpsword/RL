
import os
import json
import click
import gym

from policy import Q_Net
from data import Data


# experience replay storage
D = Data()

def train(cfg):
    interact()
    
    


@click.command()
@click.option('--exp_file')
@click.option('--log_dir')
def main(exp_file, log_dir):
    assert exp_file is not None
    if exp_file:
        with open(exp_file, 'r') as f:
            cfg = json.loads(f.read())
    log_dir = os.path.expanduser(log_dir) if log_dir



if __name__ == '__main__':
    main()


import ray
import os
import pandas as pd


def save_record(ids, folder_dir, filename):
    records_dict = {
        "episode":[],
        'steps':[],
        'reward':[],
        'gamelength':[]
    }
    for records_id_ in ids:
        record = ray.get(records_id_)
        for key, value in record.items():
            records_dict[key].extend(value)
    records_pd = pd.DataFrame(records_dict)
    records_pd.to_csv(os.path.join(folder_dir, filename)) 


def soft_update(a, b, tau):
    """
    another method for updating target model's parameter softly
    come from https://github.com/sfujim/TD3/blob/master/TD3.py
    for param, target_param in zip(a.parameters(), b.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    """
    # using a to update b softly
    for name, value in b.named_parameters():
        tmp = a
        for attr_value in name.split('.'):
            tmp = getattr(tmp, attr_value)
        value.data = tau * tmp.data + (1 - tau) * value.data


def check_env(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low
    assert len(action_low) == len(action_high)
    for i in range(len(action_low)):
        if abs(action_low[i]) != abs(action_high[i]):
            raise ValueError("Environment Error with wrong action low and high")
    return state_dim, action_dim, action_high[0]

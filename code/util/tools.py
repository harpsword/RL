
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
    # using a to update b softly
    for name, value in b.named_parameters():
        tmp = a
        for attr_value in name.split('.'):
            tmp = getattr(tmp, attr_value)
        value.data = tau * tmp.data + (1 - tau) * value.data


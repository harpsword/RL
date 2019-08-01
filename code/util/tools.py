
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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .layers import create_dataloader_AR


def ADSLD(model, test_samples, method='num', t_value=3):
    mse = nn.MSELoss(reduction='none')
    system_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            h = model( batched_feats)
            loss = mse(h, batched_targets) # 128,46,130

            instance_deviation = torch.sum(loss, dim=-1)
            topk_values, topk_indices = torch.topk(instance_deviation, k=t_value, dim=-1)
            mask = torch.zeros_like(instance_deviation)
            mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)
            system_level_deviation = torch.sum(loss * mask, dim=1)

            tmp_df = pd.DataFrame(system_level_deviation.detach().numpy())
            tmp_df['timestamp'] = batch_ts
            system_level_deviation_df = pd.concat([system_level_deviation_df, tmp_df])
    return system_level_deviation_df.reset_index(drop=True)


def SLD(time_model,space_model, test_samples, method='num', t_value=3):
    mse = nn.MSELoss(reduction='none')
    system_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)
    time_model.eval()
    space_model.eval()

    with torch.no_grad():
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            h1 = time_model( batched_feats)
            h = space_model(batched_graphs, h1)
            loss = mse(h, batched_targets) # 128,46,130

            instance_deviation = torch.sum(loss, dim=-1)
            topk_values, topk_indices = torch.topk(instance_deviation, k=t_value, dim=-1)
            mask = torch.zeros_like(instance_deviation)
            mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)
            system_level_deviation = torch.sum(loss * mask, dim=1)

            tmp_df = pd.DataFrame(system_level_deviation.detach().numpy())
            tmp_df['timestamp'] = batch_ts
            system_level_deviation_df = pd.concat([system_level_deviation_df, tmp_df])
    return system_level_deviation_df.reset_index(drop=True)

def ILD(time_model,space_model, test_samples):
    mse = nn.MSELoss(reduction='none')
    instance_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=128, shuffle=False)
    time_model.eval()
    space_model.eval()

    with torch.no_grad():
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
            h1 = time_model( batched_feats)

            h = space_model(batched_graphs, h1)

            loss = mse(h, batched_targets)

            batch_size, instance_size, channel_size = loss.shape
            string_tensor = np.array([str(row.tolist()) for row in loss.reshape(-1, channel_size)])
            tmp_df = pd.DataFrame(string_tensor.reshape(batch_size, instance_size))
            tmp_df['timestamp'] = batch_ts
            instance_level_deviation_df = pd.concat([instance_level_deviation_df, tmp_df])
    return instance_level_deviation_df.reset_index(drop=True)      

def aggregate_instance_representations(cases, instance_level_deviation_df, before=60, after=300):
    instance_representations = []
    for _, case in cases.iterrows():
        instance_representation = []
        agg_df = instance_level_deviation_df[(instance_level_deviation_df['timestamp']>=(case['timestamp']-before)) & (instance_level_deviation_df['timestamp']<(case['timestamp']+after))]
        for col_name, col_data in agg_df.items():
            if col_name == 'timestamp':
                continue
            instance_representation.extend([(col_name, eval(item)) for item in col_data])
        instance_representations.append(instance_representation)
    return instance_representations

def aggregate_failure_representations(cases, system_level_deviation_df, type_hash=None, before=60, after=300):
    failure_representations, type_labels = [], []
    for _, case in cases.iterrows():
        agg_df = system_level_deviation_df[(system_level_deviation_df['timestamp']>=(case['timestamp']-before)) & (system_level_deviation_df['timestamp']<(case['timestamp']+after))]
        failure_representations.append(list(agg_df.mean()[:-1])) # mean
        if type_hash:
            type_labels.append(type_hash[case['failure_type']])
        else:
            type_labels.append(case['failure_type'])
    return failure_representations, type_labels

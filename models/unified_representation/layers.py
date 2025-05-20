import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def collate_AR(samples):
    timestamps, graphs, feats, targets = map(list, zip(*samples))
    batched_ts = torch.stack(timestamps)
    batched_graphs = dgl.batch(graphs)
    batched_feats = torch.stack(feats)
    batched_targets = torch.stack(targets)
    return  batched_ts, batched_graphs, batched_feats, batched_targets

def create_dataloader_AR(samples, window_size=6, max_gap=60, batch_size=2, shuffle=False):
    # sliding_time_windows
    series_samples = [samples[i:i+window_size] for i in range(len(samples) - window_size + 1)]
    series_samples = [
        series_sample for series_sample in series_samples
            if all(abs(series_sample[i][0] - series_sample[i+1][0]) <= max_gap 
                for i in range(len(series_sample) - 1))
    ]
    # create a dataloader
    dataset = [[
            torch.tensor(series_sample[-1][0]),
            series_sample[-1][1], 
            torch.stack([step[2] for step in series_sample[:-1]]),
            torch.tensor(series_sample[-1][2])
        ] for _, series_sample in enumerate(series_samples)]
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_AR)
    return dataloader

# end Collate Function  ------------------------------------------

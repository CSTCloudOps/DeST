import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from layers import  create_dataloader_AR
from tsmodel import TemporalModel,SpatioModel
import matplotlib.pyplot as plt

def time_train(samples, config):
    batch_size = config['batch_size']
    epochs = config['epochs']
    channel_dim = config['channel_dim']

    
    PATIENCE = 10
    early_stop_threshold = 1e-3
    prev_loss = np.inf
    stop_count = 0

    time_model = TemporalModel(channel_dim,channel_dim)

    best_time_state_dict= time_model.state_dict()


    Loss = nn.MSELoss()
    time_opt = torch.optim.Adam(time_model.parameters(), lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(time_opt, factor=0.1, patience=5)
    dataloader = create_dataloader_AR(samples, batch_size=batch_size, shuffle=True)
    loss_history = []

    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
                
            time_opt.zero_grad()
            h = time_model( batched_feats)
            loss = Loss(h, batched_targets)
            loss.backward()
            time_opt.step()
            running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        loss_history.append(epoch_loss)  # Store loss for this epoch

        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print('Early stopping')
                time_model.load_state_dict(best_time_state_dict)
                break
        else:
            best_time_state_dict = time_model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        if epoch % 1 == 0:
            print(f'epoch {epoch} loss: ', epoch_loss)
        scheduler.step(np.mean(running_loss))
    
    return time_model





def space_train(samples,time_model, config):
    # hyperparameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    channel_dim = config['channel_dim']
    
    PATIENCE = 10
    early_stop_threshold = 1e-3
    prev_loss = np.inf
    stop_count = 0
    

    time_model = time_model
    space_model = SpatioModel(channel_dim,channel_dim)

    best_space_state_dict= space_model.state_dict()

    Loss = nn.MSELoss()

    space_opt = torch.optim.Adam(space_model.parameters(), lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(space_opt, factor=0.1, patience=5)
    space_loss_history = []

    dataloader = create_dataloader_AR(samples, batch_size=batch_size, shuffle=True)
    time_model.eval()
    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batch_ts, batched_graphs, batched_feats, batched_targets in dataloader:
                
            space_opt.zero_grad()
            h1 = time_model( batched_feats)
            h = space_model(batched_graphs, h1)

            loss = Loss(h, batched_targets)
            loss.backward()
            space_opt.step()
            running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        space_loss_history.append(epoch_loss)

        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print('Early stopping')
                space_model.load_state_dict(best_space_state_dict)
                break
        else:
            best_space_state_dict = space_model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        if epoch % 1 == 0:
            print(f'epoch {epoch} loss: ', epoch_loss)
        scheduler.step(np.mean(running_loss))
    return space_model
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv




class TemporalModel(nn.Module):
    def __init__(self, in_dim, out_dim, time_window=5, diff_order=1):

        super(TemporalModel, self).__init__()
        assert diff_order in [1, 2, 3]
        self.diff_order = diff_order
        self.hidden_dim = in_dim * 2

        # 卷积层
        self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_dim, in_dim , kernel_size=5, padding=2)

        # 全连接层
        self.fc = nn.Linear(in_dim*2  * (time_window-1), out_dim)

    def diff(self, features):
        for _ in range(self.diff_order):
            features = features[:, 1:, :, :] - features[:, :-1, :, :]
        return features

    def forward(self, features):
        batch_size, time_window, instance, channel = features.shape

        features = self.diff(features) 

        features = features.permute(0, 2, 3, 1) 
        features = features.reshape(batch_size * instance, channel, -1) 

        h1 = F.relu(self.conv1(features)) 
        h2 = F.relu(self.conv2(features))  
        h1 = h1.view(batch_size, instance, -1) 
        h2 = h2.view(batch_size, instance, -1) 

        combined = torch.cat((h1, h2), dim=-1)  
        result = self.fc(combined.view(batch_size * instance, -1)) 
        result = result.view(batch_size, instance, -1)

        return result
    


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GATEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GATConv(in_dim, hidden_dim, num_heads=2, feat_drop=dropout, attn_drop=dropout, negative_slope=0.2, allow_zero_in_degree=True)
        self.convs = nn.ModuleList([dgl.nn.GATConv(hidden_dim, hidden_dim, num_heads=2, feat_drop=dropout, attn_drop=dropout, negative_slope=0.2) for _ in range(num_layers - 2)])
        if num_layers > 1:
            self.convs.append(dgl.nn.GATConv(hidden_dim, out_dim, num_heads=2, feat_drop=dropout, attn_drop=dropout, negative_slope=0.2))

    def forward(self, g, features):
        h = F.elu(self.input_conv(g, features).mean(1))
        if self.training:
            h = self.dropout(h)
        for conv in self.convs:
            h = F.elu(conv(g, h).mean(1))
            if self.training:
                h = self.dropout(h)
        return h



class SpatioModel(nn.Module):
    def __init__(self, in_dim, out_dim, instance= 46, channel = 130, time_window = 5, hidden_dim=128):
        super(SpatioModel, self).__init__()
        self.mlp = nn.Linear(out_dim, out_dim)
        gnn_layers =2 # The number of layers of the encoder/decoder.

        gru_hidden_dim =out_dim
        gnn_hidden_dim = gru_hidden_dim*3
        gnn_out_dim = gru_hidden_dim
        self.GraphEncoder = GATEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout = 0.5,norm='none')
        self.instance = instance  # 46
        self.channel = channel  # 11
        self.time_window = time_window  # 5
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=channel, nhead=2)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.gcn1 = GCNConv(channel, hidden_dim)  
        self.gcn2 = GCNConv(hidden_dim, channel) 
        self.fc = nn.Linear(time_window * channel, channel)



    def forward(self,g, features):
        batch_size , instance, channel = features.shape
        features = features.view(-1,channel)
        h = self.GraphEncoder(g, features) # 92, 32
        h = h.view(batch_size , instance, channel)
        h = F.leaky_relu(self.mlp(h))

        return h
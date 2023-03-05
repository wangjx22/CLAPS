import torch
import globalvar as gl

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

device = gl.get_value('cuda')

# GATNet model
class GATNet(torch.nn.Module):
    def __init__(self, n_output=128, num_features_xd=78, num_features_xt=78,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GATNet, self).__init__()


        self.gat1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.gat2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        self.gat_fc = nn.Linear(output_dim, output_dim)



    def forward(self, x1, edge_index, batch):

        # encoder graph
        # return_attention_weights
        x1, weight = self.gat1(x1, edge_index, return_attention_weights=True)
        # x1 = self.gat1(x1, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1 = self.gat2(x1, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = gmp(x1, batch)

        return x1, weight

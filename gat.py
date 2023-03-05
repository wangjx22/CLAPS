import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np

import globalvar as gl
device = gl.get_value('cuda')
# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=2, num_features_xt=954, output_dim=128, dropout=0.2, file=None):
        super(GATNet, self).__init__()

        # graph drug layers
        self.drug1_gcn1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.drug1_gcn2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        # self.drug1_gcn3 = GATConv(output_dim, output_dim, dropout=dropout)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim)
        # self.drug1_fc_g2 = nn.Linear(2048, output_dim)
        self.filename = file


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2),
            nn.ReLU()
        )

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def get_col_index(self, x):
        row_size = len(x[:, 0])
        row = np.zeros(row_size)
        col_size = len(x[0, :])
        for i in range(col_size):
            row[np.argmax(x[:, i])] += 1
        return row

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=0)
        # 下面是load操作
        # read_dictionary = np.load('my_file.npy').item()
        # d = pd.DataFrame(d)
        # d.to_csv('data/result/' + path + '.csv', header=0, index=0)

    def forward(self, data1):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        print(type(x1))
        print(type(edge_index1))
        print(x1)
        print(edge_index1)

        # deal drug1
        # begin_x1 = np.array(x1.cpu().detach().numpy())
        x1, arr = self.drug1_gcn1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1, arr = self.drug1_gcn2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        # fin_x1 = np.array(x1.cpu().detach().numpy())


        # if len(batch1) <1000:
        #     dt = pd.DataFrame(begin_x1.T)
        #     # dt = pd.DataFrame(begin_x1)
        #     # dt.to_csv('data/case_study/begin_fin/' + self.filename + '_drug1_begin_x1.csv', header=None, index=None)
        #     # dt = pd.DataFrame(fin_x1)
        #     # dt.to_csv('data/case_study/begin_fin/' + self.filename + '_drug1_fin_x1.csv', header=None, index=None)
        #
        #     # begin_x1 = np.array(dt.corr(method='pearson'))  # 默认为'pearson'检验，可选'kendall','spearman'
        #     # begin_x1 = np.around(begin_x1, decimals=2)
        #     # # dt.to_csv('data/result/begin_x1.csv', index=None, header=None)
        #     # get_map(begin_x1, 'data/case_study/' + self.filename + '_drug1_begin_x1')
        #     #
        #     # dt = pd.DataFrame(fin_x1.T)
        #     # fin_x1 = np.array(dt.corr(method='pearson'))  # 默认为'pearson'检验，可选'kendall','spearman'
        #     # fin_x1 = np.around(fin_x1, decimals=2)
        #     # # dt.to_csv('data/result/fin_x1.csv', index=None, header=None)
        #     # get_map(fin_x1, 'data/case_study/' + self.filename + '_drug1_fin_x1')
        #
        #     dt = pd.DataFrame(arr)
        #     dt = np.around(dt, decimals=2)
        #     get_map(dt, 'data/case_study/' + self.filename + '_drug1_att_x1')
        #
        #     p = self.filename + '_drug1'
        #     self.save_num(x1, p)
        #
        x1 = gmp(x1, batch1)         # global max pooling


        x1 = self.drug1_fc_g1(x1)
        x1 = self.relu(x1)
        # x1 = self.drug1_fc_g2(x1)
        # x1 = self.relu(x1)




        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)



        # add some dense layers
        xc = self.fc1(cell)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

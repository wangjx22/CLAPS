import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn.utils.rnn import pad_sequence
import globalvar as gl
device = gl.get_value('cuda')

"""
下游任务 gCn模型
"""
class Model(nn.Module):
    def __init__(self, n_output=1, cell_dim=954, output_dim=128, dropout=0.2, encoder=None):
        super(Model, self).__init__()

        self.n_output = n_output
        self.encoder = encoder
        self.dropout = dropout

        # predict
        self.pre = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Dropout(dropout),
            nn.Linear(64, self.n_output)
        )

    def forward(self, data, training=False):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y

        # encoder drug1
        x1, w = self.encoder(x, edge_index, batch)

        xc = F.normalize(x1)

        out = self.pre(xc)
        out = out.reshape(-1)

        return out, y
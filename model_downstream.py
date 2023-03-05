import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn.utils.rnn import pad_sequence
import globalvar as gl
device = gl.get_value('cuda')
from transformer_smiles import Emb_Pos_conder
from transformer_smiles import Encoder

"""
下游任务
"""
class Model(nn.Module):
    def __init__(self, dropout=0.2, seq_len=None, d_model=None, trans_encoder=None, Emb_Pos_conder=None, output=None):
        super(Model, self).__init__()
        self.d_model = d_model
        self.output = output
        self.trans_encoder = trans_encoder
        self.Emb_Pos_conder = Emb_Pos_conder
        self.seq_len = seq_len
        self.dropout = dropout
        # coder = Encoder(d_model=d_model, d_ff=d_q, d_k=d_k, d_v=d_v, n_heads=n_heads,
        #                 n_layers=n_layers).to(device)
        self.pre_head = nn.Sequential(
            nn.Linear(self.d_model, 4096),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.output),
            nn.Sigmoid()
        ).to(device)

    def forward(self, data):
        smiles = data

        emb = self.Emb_Pos_conder(smiles)
        enc, enc_self_attns = self.trans_encoder(smiles, emb)

        lenth = len(enc_self_attns)
        layer_sum = enc_self_attns[0]
        flag = 1
        while lenth != 1:
            layer_sum = layer_sum + enc_self_attns[flag]
            lenth = lenth - 1

        head_sum = layer_sum.sum(dim=1)
        mask = head_sum.sum(dim=1)
        # tem = torch.sum(enc, dim=1)
        # enc_outputs = torch.reshape(enc, (enc.shape[0], -1)).to(device)

        enc = torch.sum(enc, dim=1)
        out = self.pre_head(enc)


        return out, enc, mask
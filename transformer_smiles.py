# -*- coding: utf-8 -*-

import math
import torch
import random
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.utils.data as Data


# def make_data(sentences):
#     enc_inputs, dec_inputs, dec_outputs = [], [], []
#     for i in range(len(sentences)):
#         enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
#         dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
#         dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
#
#         enc_inputs.extend(enc_input)
#         dec_inputs.extend(dec_input)
#         dec_outputs.extend(dec_output)
#
#     return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
#
#
# enc_inputs, dec_inputs, dec_outputs = make_data(sentences)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, enc_inputs, enc_outputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn



class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Emb_Pos_conder(nn.Module):
    def __init__(self, src_vocab_size, d_model):
        super(Emb_Pos_conder, self).__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.src_emb = nn.Embedding(self.src_vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)

    def forward(self, smiles):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        enc_outputs = self.src_emb(smiles)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        return enc_outputs

class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, precet, mask_st, seq_len, dropout, trans_encoder, Emb_Pos_conder, DNN_dim1, DNN_dim2):
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.dropout = dropout
        self.seq_len = seq_len
        self.d_v = d_v
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        self.d_ff = d_ff
        self.precet = precet
        self.mask_st = mask_st
        self.n_layers = n_layers
        self.trans_encoder = trans_encoder
        self.DNN_dim1 = DNN_dim1
        self.DNN_dim2 = DNN_dim2
        # self.src_emb = nn.Embedding(self.src_vocab_size, self.d_model).to(device)
        # self.pos_emb = PositionalEncoding(self.d_model).to(device)
        self.emb_pos = Emb_Pos_conder
        # 自注意力机制选择遮盖
        self.attention = Encoder(self.d_model, self.d_ff, self.d_k, self.d_v, n_heads=3, n_layers=1) #.to(device)

        # self.encoder = Encoder(self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads, self.n_layers) #.to(device)
        # self.decoder = Decoder().cuda()
        # self.projection = nn.Linear(d_model, self.tgt_vocab_size, bias=False).cuda()

        self.pre_head = nn.Sequential(
            nn.Linear(self.d_model, self.DNN_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.DNN_dim2, 128)
        )

    def forward(self, data1):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        smiles = data1

        # enc_outputs = self.src_emb(smiles)  #                                                                                                                                                                                                                                                                                                                        [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_outputs = self.emb_pos(smiles) # Initialize original vector

        # mask
        mask_outout, mask_slef_attn = self.attention(smiles, enc_outputs) # Get the vector of the attachments of smiles
        # list addself
        lenth = len(mask_slef_attn)
        layer_sum = mask_slef_attn[0]
        flag = 1
        while lenth != 1:
            layer_sum = layer_sum + mask_slef_attn[flag]
            lenth = lenth - 1

        head_sum = layer_sum.sum(dim=1)

        mask = head_sum.sum(dim=1)


        # Set the character of corresponding weight to 0 to mask
        mask_smiles, index_mask = creat_mask_matrix(mask, smiles, self.precet, self.mask_st)


        enc_outputs_mask = self.emb_pos(mask_smiles)


        project_org, enc_self_attns = self.trans_encoder(smiles, enc_outputs)
        project_mask, enc_self_attns_mask = self.trans_encoder(smiles, enc_outputs_mask)



        return torch.mean(enc_outputs, dim=1).values, self.pre_head(torch.sum(project_org, dim=1)), torch.mean(mask_outout, dim=1).values, self.pre_head(torch.sum(project_mask, dim=1))




def roulette(probability):
    probabilityTotal = np.zeros(len(probability))
    probabilityTmp = 0
    for i in range(len(probability)):
        probabilityTmp += probability[i]
        probabilityTotal[i] = probabilityTmp
    randomNumber=np.random.rand()
    result = 0
    for i in range(1, len(probabilityTotal)):
        if randomNumber < probabilityTotal[0]:
            result = 0
            break
        elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
            result = i
    return result


def creat_mask_matrix(mask, smiles, precet, mask_st):
    """mask (batch_size, src_len) is matrix sumed
    smiles (batch_size, src_len) is input
    this function is to creat a mask_matrix"""
    raw = smiles.shape[0]
    line = smiles.shape[1]
    one = torch.ones(raw, line).int().cuda()
    index_one = torch.zeros(raw, line).long().cuda()
    # Create a new marking matrix and set the unmasked to 0. When calculating the loss at last, the gradient with index 0 will not be calculated
    if mask_st == 'roulette':
        for i in range(raw):
            for j in range(line):
                if mask[i][j] == 0 or j == line:
                    soft = F.softmax(mask[i, :j], dim=0)

                    for num in range(j):
                        pos = roulette(soft) # Return to the selected position
                        one[i][pos] = 0
                        index_one[i][pos] = smiles[i][pos]

                    break

    elif mask_st == 'top':
        for i in range(raw):
            for j in range(line):
                # largest =  true return the max value
                if mask[i][j] == 0 or j == line:
                    values, indices = torch.topk(mask[i, :j], int(math.ceil(precet * j)), largest=True,
                                                 sorted=False)
                    one[i][indices] = 0
                    index_one[i][indices] = smiles[i][indices]
                    break
    else:
        for i in range(raw):
            for j in range(line):
                #
                if mask[i][j] == 0 or j == line:
                    indices = random.sample(range(j), int(math.ceil(precet * j)))
                    one[i][indices] = 0
                    index_one[i][indices] = smiles[i][indices]
                    break

    mask_mat = smiles.mul(one)

    return mask_mat, index_one

# def creat_mask_matrix(mask, smiles, precet):
#     """mask (batch_size, src_len) is matrix sumed
#     smiles (batch_size, src_len) is input
#     this function is to creat a mask_matrix"""
#     raw = smiles.shape[0]
#     line = smiles.shape[1]
#     one = torch.ones(raw, line).int().cuda()
#     for i in range(raw):
#         for j in range(line):
#             if mask[i][j] == 0 or j == line:
#                 values, indices = torch.topk(mask[i][:j], int(math.ceil(precet * j)), largest=True, sorted=False) # False is get the smallest number
#                 one[i][indices] = 0
#                 # soft = F.softmax(mask[i][:j], dim=0)
#                 # for num in range(j):
#                 #     if random.random() < soft[num]:
#                 #         one[i][num] = 0
#                 #     else:
#                 #         continue
#                 break
#             else:
#                 continue
#     re = smiles.mul(one)
#     return re

def creat_mask_matrix_contrast(mask, smiles, precet):
    """mask (batch_size, src_len) is matrix sumed
    smiles (batch_size, src_len) is input
    this function is to creat a mask_matrix"""
    raw = smiles.shape[0]
    line = smiles.shape[1]
    one = torch.ones(raw, line).int().cuda()
    for i in range(raw):
        for j in range(line):
            if mask[i][j] == 0 or j == line:
                values, indices = torch.topk(mask[i][:j], int(math.ceil(precet * j)), largest=False, sorted=False) # False is get the smallest number
                one[i][indices] = 0
                # soft = F.softmax(mask[i][:j], dim=0)
                # for num in range(j):
                #     if random.random() < soft[num]:
                #         one[i][num] = 0
                #     else:
                #         continue
                break
            else:
                continue
    re = smiles.mul(one)
    return re

def create_diff_view(mask,smiles,precet):
    raw = smiles.shape[0]
    line = smiles.shape[1]
    one = torch.ones(raw, line).int().cuda()
    for i in range(raw):
        for j in range(line):
            if mask[i][j] == 0 or j == line:
                break
            if random.random() < precet:
                one[i][j] = 0
            # if mask[i][j] == 0 or j == line:
                # values, indices = torch.topk(mask[i][:j], int(math.ceil(precet * j)), largest=True,
                #                              sorted=False)  # False is get the smallest number
                # one[i][indices] = 0

                # soft = F.softmax(mask[i][:j], dim=0)
                # for num in range(j):
                #     if random.random() < soft[num]:
                #         one[i][num] = 0
                #     else:
                #         continue
            #     break
            # else:
            #     continue
    re = smiles.mul(one)
    return re



def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word

    return dec_input


# # Test
# enc_inputs, _, _ = next(iter(loader))
# for i in range(len(enc_inputs)):
#     greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
#     predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
#     predict = predict.data.max(1, keepdim=True)[1]
#     print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])

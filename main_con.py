# -*- coding: UTF-8 -*-
import argparse
import torch.multiprocessing as mp
import math
import time
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from torch.utils.data import DataLoader as DL
import torch.optim as optim
from tqdm import tqdm
from sklearn.manifold import TSNE
from MyDataSet_smiles import MyDataSet



# from transformer import Transformer
from transformer_smiles import Transformer
from transformer_smiles import Encoder
from transformer_smiles import Emb_Pos_conder
from utils_pretrain import *
from nt_xent import NT_Xent
import matplotlib.pyplot as plt
plt.switch_backend('agg')
"""
model pretrain
"""
def calc_loss(x, x_aug, temperature=0.2, sym=True):
    # x and x_aug shape -> Batch x proj_hidden_dim

    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1

    return loss
# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, device):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_org = torch.Tensor()
    feature_mask = torch.Tensor()
    for tem in train_bar:
        org, out_org, mask, out_mask = net(tem.to(device))
        # feature_org = torch.cat((feature_org, torch.Tensor(org.to('cpu').data.numpy())), 0)
        # feature_mask = torch.cat((feature_mask, torch.Tensor(mask.to('cpu').data.numpy())), 0)
        # view_loss = calc_loss(org, mask)
        criterion = NT_Xent(out_org.shape[0], temperature, 1)
        loss = criterion(out_org, out_mask)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature_org, feature_mask


def get_dict(datafile):

    # smiles 字典 统计所有smiles 字符出现频率 有高到低字典排序
    src_dict = {}
    with open("pretrain/processed/" + datafile + "_dict.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            src_dict[k] = int(v)
    f.close()
    sort_dict = {key: rank for rank, key in enumerate(sorted(src_dict.values(), reverse=True), 1)}
    vocab_dict = {k: sort_dict[v] for k, v in src_dict.items()}

    vocab_dict['<pad>'] = 0
    return vocab_dict

def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2=round((SSR / SST) ** 2,3)
    return r2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CLAPS')
    parser.add_argument('--datafile', default='in-vitro', help='orginal data for input in-vitro tryout now')
    parser.add_argument('--path', default='pretrain', help='orginal data for input')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=1500, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--mask_st', default="roulette", type=str, help='mask strategies', choices=["random", "top", "roulette"])
    parser.add_argument('--prob', default=0.25, type=float, help='the probability of top mask or random mask')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    parser.add_argument('--downtask', default='model_downstream.py', help='Number of sweeps over the dataset to train')
    parser.add_argument('--dim_k', default=128, type=int, help='the Wq and Wk dimension of attention network')
    parser.add_argument('--dim_v', default=128, type=int, help='the Wv dimension of attention network')
    parser.add_argument('--n_heads', default=4, type=int, help='the number of head in attention network')
    parser.add_argument('--n_layers', default=3, type=int, help='the layer of attention encoder')
    parser.add_argument('--d_model', default=512, type=int, help='the dimension of embedding')
    parser.add_argument('--lr', default=0.001, type=float, help='the Learning rate')
    parser.add_argument('--DNN_dim1', default=2048, type=int, help='the 1st fc dimension of DNN')
    parser.add_argument('--DNN_dim2', default=512, type=int, help='the 2nd fc dimension of DNN')




    # args parse
    args = parser.parse_args()
    print(args)
    d_q, d_k, d_v, n_heads, n_layers, d_model = args.dim_k, args.dim_k, args.dim_v, args.n_heads, args.n_layers, args.d_model
    mask_st = args.mask_st
    precet = args.prob
    dropout = args.dropout
    temperature, datafile = args.temperature, args.datafile
    DNN_dim1, DNN_dim2 = args.DNN_dim1, args.DNN_dim1
    batch_size, epochs = args.batch_size, args.epochs

    train_datas = []
    # data prepare
    train_data = PreDataset(root=args.path, dataset=args.datafile)


    vocab_dict = get_dict(datafile)
    vl = len(vocab_dict)
    PAD_IDX = vocab_dict.get('<pad>')
    # encoder drug_smiles
    smile_seqs = []

    # Convert the smiles of a batch into a torch vector
    for smile in train_data:
        smile_seq = [int(vocab_dict.get(i)) for i in smile]

        smile_seqs.append(torch.LongTensor(smile_seq))

    # Uniform sequence length
    src_seq = pad_sequence(smile_seqs, batch_first=True, padding_value=PAD_IDX)
    src_seq_len = src_seq.shape[1] # sequence lenth
    train_data = MyDataSet(src_seq)


    # model setup and optimizer config


    # encoder
    # model_encoder1 = GCNet().cuda()
    emb_pos = Emb_Pos_conder(src_vocab_size=vl, d_model=d_model)
    coder = Encoder(d_model=d_model, d_ff=d_q, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers)
    model = Transformer(tgt_vocab_size=None, d_model=d_model,
                        d_ff=d_q, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers,
                        precet=precet, mask_st=mask_st, seq_len=src_seq_len, dropout=dropout, trans_encoder=coder, Emb_Pos_conder=emb_pos, DNN_dim1=DNN_dim1, DNN_dim2=DNN_dim2)


    # DataParallel training
    if torch.cuda.device_count() > 1:
        print("Gemfield have ", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    emb_pos_name_pre = '{}_src_vocab_size{}_d_model{}'.format(datafile, vl, d_model)
    coder_name_pre = '{}_src_vocab_size{}_d_model{}_d_q{}_d_k{}_d_v{}_heads{}_layers{}_precet{}'.format(datafile, vl, d_model, d_q, d_k, d_v, n_heads, n_layers, precet)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    # optimizer_for_mask = optim.Adam(model.attention.parameters(), lr=1e-4, weight_decay=1e-6)

    # for name, param in model.named_parameters():
    #     if "attention" in name:
    #         param.requires_grad = False
    # optimizer_for_trans = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-6)


    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_batch{}_epoch{}'.format(datafile, batch_size, epochs)
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)
    tsne = TSNE()
    AUCs = ('Epoch\tloss\tr2\ttime')
    minloss = 1000000
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loader = DL(train_data, batch_size=batch_size, shuffle=True)
        train_loss, org, mask = train(model, train_loader, optimizer, device)

        if train_loss < minloss:
            train_loss = minloss
            torch.save(emb_pos.state_dict(), 'data/pretrain/model/Top_' + emb_pos_name_pre + '_emb_pos_epoch_' + str(epoch) + '.pkl')
            torch.save(coder.state_dict(), 'data/pretrain/model/Top_' + coder_name_pre + '_encoder_epoch_' + str(epoch) + '.pkl')

        # if epoch % 10 == 0:
        #     X_embedded = tsne.fit_transform(org)
        #     plt.figure()
        #     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, cmap='viridis')
        #     plt.title('epoch:' + str(epoch))
        #     plt.savefig('results/'+save_name_pre+'/tsne-org' + str(epoch) + '.png')
        #
        #     X_embedded2 = tsne.fit_transform(mask)
        #     plt.figure()
        #     plt.scatter(X_embedded2[:, 0], X_embedded2[:, 1], s=10, cmap='viridis')
        #     plt.title('epoch:' + str(epoch))
        #     plt.savefig('results/'+save_name_pre+'/tsne-mask' + str(epoch) + '.png')
        #
        #     r2 = stats.pearsonr(org.numpy().flatten(), mask.numpy().flatten())
        #     evaluations = [epoch, train_loss, r2, time.time() - start]
        #     save_AUCs(evaluations, 'results/'+save_name_pre+'/evaluation.txt')





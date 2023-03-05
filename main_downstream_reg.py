# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
import argparse
import math
import time
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.metrics import roc_auc_score
import sklearn
from torch.utils.data import DataLoader as DL
import torch.optim as optim
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
import globalvar as gl
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
    print('The code uses CPU!!!')




class MyDownTaskDataSet(Dataset):
    def __init__(self, smiles, label):
        self.smiles = smiles
        self.label = label

    def __getitem__(self, index):
        return self.smiles[index], self.label[index]

    def __len__(self):
        return len(self.label)

# from transformer import Transformer
from transformer_smiles import Transformer
from transformer_smiles import Encoder
from transformer_smiles import Emb_Pos_conder
from utils_dowmstream import *
from nt_xent import NT_Xent
from model_downstream_reg import Model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
"""
模型预训练
"""

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    # feature_graph = torch.Tensor()
    # feature_org = torch.Tensor()
    for tem in train_bar:
        smiles, label = tem
        out = net(smiles.to(device))
        train_optimizer.zero_grad()
        label = label.type(torch.FloatTensor).to(device)
        loss = loss_fn(out, label)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num


def predict(net, data_loader):
    net.eval()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    for tem in train_bar:
        smiles, label = tem
        out = net(smiles.to(device))
        # ys = F.softmax(out, 1).to('cpu').data.numpy()
        ys = out.to('cpu').data.numpy()

        total_preds = torch.cat((total_preds, torch.Tensor(ys)), 0)
        total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def get_dict(datafile):

    # smiles 字典 统计所有smiles 字符出现频率 有高到低字典排序 1- 43
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
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--modelfile', default='in-vitro', help='orginal data for input in-vitro tryout now')
    parser.add_argument('--pre_dict', default='in-vitro', help='orginal data for input in-vitro tryout now')
    parser.add_argument('--path', default='down_task', help='orginal data for input')
    parser.add_argument('--downtask_feature_dim', default=32, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--downtask', default='Lipo', help='the dataset to train')
    d_q, d_k, d_v, n_heads, n_layers, d_model = 128, 128, 128, 4, 3, 512
    # kind = 'Roulette'
    precet = 0.25
    dropout = 0.2

    att = ['Roulette']
    datasplit = ['normal']

    clr_tasks = {'ESOL': 1, 'FreeSolv': 1, 'Lipo': 1}
    # ran_split = 'normal'


    # args parse
    args = parser.parse_args()
    print(args)
    temperature, k, datafile = args.temperature, args.k, args.modelfile

    batch_size, epochs = args.batch_size, args.epochs

    for kind in att:
        for ran_split in datasplit:


            # data prepare
            train_data = TaskDataset(dataset='train', task=args.downtask, pre_dict=args.pre_dict, ran_split=ran_split)
            train_loader = DL(train_data, batch_size=batch_size, shuffle=True)
            seq_len = train_data.data.smiles.shape[1]
            valid_data = TaskDataset(dataset='valid', task=args.downtask, pre_dict=args.pre_dict, ran_split=ran_split)
            valid_loader = DL(valid_data, batch_size=batch_size, shuffle=True)
            test_data = TaskDataset(dataset='test', task=args.downtask, pre_dict=args.pre_dict, ran_split=ran_split)
            test_loader = DL(test_data, batch_size=batch_size, shuffle=True)



            vocab_dict = get_dict(args.pre_dict)
            vl = len(vocab_dict)



            # model setup and optimizer config
            # encoder
            emb_pos_name_pre = '{}_{}_src_vocab_size{}_d_model{}'.format(kind, datafile, vl, d_model)
            coder_name_pre = '{}_{}_src_vocab_size{}_d_model{}_d_q{}_d_k{}_d_v{}_heads{}_layers{}_precet{}'.format(
            kind, datafile, vl, d_model, d_q, d_k, d_v, n_heads, n_layers, precet)

            emb_pos = Emb_Pos_conder(src_vocab_size=vl, d_model=d_model).to(device)
            emb_pos.load_state_dict(torch.load('data/pretrain/model/' + emb_pos_name_pre + '_emb_pos.pkl'))
            emb_pos.eval()

            coder = Encoder(d_model=d_model, d_ff=d_q, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers).to(device)
            coder.load_state_dict(torch.load('data/pretrain/model/' + coder_name_pre + '_encoder.pkl'))
            coder.eval()



            model = Model(dropout=dropout, seq_len=seq_len, d_model=d_model, trans_encoder=coder, Emb_Pos_conder=emb_pos, output=clr_tasks[args.downtask]).to(device)
            if torch.cuda.device_count() > 1:
                print("Gemfield have ", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            # model_encoder2 = Transformer(vocab_size=vl)
            # model = Net(encoder1=None, encoder2=model_encoder2).cuda()
            # loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn = torch.nn.MSELoss()

            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

            # dowm_model = GNNet().cuda()
            # dowm_optimizer = optim.Adam(dowm_model.parameters(), lr=0.0001, weight_decay=1e-6)

            # training loop
            results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
            save_name_pre = '{}_{}_{}_{}'.format(batch_size, epochs, datafile, kind)
            if not os.path.exists('results/'+save_name_pre):
                os.mkdir('results/'+save_name_pre)
            tsne = TSNE()
            AUCs = ('Epoch\tloss\tr2\ttime')

            test_AUCs = 'results/' + save_name_pre + '/' + args.downtask + '_' + ran_split + '_test.txt'
            res = ('Epoch\tmse\tmae\tr2')
            with open(test_AUCs, 'w') as f:
                f.write(res + '\n')

            valid_AUCs = 'results/' + save_name_pre + '/' + args.downtask + '_' + ran_split + '_valid.txt'
            res = ('Epoch\tmse\tmae\tr2')
            with open(valid_AUCs, 'w') as f:
                f.write(res + '\n')

            min_loss = 1000000000
            max_test_auc = 0
            max_valid_auc = 0
            for epoch in range(1, epochs + 1):
                start = time.time()

                train_loss = train(model, train_loader, optimizer)


                T, S= predict(model, valid_loader)
                mse = mean_squared_error(T, S)
                mae = mean_absolute_error(T, S)
                r2 = r2_score(T, S)
                AUCs = [epoch, mse, mae, r2]
                save_AUCs(AUCs, valid_AUCs)
                # compute preformence


                T, S = predict(model, test_loader)
                # compute preformence
                mse = mean_squared_error(T, S)
                mae = mean_absolute_error(T, S)
                r2 = r2_score(T, S)
                AUCs = [epoch, mse, mae, r2]
                save_AUCs(AUCs, test_AUCs)

                # random_num = random.sample(range(0, len(features)), 5000)
                #plt
                # if epoch % 50 == 0:
                #     X_embedded = tsne.fit_transform(features[random_num])
                #     plt.figure()
                #     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, cmap='viridis')
                #     plt.title('epoch:' + str(epoch))
                #     plt.savefig('results/'+save_name_pre+'/tsne-' + str(epoch) + '.png')
                #
                #     X_embedded2 = tsne.fit_transform(org[random_num])
                #     plt.figure()
                #     plt.scatter(X_embedded2[:, 0], X_embedded2[:, 1], s=10, cmap='viridis')
                #     plt.title('epoch:' + str(epoch))
                #     plt.savefig('results/'+save_name_pre+'/2tsne-' + str(epoch) + '.png')
                #
                #     r2 = stats.pearsonr(features.numpy().flatten(), org.numpy().flatten())
                #     evaluations = [epoch, train_loss, r2, time.time() - start]
                #     save_AUCs(evaluations, 'results/'+save_name_pre+'/evaluation.txt')



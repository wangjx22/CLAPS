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
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
import globalvar as gl
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda:3')
    gl.set_value('cuda:3', device)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    gl.set_value('cuda:3', device)
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
from model_downstream import Model
import matplotlib.pyplot as plt

plt.switch_backend('agg')
"""
downstream prediction
"""

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_labels = torch.Tensor()
    feature = torch.Tensor()
    for tem in train_bar:
        smiles, label = tem
        out, fe, att_mask = net(smiles.to(device))
        feature = torch.cat((feature, torch.Tensor(fe.to('cpu').data.numpy())), 0)
        total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)
        train_optimizer.zero_grad()
        label = label.unsqueeze(1).type(torch.FloatTensor).to(device)
        loss = loss_fn(out, label)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature, total_labels.numpy().flatten()


def predict(net, data_loader):
    net.eval()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    feature = torch.Tensor()
    with torch.no_grad():
        for tem in train_bar:
            smiles, label = tem
            out, fe, att_mask = net(smiles.to(device))
            feature = torch.cat((feature, torch.Tensor(fe.to('cpu').data.numpy())), 0)
            # ys = F.softmax(out, 1).to('cpu').data.numpy()
            # ys = out.to('cpu').data.numpy()

            total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
            # total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)
    return total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten(), feature, att_mask


def get_dict(datafile):

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
    parser.add_argument('--batch_size', default=200, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--downtask', default='SIDER', help='the dataset to train')
    d_q, d_k, d_v, n_heads, n_layers, d_model = 128, 128, 128, 4, 3, 512
    # kind = 'Roulette'
    precet = 0.25
    dropout = 0.2
    ep = 'epoch_2'

    att = ['Roulette']
    datasplit = ['random']

    clr_tasks = {'BBBP': 1, 'HIV': 1, 'BACE': 1, 'Tox21': 12, 'ClinTox': 2, 'SIDER': 27}
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
            valid_loader = DL(valid_data, batch_size=batch_size, shuffle=False)
            test_data = TaskDataset(dataset='test', task=args.downtask, pre_dict=args.pre_dict, ran_split=ran_split)
            test_loader = DL(test_data, batch_size=batch_size, shuffle=False)



            vocab_dict = get_dict(args.pre_dict)
            vl = len(vocab_dict)



            # model setup and optimizer config
            # encoder

            emb_pos_name_pre = '{}_{}_src_vocab_size{}_d_model{}'.format(kind, datafile, vl, d_model)
            coder_name_pre = '{}_{}_src_vocab_size{}_d_model{}_d_q{}_d_k{}_d_v{}_heads{}_layers{}_precet{}'.format(
            kind, datafile, vl, d_model, d_q, d_k, d_v, n_heads, n_layers, precet)

            emb_pos = Emb_Pos_conder(src_vocab_size=vl, d_model=d_model).to(device)
            emb_pos.load_state_dict(torch.load('data/pretrain/model/' + emb_pos_name_pre + '_emb_pos_epoch_1.pkl'))

            coder = Encoder(d_model=d_model, d_ff=d_q, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers).to(device)
            coder.load_state_dict(torch.load('data/pretrain/model/' + coder_name_pre + '_encoder_epoch_1.pkl'))



            model = Model(dropout=dropout, seq_len=seq_len, d_model=d_model, trans_encoder=coder, Emb_Pos_conder=emb_pos, output=clr_tasks[args.downtask]).to(device)
            # if torch.cuda.device_count() > 1:
            #     print("Gemfield have ", torch.cuda.device_count(), "GPUs!")
            #     model = torch.nn.DataParallel(model)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            # model_encoder2 = Transformer(vocab_size=vl)
            # model = Net(encoder1=None, encoder2=model_encoder2).cuda()
            # loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn = torch.nn.BCELoss()

            optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)

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
            res = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA')
            with open(test_AUCs, 'w') as f:
                f.write(res + '\n')

            valid_AUCs = 'results/' + save_name_pre + '/' + args.downtask + '_' + ran_split + '_valid.txt'
            res = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA')
            with open(valid_AUCs, 'w') as f:
                f.write(res + '\n')

            min_loss = 1000000000
            max_test_auc = 0
            max_valid_auc = 0
            T_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500, method='barnes_hut')
            for epoch in range(1, epochs + 1):
                start = time.time()

                train_loss, feature_train, total_labels = train(model, train_loader, optimizer)

                Y = T_model.fit_transform(feature_train)
                plt.scatter(Y[:, 0], Y[:, 1], 20, total_labels)
                plt.savefig('results/' + kind + '_' + args.downtask + 'train' + str(epoch) + '.png')
                plt.clf()



                T, S, feature_valid, valid_mask = predict(model, valid_loader)
                # compute preformence
                AUC = roc_auc_score(T, S)
                if AUC > max_valid_auc:
                    np.savetxt('results/' + kind + '_' + args.downtask + 'train.csv',
                               feature_train.cpu().numpy(), fmt='%.5f', delimiter=',')
                    max_valid_auc = AUC
                    AUCs = [epoch, AUC]
                    save_AUCs(AUCs, valid_AUCs)
                    np.savetxt('results/' + kind + '_' + args.downtask + 'valid' + str(epoch) + '.csv', valid_mask.cpu().numpy(), fmt='%.5f', delimiter=',')



                    Y = T_model.fit_transform(feature_valid)
                    plt.scatter(Y[:, 0], Y[:, 1], 20, T)  # Labels is the corresponding label of each line, and 20 is the size of the label
                    plt.savefig('results/' + kind + '_' + args.downtask + 'valid' + str(epoch) + '.png')  # save figure
                    plt.clf()

                T, S, feature_test, test_mask = predict(model, test_loader)
                # compute preformence
                AUC = roc_auc_score(T, S)
                if AUC > max_test_auc:
                    max_test_auc = AUC
                    AUCs = [epoch, AUC]
                    save_AUCs(AUCs, test_AUCs)
                    np.savetxt('results/' + kind + '_' + args.downtask + 'test' + str(epoch) + '.csv', test_mask.cpu().numpy(), fmt='%.5f', delimiter=',')

                    Y = T_model.fit_transform(feature_test)
                    plt.scatter(Y[:, 0], Y[:, 1], 20, T)
                    plt.savefig('results/' + kind + '_' + args.downtask + 'test' + str(epoch) + '.png')
                    plt.clf()


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



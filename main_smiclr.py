# -*- coding: UTF-8 -*-
import argparse
import math
import random
import time
import torch
import torch.optim as optim
from numpy import corrcoef
from torch import nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
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

from transformer import Transformer
from utils_gcn_pretrain import *
from model_transformer_pre import TransformerCon
from model_gcn_pre import GCNCon
from model_gat_pre import GATCon
from nt_xent import NT_Xent
from encoder_gnn import GCNet
from encoder_gat import GATNet
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
plt.switch_backend('agg')
"""
模型预训练

"""

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, vocab_dict):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_graph = torch.Tensor()
    feature_org = torch.Tensor()
    for tem in train_bar:
        graph1, out_1, org2, out_2 = net(tem.to(device),vocab_dict)
        feature_graph = torch.cat((feature_graph, torch.Tensor(graph1.to('cpu').data.numpy())), 0)
        feature_org = torch.cat((feature_org, torch.Tensor(org2.to('cpu').data.numpy())), 0)
        criterion = NT_Xent(out_1.shape[0], temperature, 1)
        loss = criterion(out_1, out_2)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature_graph, feature_org


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--datafile', default='in-vitro', help='orginal data for input in-vitro tryout now')
    parser.add_argument('--path', default='pretrain', help='orginal data for input')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=200, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=150, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--downtask', default='model_downstream.py', help='Number of sweeps over the dataset to train')


    # args parse
    args = parser.parse_args()
    print(args)
    feature_dim, temperature, k, datafile = args.feature_dim, args.temperature, args.k, args.datafile

    batch_size, epochs = args.batch_size, args.epochs
    mode = 'gat'
    train_datas = []
    # data prepare
    # if datafile != 'now':
    #     train_data = TestbedDataset(root=args.path, dataset=args.datafile)
    # else:
    #     for i in range(10):
    #         dataset = args.datafile + '_' + str(i + 1)
    #         print('train_datas: '+ dataset)
    #         train_datas.append(TestbedDataset(root=args.path, dataset=dataset))
    train_data = TestbedDataset(root=args.path, dataset=args.datafile,patt='_re1')

    # vocab_dict = get_dict(datafile)
    # vl = len(vocab_dict)
    # model setup and optimizer config
    vocab_dict = train_data.vocab

    # encoder
    # model_encoder1 = Transformer(vl)
    # model_encoder2 = Transformer(vl)
    # model = TransformerCon(encoder1=model_encoder1, encoder2=model_encoder2).cuda()

    if mode == 'gcn':
        print('use GCN encoder')
        model_encoder1 = GCNet().cuda()
        model_encoder2 = GCNet().cuda()
        model = GCNCon(encoder1=model_encoder1, encoder2=model_encoder2).cuda()
    elif mode == 'gat':
        print('use GAT encoder')
        model_encoder1 = GATNet().cuda()
        model_encoder2 = GATNet().cuda()
        model = GATCon(encoder1=model_encoder1, encoder2=model_encoder2).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)



    # dowm_model = GNNet().cuda()
    # dowm_optimizer = optim.Adam(dowm_model.parameters(), lr=0.0001, weight_decay=1e-6)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(batch_size, epochs, datafile,'re_gat_dex15')
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)
    tsne = TSNE()
    AUCs = ('Epoch\tloss\tr2\ttime')
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loss, features, org = train(model, train_loader, optimizer, vocab_dict)

        random_num = random.sample(range(0, len(features)), 5000)
        end = time.time() - start
        #plt
        if epoch % (epochs / 4) == 0:
            X_embedded = tsne.fit_transform(features[random_num])
            plt.figure()
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, cmap='viridis')
            plt.title('epoch:' + str(epoch))
            plt.savefig('results/'+save_name_pre+'/tsne-' + str(epoch) + '.png')

            X_embedded2 = tsne.fit_transform(org[random_num])
            plt.figure()
            plt.scatter(X_embedded2[:, 0], X_embedded2[:, 1], s=10, cmap='viridis')
            plt.title('epoch:' + str(epoch))
            plt.savefig('results/'+save_name_pre+'/2tsne-' + str(epoch) + '.png')

        if epoch % 100 == 0:
            r2 = stats.pearsonr(features.numpy().flatten(), org.numpy().flatten())
            evaluations = [epoch, train_loss, r2, end]
            save_AUCs(evaluations, 'results/'+save_name_pre+'/evaluation.txt')
        if epoch % (epochs) == 0:
            df1 = pd.DataFrame(features.numpy())
            df2 = pd.DataFrame(org.numpy())
            df1.to_csv('results/' + save_name_pre + '/tsne-' + str(epoch) + '.csv', header=None, index=None)
            df2.to_csv('results/'+save_name_pre +'/2tsne-' + str(epoch) + '.csv', header=None, index=None)


        # for parameters in model.gat_fc.parameters():
        #     print(parameters)
        # dowm_model.gat_fc = model.gat_fc
        # for parameters in dowm_model.gat_fc.parameters():
        #     print(parameters)
        # results['train_loss'].append(train_loss)

        # test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    torch.save(model_encoder1.state_dict(), 'results/model/model_encoder_'+mode+'1_' + save_name_pre + '.pkl')
    torch.save(model_encoder2.state_dict(), 'results/model/model_encoder_'+mode+'2_' + save_name_pre + '.pkl')


#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score,r2_score,mean_squared_error
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
from utils_clr_dowmstream import *
from encoder_gnn import GCNet
from encoder_gat import GATNet
from model_clr_downstream import Model
import random
import torch.nn as nn
import torch.nn.functional as F
import math
import globalvar as gl

""""
药物下游任务
"""

def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE= mse ** 0.5
    return mae, mse, RMSE

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

# train for one epoch to learn unique features
def train(model, device, data_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        output, y = model(data)
        pred = nn.Sigmoid()(output)
        loss = loss_fn(pred,y)
        # print('loss', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(data_loader.dataset),
                                                                           100. * batch_idx / len(data_loader),
                                                                           loss.item()))


def predicting(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output, y = model(data)
            pred = nn.Sigmoid()(output)
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
    return total_preds.numpy().flatten(), total_labels.numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--path', default='clr_task', help='orginal data for input')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=4000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--encoder', default='gat', type=str, help='Number of encoder gcn 2 ')
    parser.add_argument('--predatafile', default='tryout', help='orginal data for input in-vitro tryout now')
    # args parse
    args = parser.parse_args()

    print(args)
    feature_dim, temperature, k, predatafile = args.feature_dim, args.temperature, args.k, args.predatafile

    batch_size, epochs, encoder = args.batch_size, args.epochs, args.encoder

    LOG_INTERVAL = 20

    # best_tasks = SIDER scaffold ClinTox SIDER
    clr_tasks = {'BBBP': 1, 'HIV': 1, 'BACE': 1, 'Tox21': 12, 'ClinTox': 2, 'SIDER': 27, 'MUV': 17}
    task = 'HIV'
    # data prepare
    train_data = TestbedDataset(root=args.path, dataset='train', task=task)
    valid_data = TestbedDataset(root=args.path, dataset='valid', task=task)
    test_data = TestbedDataset(root=args.path, dataset='test', task=task)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=None)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=None)

    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=None)

    # encoder_file = '{}_{}_{}_{}'.format(encoder,batch_size, epochs, predatafile)
    encoder_file = 'gat1_128_10_now_1_re_gat_de25_p1' #'gat1_512_200_in-vitro_re_gat_de15 ' #'gat1_200_150_in-vitro_re_gat_dex'#'gat1_200_300_in-vitro_re_gat_de_min' #'gcn1_300_150_in-vitro_re_rand'  'gcn1_100_300_in-vitro_re'
    """
    读取预训练模型的数据
    使用 GNN encoder or transformer encoder 来预测药物下游任务
    """
    if encoder == 'gcn':
    #GNN
        model_encoder = GCNet().cuda()
        model_encoder.load_state_dict(torch.load('results/model/model_encoder_'+encoder_file+'.pkl'))
        model = Model(n_output=clr_tasks[task], encoder=model_encoder).to(device)
    if encoder == 'gat':
        model_encoder = GATNet().cuda()
        model_encoder.load_state_dict(torch.load('results/model/model_encoder_' + encoder_file + '.pkl'))
        model = Model(n_output=clr_tasks[task], encoder=model_encoder).to(device)

    loss_fn = nn.BCELoss() #BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    save_file ='{}_{}_{}'.format(batch_size, epochs,task)
    if not os.path.exists('results/down_task/clr/model_encoder_'+encoder_file+'_'+task):
        os.makedirs('results/down_task/clr/model_encoder_'+encoder_file+'_'+task)
    result_file_name = 'results/down_task/clr/model_encoder_'+ encoder_file+'_'+task+'/'+save_file+'_result.csv'
    valid_AUCs = 'results/down_task/clr/model_encoder_'+encoder_file+'_'+task+'/'+save_file+'_validAUCs.txt'
    test_AUCs = 'results/down_task/clr/model_encoder_'+encoder_file+'_'+task+'/'+save_file+'_testAUCs.txt'
    model_file_name ='results/down_task/clr/model_encoder_'+encoder_file+'_'+task+'/'+save_file+'_encoder.pkl'
    AUCs = ('Epoch\tAUC\tR2\tMSE\tmse\tmae\trmse\tr2')

    with open(valid_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(test_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    stopping_monitor = 0
    independent_num = []
    for epoch in range(epochs):
        train(model, device, train_data_loader, optimizer, epoch + 1)

        if (epoch + 0) % 10 == 0:
            S, T = predicting(model, device, valid_data_loader)
            # T is correct score
            # S is predict score

            # compute preformence
            mae, mse, rmse = compute_mae_mse_rmse(T, S)
            r2 = compute_rsquared(S, T)

            R2 = r2_score(T, S)
            MSE = mean_squared_error(T, S)
            auc = roc_auc_score(T, S)
            AUCs = [epoch, auc, R2, MSE, mse, mae, rmse, r2]
            print('AUC: ', AUCs)
            # save data

            if best_auc < auc:
                best_auc = auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                save_AUCs(AUCs, valid_AUCs)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
                independent_num.append(T)
                independent_num.append(S)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 100:
                break

    txtDF = pd.DataFrame(data=independent_num)
    txtDF.to_csv(result_file_name, index=False, header=False)

    model.load_state_dict(torch.load(model_file_name))
    S, T = predicting(model, device, test_data_loader)
    mae, mse, rmse = compute_mae_mse_rmse(S, T)
    r2 = compute_rsquared(S, T)

    R2 = r2_score(T, S)
    MSE = mean_squared_error(T, S)
    auc = roc_auc_score(T, S)
    AUCs = [0, auc, R2, MSE, mse, mae, rmse, r2]
    print('test_AUC: ', AUCs)
    # save data
    save_AUCs(AUCs, test_AUCs)

    ## no fine-tuning
    # if not os.path.exists('results/down_task/notune/model_encoder_'+encoder_file+'_'+task):
    #     os.makedirs('results/down_task/notune/model_encoder_'+encoder_file+'_'+task)
    # notun_AUCs = 'results/down_task/notune/model_encoder_' + encoder_file + '_' + task + '/' + save_file + '_notuneAUCs.txt'
    # S, T = predicting(model, device, train_data_loader)
    # mae, mse, rmse = compute_mae_mse_rmse(S, T)
    # r2 = compute_rsquared(S, T)
    #
    # R2 = r2_score(T, S)
    # MSE = mean_squared_error(T, S)
    # auc = roc_auc_score(T, S)
    # AUCs = ['train', auc, R2, MSE, mse, mae, rmse, r2]
    # print('AUC: ', AUCs)
    # # save data
    # save_AUCs(AUCs, notun_AUCs)
    #
    # S, T = predicting(model, device, valid_data_loader)
    # mae, mse, rmse = compute_mae_mse_rmse(S, T)
    # r2 = compute_rsquared(S, T)
    #
    # R2 = r2_score(T, S)
    # MSE = mean_squared_error(T, S)
    # auc = roc_auc_score(T, S)
    # AUCs = ['valid', auc, R2, MSE, mse, mae, rmse, r2]
    # print('validAUC: ', AUCs)
    # # save data
    # save_AUCs(AUCs, notun_AUCs)
    #
    # S, T = predicting(model, device, test_data_loader)
    # mae, mse, rmse = compute_mae_mse_rmse(S, T)
    # r2 = compute_rsquared(S, T)
    #
    # R2 = r2_score(T, S)
    # MSE = mean_squared_error(T, S)
    # auc = roc_auc_score(T, S)
    # AUCs = ['test', auc, R2, MSE, mse, mae, rmse, r2]
    # print('testAUC: ', AUCs)
    # # save data
    # save_AUCs(AUCs, notun_AUCs)
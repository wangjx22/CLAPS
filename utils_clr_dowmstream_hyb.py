import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
import csv
# from creat_data_DC import smile_to_graph
import re

import deepchem as dc

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='tmp', dataset='train', task='bbbp',
                transform=None, pre_transform=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset
        self.dataset = dataset
        self.task = task
        if os.path.isfile(self.processed_paths[0]):
            if dataset == 'train':
                print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
                self.data, self.slices = torch.load(self.processed_paths[0])
            if dataset == 'valid':
                print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[1]))
                self.data, self.slices = torch.load(self.processed_paths[1])
            if dataset == 'test':
                print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[2]))
                self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            # creat_data(self.dataset)
            self.process(root, task)

            if dataset == 'train':
                self.data, self.slices = torch.load(self.processed_paths[0])
            if dataset == 'valid':
                self.data, self.slices = torch.load(self.processed_paths[1])
            if dataset == 'test':
                self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.task + '_train.pt', self.task + '_valid.pt', self.task + '_test.pt']
        # return [self.task + '_train.csv', self.task + '_valid.csv', self.task + '_test.csv']
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, root, task):
        splitter = 'scaffold' #'random'
        featurizer = 'ECFP'
        print(dc.__version__)
        if task == 'BBBP':
            tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer=featurizer, splitter=splitter)
        elif task == 'Tox21':
            tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer, splitter=splitter)
        elif task == 'ClinTox':
            tasks, datasets, transformers = dc.molnet.load_clintox(featurizer=featurizer, splitter=splitter)
        elif task == 'HIV':
            tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer, splitter=splitter)
        elif task == 'BACE':
            tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer=featurizer, splitter=splitter)
        elif task == 'SIDER':
            tasks, datasets, transformers = dc.molnet.load_sider(featurizer=featurizer, splitter=splitter)
        elif task == 'MUV':
            tasks, datasets, transformers = dc.molnet.load_muv(featurizer=featurizer, splitter=splitter)

        train, valid, test = datasets
        save(self, train, 0)
        save(self, valid, 1)
        save(self, test, 2)

def save(self, dataset, path):
    data_list = []
    # smilelist = []
    # labellist = []
    for i in range(len(dataset)):
        smile = dataset.ids[i]
        label = dataset.y[i]
        if len(smile) <= 5:
            continue
        print('smiles', smile)
        # c_size, features, edge_index, atoms = smile_to_graph(smile)
        # if len(edge_index) > 0:
        #     edge_index = torch.LongTensor(edge_index).transpose(1, 0)
        # else:
        #     edge_index = torch.LongTensor(edge_index)
        # GCNData = DATA.Data(x=torch.Tensor(features),
        #                     edge_index=edge_index,
        #                     y=torch.Tensor(label),
        #                     smiles=smile)
        # # append graph, label and target sequence to data list
        #
        # data_list.append(GCNData)
    #     smilelist.append(smile)
    #     labellist.append(label)
    # with open(self.processed_paths[path], 'w') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(dataset)):
    #         writer.writerow(smilelist[i])
    #         writer.writerow(labellist[i])
    if self.pre_filter is not None:
        data_list = [data for data in data_list if self.pre_filter(data)]

    if self.pre_transform is not None:
        data_list = [self.pre_transform(data) for data in data_list]
    print('Graph construction done. Saving to file.')
    data, slices = self.collate(data_list)
    # save preprocessed data:
    torch.save((data, slices), self.processed_paths[path])


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
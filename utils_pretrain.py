import os
import numpy as np
from math import sqrt
from scipy import stats
import torch
import pandas as pd
from creat_data_DC import smile_to_graph
import re
from torch.utils.data import Dataset
"""
预训练数据处理
"""
class PreDataset(Dataset):
    def __init__(self, root='tmp', dataset='_drug1'):

        #root is required for save preprocessed data, default is '/tmp'
        super(PreDataset, self).__init__()
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile('pretrain/processed/' + dataset + '.pt'):
            print('Pre-processed data found: {}, loading ...'.format('pretrain/processed/' + dataset + '.pt'))
            self.data = torch.load('pretrain/processed/' + dataset + '.pt')
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format('pretrain/processed/' + dataset + '.pt'))
            # creat_data(self.dataset)
            self.process(root, self.dataset)
            self.data = torch.load('pretrain/processed/' + dataset + '.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, root, dataset):
        data_list = []
        compound_iso_smiles = []
        df = pd.read_csv('data/' + root + '/data/' + dataset + '.csv')
        compound_iso_smiles += list(df['smiles'])
        compound_iso_smiles = set(compound_iso_smiles)
        org_dict = {}
        count = 0
        for smile in compound_iso_smiles:
            count = count + 1
            if len(smile) < 8:
                continue
            atoms = smile_to_graph(smile)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            org_dict, smiles = get_dict(smile, atoms, org_dict)


            data_list.append(smiles)
            print(count, smiles)


        print('Saving to file.')
        with open("pretrain/processed/" + dataset + "_dict.txt", 'w') as f:
            for k, v in org_dict.items():
                f.write(str(k) + ' ' + str(v) + '\n')
            f.close()
        # save preprocessed data:
        # save preprocessed data:
        torch.save(data_list, 'pretrain/processed/' + dataset + '.pt')



# use result=pd.value_counts(list) 防止重复
def get_dict(smile, atoms, org_dict):
    smile_dict = {}
    temp_smile = smile
    smile_list = []
    #构造 atom_dict remain_dict
    for atom in atoms:
        # atom_dict.update({atom: atom_dict.get(atom, 0) + 1})
        if len(atom) == 2 and len(temp_smile) > 2:
            while atom != temp_smile[:2]:
                if len(temp_smile) > 2:
                    smile_list.append(temp_smile[0])
                    temp_smile = temp_smile[1:]
                else:
                    break
        if len(atom) == 1 and len(temp_smile) > 1:
            while atom != temp_smile[0]:
                if len(temp_smile) > 1:
                    smile_list.append(temp_smile[0])
                    temp_smile = temp_smile[1:]
                else:
                    break
        patt = r""+atom
        smile_list.append(atom)
        temp_smile = re.sub(patt, "", temp_smile, 1)

    while len(temp_smile) > 0:
        smile_list.append(temp_smile[0])
        temp_smile = temp_smile[1:]

    # update org_dict
    for s in smile_list:
        smile_dict.update({s: smile_dict.get(s, 0) + 1})

    for org in smile_dict:
        org_dict.update({org: org_dict.get(org,0) + smile_dict.get(org)})

    return org_dict, smile_list

def replace_char(string,char,index):
    string = list(string)
    string[index] = char
    return ''.join(string)

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
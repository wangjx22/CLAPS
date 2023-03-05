import os
import numpy as np
from math import sqrt
from scipy import stats
import torch
from torch.nn.utils.rnn import pad_sequence
from creat_data_DC import smile_to_graph
from torch.utils.data import Dataset
from csv import reader
"""
预训练数据处理
"""
class MyDownTaskDataSet(Dataset):
    def __init__(self, smiles, label):
        self.smiles = smiles
        self.label = label

    def __getitem__(self, index):
        return self.smiles[index], self.label[index]

    def __len__(self):
        return len(self.label)



class TaskDataset(Dataset):
    def __init__(self, dataset='train', task='bbbp', pre_dict=None, ran_split=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TaskDataset, self).__init__()
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.task = task
        if os.path.isfile('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'):
            print('Pre-processed data found: {}, loading ...'.format('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'))
            if dataset == 'train':
                print('Pre-processed data found: {}, loading ...'.format('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'))
                self.data = torch.load('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt')
            if dataset == 'valid':
                print('Pre-processed data found: {}, loading ...'.format('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'))
                self.data = torch.load('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt')
            if dataset == 'test':
                print('Pre-processed data found: {}, loading ...'.format('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'))
                self.data = torch.load('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt')
            # self.data = torch.load('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt')
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt'))
            # creat_data(self.dataset)
            self.process(task, pre_dict, ran_split)
            self.data = torch.load('down_task/processed/' + ran_split + '/' + task + '_' + dataset + '.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, task, pre_dict, ran_split):
        vocab_dict = get_dict(pre_dict)
        PAD_IDX = vocab_dict.get('<pad>')

        smiles = []
        labels = []
        with open('down_task/processed/' + ran_split + '/' + task + '_train.csv', encoding='UTF-8')as raw_data:
            readers = reader(raw_data, delimiter=',')
            x = list(readers)
            i = 0
            for tem in x:
                if (i % 2) == 0:
                    smiles.append(tem)
                else:
                    t = []
                    for it in tem:
                        t.append(float(it))
                    labels.append(t)



                i = i + 1
        f1 = len(smiles)



        with open('down_task/processed/' + ran_split + '/' + task + '_valid.csv', encoding='UTF-8')as raw_data:
            readers = reader(raw_data, delimiter=',')
            x = list(readers)
            i = 0
            for tem in x:
                if (i % 2) == 0:
                    smiles.append(tem)
                else:
                    t = []
                    for it in tem:
                        t.append(float(it))
                    labels.append(t)
                i = i + 1

        f2 = len(smiles)


        with open('down_task/processed/' + ran_split + '/' + task + '_test.csv', encoding='UTF-8')as raw_data:
            readers = reader(raw_data, delimiter=',')
            x = list(readers)
            i = 0
            for tem in x:
                if (i % 2) == 0:
                    smiles.append(tem)
                else:
                    t = []
                    for it in tem:
                        t.append(float(it))
                    labels.append(t)
                i = i + 1

        smile2numb(task, smiles, labels, vocab_dict, PAD_IDX, f1, f2, ran_split)



        # splitter = 'scaffold'  # 'random'
        # featurizer = 'ECFP'
        # print(dc.__version__)
        # if task == 'BBBP':
        #     tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer=featurizer, splitter=splitter)
        # elif task == 'Tox21':
        #     tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer, splitter=splitter)
        # elif task == 'ClinTox':
        #     tasks, datasets, transformers = dc.molnet.load_clintox(featurizer=featurizer, splitter=splitter)
        # elif task == 'HIV':
        #     tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer, splitter=splitter)
        # elif task == 'BACE':
        #     tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer=featurizer, splitter=splitter)
        # elif task == 'SIDER':
        #     tasks, datasets, transformers = dc.molnet.load_sider(featurizer=featurizer, splitter=splitter)
        # elif task == 'MUV':
        #     tasks, datasets, transformers = dc.molnet.load_muv(featurizer=featurizer, splitter=splitter)
        # train, valid, test = datasets
        # save(self, train, 'train', task)
        # save(self, valid, 'valid', task)
        # save(self, test, 'test', task)

def save(dataset, kind, task):
    data_list = []
    label_list = []
    org_dict = {}
    count = 1
    lenth = len(dataset)
    for i in range(len(dataset)):
        smile = dataset.ids[i]
        label = dataset.y[i]
        if len(smile) < 8:
            continue
        atoms = smile_to_graph(smile)
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        org_dict, smiles = get_dict(smile, atoms, org_dict)

        data_list.append(smiles)
        label_list.append(label)
        print('{}/{}'.format(count, lenth))


    print('Saving to file.')
    with open("down_task/processed/" + task + '_' + kind + "_dict.txt", 'w') as f:
        for k, v in org_dict.items():
            f.write(str(k) + ' ' + str(v) + '\n')
        f.close()
    # save preprocessed data:
    # save preprocessed data:
    torch.save((data_list, label_list), 'down_task/processed/' + task + '_' + kind + '.pt')



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

def smile2numb(task, data, label, vocab_dict, PAD_IDX, f1, f2, ran_split):
    smile_seqs = []
    # 将一个batch的smiles 转化为 torch向量
    for smile in data:
        smile_seq = [int(vocab_dict.get(i, 0)) for i in smile]
        smile_seqs.append(torch.LongTensor(smile_seq))


    # 统一序列长度
    src_seq = pad_sequence(smile_seqs, batch_first=True, padding_value=PAD_IDX)
    train_data = MyDownTaskDataSet(smiles=src_seq[:f1], label=torch.FloatTensor(label[:f1]))
    torch.save(train_data, 'down_task/processed/' + ran_split + '/' + task + '_train.pt')

    valid_data = MyDownTaskDataSet(smiles=src_seq[f1:f2], label=torch.FloatTensor(label[f1:f2]))
    torch.save(valid_data, 'down_task/processed/' + ran_split + '/' + task + '_valid.pt')

    test_data = MyDownTaskDataSet(smiles=src_seq[f2:], label=torch.FloatTensor(label[f2:]))
    torch.save(test_data, 'down_task/processed/' + ran_split + '/' + task + '_test.pt')

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
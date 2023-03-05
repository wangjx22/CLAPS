from torch.utils.data import Dataset
import torch
class MyDataSet(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles

    def __getitem__(self, index):
        return self.smiles[index]

    def __len__(self):
        return len(self.smiles)

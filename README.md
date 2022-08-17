# CLAGVS


## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```


## Dataset
ZINC15 (http://zinc15.docking.org} and MoleculeNet

## Usage
### Train CLAGVS in ZINC15 dataset
```
python main_con.py
```

### Test in MoleculeNet

```
python main_downstream.py
python main_downstream_reg.py
```

### 


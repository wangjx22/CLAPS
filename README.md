# CLAPS


## 1. Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

The main environment for CLAPS can be installed with this command:
```
conda env create -f environment.yml
```




## 2. Usage
### 2.1 Pre-train CLAPS in ZINC15 dataset
Unzip the pretrain/processed/in-vitro.rar before pre-training
```
python main_con.py --datafile='in-vitro' --path='pretrain' --batch_size 1500 --mask_st="roulette" --dropout 0.2 --n_heads 4 --n_layers 3 --lr 0.001 --DNN_dim1 2048 --DNN_dim2 512
```

### 2.2 Test in MoleculeNet benchmark

```
python main_downstream.py
python main_downstream_reg.py
```
## 3. Dataset
ZINC15 (http://zinc15.docking.org) and MoleculeNet
### 3.1 classification

#### BBBP
The Blood–brain barrier penetration (BBBP) dataset comes from a recent study (PMID: 22612593) 
on the modeling and prediction of the barrier permeability. As a membrane separating circulating blood 
and brain extracellular fluid, the blood–brain barrier blocks most drugs, hormones and neurotransmitters. 
Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central 
nervous system. This dataset includes over 2,000 compounds on their permeability properties

load the dataset as follow:
```
deepchem.molnet.load_bbbp(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```


The meaning of the columns in the original document includes the following:

"Name" – compound name

"Smiles" – SMILES representation of molecular structure

"P_np" – binary representation of permeability/non permeability

#### Tox21
The “Toxicology in the 21st Century” (Tox21) initiative created a public database measuring toxicity 
of compounds, which has been used in the 2014 Tox21 Data Challenge (http://tripod.nih.gov/tox21/challenge/). 
This dataset contains qualitative toxicity measurements for 8,014 compounds on 12 different targets, 
including nuclear receptors and stress response pathways.

load the dataset as follow:
```
deepchem.molnet.load_tox21(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]
```

The meaning of the columns in the original document includes the following:

"Smiles" – SMILES representation of molecular structure

"NR-XXX" – activity result expressed by nuclear receptor signal of target XXX

"SR-XXX" - forced reaction activity result of target XXX

More information at https://tripod.nih.gov/tox21/challenge/data.jsp

#### ClinTox
The ClinTox dataset, introduced as part of this work, compares drugs approved 
by the FDA and drugs that have failed clinical trials for toxicity reasons (PMID: 27642066). 
The dataset includes two classification tasks for 1,491 drug compounds with known chemical structures: 
(1) clinical trial toxicity (or absence of toxicity) and (2) FDA approval status. List of FDA-approved 
drugs are compiled from the SWEETLEAD database (PMID: 24223973), and list of drugs that failed clinical 
trials for toxicity reasons are compiled from the Aggregate Analysis of ClinicalTrials.gov (AACT) 
database (http://www.ctti-clinicaltrials.org/aact-database).

load the dataset as follow:
```
deepchem.molnet.load_clintox(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```


The meaning of the columns in the original document includes the following:

"Smiles" – SMILES representation of molecular structure

"FDA_APPROVED" – FDA approval status

"CT_TOX" – Clinical trial results



#### HIV
The HIV dataset was introduced by the Drug Therapeutics Program (DTP) 
AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for 
over 40,000 compounds (http://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data). 
Screening results were evaluated and placed into three categories: confirmed inactive (CI), 
confirmed active (CA) and confirmed moderately active (CM). We further combine the latter two labels, 
making it a classification task between inactive (CI) and active (CA and CM).

load the dataset as follow:
```
deepchem.molnet.load_hiv(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```

The meaning of the columns in the original document includes the following:

"Smiles": SMILES representation of molecular structure

"Activity": three types of labels for filtering results: CI/CM/CA

"HIV_active": binary label of filter result: 1 (CA/CM) and 0 (CI)

 



#### SIDER
The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR) (PMID: 26481350). 
The version of the SIDER dataset in DeepChem has grouped drug side-effects into 
27 system organ classes following MedDRA (http://www.meddra.org/) classifications measured 
for 1,427 approved drugs (following previous usage).

load the dataset as follow:
```
deepchem.molnet.load_sider(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]

 
```

The meaning of the columns in the original document includes the following:

 "Smiles": SMILES representation of molecular structure

"Hepatological disorders": "Injury, toxicity and complexity of procedures: record the side effects of drugs. 
See http://sideeffects.embl.de/se/?page=98 for details on ADRs.



#### BACE
The BACE dataset provides quantitative (IC50) and qualitative (binary label) 
binding results for a set of inhibitors of human β-secretase 1 (BACE-1) (PMID: 27689393). 
All data are experimental values reported in scientific literature over the past decade, 
some with detailed crystal structures available. Which have 1,522 compounds.

load the dataset as follow:
```
deepchem.molnet.load_bace_regression(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator,

str]] = ['normalization'], reload: bool = True, data_dir: Optional[str]

= None, save_dir: Optional[str] = None, **kwargs) → Tuple[List[str],

Tuple[deepchem.data.datasets.Dataset, . . . ], List[transformers.Transformer]]
```

The meaning of the columns in the original document includes the following:

"Mol" – SMILES representation of molecular structure

"PIC50" - negative logarithm of IC50 affinity

"Class" – binary label of inhibitor


### 3.2 regression
#### Lipo 
Lipo is an important feature of drug molecules that affects both 
membrane permeability and solubility. This dataset, curated from ChEMBL database, 
provides experimental results of octanol/water distribution coefficient of 4,200 compounds.

load the dataset as follow:
```
deepchem.molnet.load_lipo(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]
```
 

The meaning of the columns in the original document includes the following:

"Smiles" – SMILES representing molecular structure

"Exp" – octanol/water distribution system (logD) of the measured compound, used as a label






#### ESOL
ESOL is a small dataset consisting of water solubility data for 1,128 
compounds (PMID: 15154768). The dataset has been used to train models that estimate 
solubility directly from chemical structures (as encoded in SMILES strings). 
Note that these structures don't include 3D coordinates, since solubility is a property of a 
molecule and not of its particular conformers.

load the dataset as follow:
```
deepchem.molnet.load_delaney(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```

The meaning of the columns in the original document includes the following:

"Compound ID" – compound name

"Smiles" – SMILES representation of molecular structure

"Measured log solubility in moles per litre" – the logarithmic water solubility of the complex, used as a label.





#### FreeSolv
The Free Solvation Database (FreeSolv) provides experimental and calculated hydration 
free energy of small molecules in water (PMID: 24928188). The calculated values are derived from 
alchemical free energy calculations using molecular dynamics simulations. 
We include the experimental values in the benchmark collection, and use calculated values for comparison.

load the dataset as follow:
```

deepchem.molnet.load_sampl(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]
```

The meaning of the columns in the original document includes the following:

"Iupac" – IUPAC of compound

"Smiles" – SMILES representation of molecular structure

"Expt" – the measured dissolution energy of the compound (unit: kcal/mol), used as a label

"Calc" – calculated dissolution energy of compound (unit: kcal/mol)
### 


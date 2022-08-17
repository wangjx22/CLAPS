## classification

### BBBP

BBBP是Blood-Brain-Barrier Penetration的缩写。

```
load_bbbp(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```

加载BBBP数据集。

blood-brain barrier penetration (BBBP)数据集为屏障渗透性建模和预测而设计。

由于膜将循环血液和脑的细胞间液分离，血脑屏障阻碍了大部分药物，激素，神经递质。因此透过该屏障是开发中枢神经系统药物的长期关注点。

该数据集包括2000个化合物的渗透性的二值标签。

该数据集推荐使用Scaffold分割。

原始的CSV文件包括如下列：

 “name” – 化合物名称

 “smiles” – 分子结构的SMILES表示

 “p_np” – 渗透/非-渗透的二值表示

### Tox21

```
oad_tox21(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]
```

加载Tox21数据集

“Toxicology in the 21st Century” (Tox21)的目的是公布化合物的毒性数据，曾在2014 Tox21 Data Challenge中使用。该数据集包括8千个化合物对12个不同靶点的毒性的定性数据，包括nuclear receptors and stress response pathways数据。

数据集推荐使用随机分割器。

 

原始的CSV文件包括如下列：

 “smiles” – 分子结构的SMILES表示

 “NR-XXX” – 核受体信号表示的活性结果

 “SR-XXX” – 强制反应活性结果

详见https://tripod.nih.gov/tox21/challenge/data.jsp

### ClinTox

```
load_clintox(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```

加载Clintox数据集

Clintox数据集比较FDA批准的和临床失败的药物的毒性。

该数据集为1491个已知结构的化合物完成两个分类任务：

1.临床试验的互性（或没有毒性)

2.FDA批准状态

FDA批准的药物编译自SWEETLEAD数据库，因为毒性原因导致临床失败的药物编译自Aggregate Analysis of ClinicalTrials.gov(AACT)数据库。

该数据集推荐使用随机分割器。

原始的CSV文件包括如下列：

 “smiles” – 分子结构的SMILES表示

 “FDA_APPROVED” – FDA批准状态

 “CT_TOX” – 临床试验结果



### HIV

```
load_hiv(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]
```

加载HIV数据集

HIV数据集由Drug Therapeutics Program (DTP) AIDS Antiviral Screen介绍，它测试了40000个化合物抑制HIV复制的能力。对筛选结果进行评估并分为3类：无活性（CI），有活性（CA)，中等活性（CM）。我们进一步组合后两个标签，分为非活性（CI）和活性（CA和CM）。

该数据集推荐使用Scaffold分割器。

原始的CSV文件包括如下列：

“smiles”: 分子结构的SMILES表示

“activity”: 筛选结果的三类标签: CI/CM/CA

“HIV_active”: 筛选结查的二值标签: 1 (CA/CM) and 0 (CI)



###  BACE

```
load_bace_classification(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP',

splitter: Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold',

transformers: List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator,

str]] = ['balancing'], reload: bool = True, data_dir: Optional[str] =

None, save_dir: Optional[str] = None, **kwargs)→Tuple[List[str], Tuple[

deepchem.data.datasets.Dataset, . . . ], List[transformers.Transformer]]
```

加载BACE数据集，分类标签。

带有分类标签的BACE dataset。

 



### SIDER

```
load_sider(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]]

= ['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]

 
```

加载SIDER数据集

Side Effect Resource (SIDER)是上市药品和不良反应（ADR）数据库。DeepChem中SIDER数据集将不良反应分为27个系统组织类，按照MedDRA分类，测量了1427个上市药品。

该数据集采用随机分割器。

原始的CSV文件包括如下列：

 “smiles”: 分子结构的SMILES表示

 “Hepatobiliary disorders” ~ “损伤,毒性和程序的复杂: 记录药物的副作用。请见 http://sideeffects.embl.de/se/?page=98 for details on ADRs.



### MUV

```
load_muv(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['balancing'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]

 
```

加载MUV数据集

Maximum Unbiased Validation (MUV) group是选自PubChem BioAssay的标杆数据集，使用提炼的最近邻分析。

MUV数据集包括17个挑战任务，约9万个化合物，它用来验证虚拟筛选技术。

原始的CSV文件包括如下列：

 “mol_id” – 化合物的PubChem CID

 “smiles” – 分子结构的SMILES表示

 “MUV-XXX” – 生物活性测定结果 (活性/非活性)



## regression

### BACE

```
load_bace_regression(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator,

str]] = ['normalization'], reload: bool = True, data_dir: Optional[str]

= None, save_dir: Optional[str] = None, **kwargs) → Tuple[List[str],

Tuple[deepchem.data.datasets.Dataset, . . . ], List[transformers.Transformer]]
```

加载BACE数据集，回归标签。

BACE数据集为BACE-1抑制剂提供量化的IC50和定性（二值）的结合结果。

所有的数据是过去几十年科学文献的数据值，可些有详细的晶体结构。一共提供了1522个化合物，以及IC50的回归标签。本数据集推荐使用分割构件。

原始数据CSV文件包含如下列：

 “mol” – 分子结构的SMILES表示

 “pIC50” - IC50亲和力的负对数

 “class” – 抑制剂的二值标签



### Lipo 

Lipo Datasets

load_lipo(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]

 

加载Lipophilicity数据集

亲脂性是药物分子的重要特征，它影响膜透过性和溶解度。Lipophilicity数据集，手工操作自ChEMBL数据库，提供了4200个化合物的octanol/water分配系数实验结果，

（pH 7.4时的logD值）。

数据集推荐使用Scaffold分割。

原始的CSV文件包括如下列：

 “smiles” – 表示分子结构的SMILES

 “exp” – 测得的化合物的 octanol/water 分配系统 (logD), 用作标签



###  PPB 

load_ppb(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]



### ESOL

Delaney Datasets

load_delaney(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset,

. . . ], List[transformers.Transformer]]

加载Delaney数据集

Delaney (ESOL)数据集是包含有1128个化合物的结构和水溶性数据的回归数据集。该数据集用于训练从分子结构（SMILES编码）预测溶解度的机器学习模型。

该数据集推荐使用Scaffold分割。

 

原始的CSV文件包括

“Compound ID” – 化合物名称

 “smiles” – 分子结构的SMILES表示

 “measured log solubility in mols per litre” – 合物物的对数水溶解度,用作标签。





### FreeSolv

SAMPL Datasets

load_sampl(featurizer: Union[deepchem.feat.base_classes.Featurizer, str] = 'ECFP', splitter:

Optional[Union[deepchem.splits.splitters.Splitter, str]] = 'scaffold', transformers:

List[Union[deepchem.molnet.load_function.molnet_loader.TransformerGenerator, str]] =

['normalization'], reload: bool = True, data_dir: Optional[str] = None, save_dir: Optional[

str] = None, **kwargs) → Tuple[List[str], Tuple[deepchem.data.datasets.Dataset, . . . ],

List[transformers.Transformer]]

加载SAMPL(FreeSolv)数据集

 

Free Solvation数据库，FreeSolv(SAMPL)，提供了小分子在水中的实验的和计算的氢自由能。该值来自alchemical free energy calculations，使用分子动力学模拟。实验值包括标杆集合。

该数据集推荐使用随机分割器。

原始的CSV文件包括如下列：

 “iupac” – 化合物的IUPAC

 “smiles” – 分子结构的SMILES表示

 “expt” – 测定的化合物的溶解能 (unit: kcal/mol),用作标签

 “calc” – 计算的化合物的溶解解能 (unit: kcal/mol)
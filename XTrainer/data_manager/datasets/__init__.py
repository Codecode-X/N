"""数据集具体实现类"""
from .build import DATASET_REGISTRY, build_dataset 

# -------分类数据集-------
from .Caltech101 import Caltech101 

# -------MCQ数据集-------
from .CocoMcq import CocoMcq

# -------Retrieval数据集-------
from .CocoRetrieval import CocoRetrieval
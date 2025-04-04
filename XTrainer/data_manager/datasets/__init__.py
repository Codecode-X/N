"""数据集具体实现类"""
from .build import DATASET_REGISTRY, build_dataset 

# -------分类数据集-------
from .Caltech101 import Caltech101 

# -------MCQ数据集-------
from .CocoMCQ import CocoMCQ

# -------Retrieval数据集-------
from .CocoRetrieval import CocoRetrieval
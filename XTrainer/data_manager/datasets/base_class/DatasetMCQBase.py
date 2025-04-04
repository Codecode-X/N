import random
from collections import defaultdict
from ..build import DATASET_REGISTRY
from .DatasetBase import DatasetBase
from utils import read_json, write_json
import os
from utils import check_isfile
import pandas as pd

@DATASET_REGISTRY.register()
class DatasetMCQBase(DatasetBase):
    """
    MCQ数据集类的基类。
    继承自 DatasetBase 类。

    对外访问属性 (@property) :
        - 父类 DatasetBase 的属性:
            - train (list): 带标签的训练数据。
            - val (list): 验证数据（可选）。
            - test (list): 测试数据。
        - 当前读取：None

    对内访问属性:
        - 父类 DatasetBase 的属性:
            - num_shots (int): 少样本数量。
            - seed (int): 随机种子。
            - p_trn (float): 训练集比例。
            - p_val (float): 验证集比例。
            - p_tst (float): 测试集比例。
            - dataset_dir (str): 数据集目录。
        - 当前读取：
            - csv_file (str): cfg.DATASET.CSV_FILE: 标注文件路径 | 例如：/root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
            - num_choices (int): cfg.DATASET.NUM_CHOICES: 答案选项数量。 例如：4

    基本方法:
        - 实现 DatasetBase 类的抽象方法/接口：
            - read_split: 读取数据分割文件。
            - save_split: 保存数据分割文件。
            - generate_fewshot_dataset: 生成小样本数据集（通常用于训练集）。

    抽象方法/接口 (需要具体数据集子类实现):
        - read_and_split_data: 读取数据并分割为 train, val, test 数据集 (需要根据每个MCQ数据集的格式自定义)。
    
    """

    def __init__(self, cfg):
        """ 
        初始化数据集的基本属性

        参数:
            - cfg (CfgNode): 配置。

        配置
            - 当前读取：
                - csv_path (str): cfg.DATASET.CSV_PATH: 标注文件路径 | 例如：/root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
                - num_choices (int): cfg.DATASET.NUM_CHOICES: 答案选项数量。 例如：4
            - 父类读取：
                - dataset_dir (str): cfg.DATASET.DATASET_DIR: 数据集目录。| 例如：/root/autodl-tmp/NegBench/images/MCQ
                - num_shots (int): cfg.DATASET.NUM_SHOTS: 少样本数量 | -1 表示使用全部数据，0 表示 zero-shot，1 表示 one-shot，以此类推。
                - seed (int): cfg.SEED: 随机种子。
                - p_trn (float): cfg.DATASET.SPLIT[0]: 训练集比例。
                - p_val (float): cfg.DATASET.SPLIT[1]: 验证集比例。
                - p_tst (float): cfg.DATASET.SPLIT[2]: 测试集比例。
        
        主要步骤：
            1. 读取新增配置。
            2. 调用父类 DatasetBase 构造方法：
                1. 读取配置。
                2. 读取数据并分割为 train, val, test 数据集。（待子类实现 get_data 和 get_fewshot_data 方法）
                3. 如果 num_shots >= 0，则从 train 和 val 中进行少样本采样，生成少样本的 train 和 val 数据集。
        """
        # ---读取配置---
        self.csv_file = cfg.DATASET.CSV_FILE 
        self.num_choices = cfg.DATASET.NUM_CHOICES

        # ---调用父类构造方法，获取 self.train, self.val, self.test 等属性---
        super().__init__(cfg)  # 调用父类构造方法

    # -----------------------父类 DatasetBase 要求子基类实现的抽象方法-----------------------

    def read_split(self, split_file):
        """
        读取数据分割文件 (实现父类的抽象方法)。
        
        参数：
            - split_file (str): (新保存/读取)分割文件路径。
        
        返回：
            - 训练、验证和测试数据 (MCQDatum 对象列表类型)。
        """
        def _convert(items):
            out = []
            for impath, num_choices, choices, correct_answer, correct_answer_type in items:
                item = MCQDatum(impath=impath, num_choices=num_choices, choices=choices, correct_answer=correct_answer, correct_answer_type=correct_answer_type)
                out.append(item)
            return out

        print(f"Reading split from {split_file}")  # 打印读取分割文件的信息
        split = read_json(split_file)  # 读取 JSON 文件
        train = _convert(split["train"])  # 转换训练数据
        val = _convert(split["val"])  # 转换验证数据
        test = _convert(split["test"])  # 转换测试数据

        return train, val, test  # 返回训练、验证和测试数据 (MCQDatum 对象列表类型)
    

    def save_split(self, train, val, test, split_file):
        """
        保存数据分割文件 (实现父类的抽象方法)。
        
        参数：
            - train (list): 训练数据集。
            - val (list): 验证数据集。
            - test (list): 测试数据集。
            - split_file (str): 数据分割文件路径。

        返回：
            - None
        """
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                num_choices = item.num_choices
                choices = item.choices
                correct_answer = item.correct_answer
                correct_answer_type = item.correct_answer_type
                out.append((impath, num_choices, choices, correct_answer, correct_answer_type))
            return out
        train = _extract(train)  # 提取训练数据
        val = _extract(val)  # 提取验证数据
        test = _extract(test)  # 提取测试数据

        split = {"train": train, "val": val, "test": test}  # 创建分割字典

        write_json(split, split_file)  # 写入 JSON 文件
        print(f"Saved split to {split_file}")  # 打印保存分割文件的信息
    

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """生成小样本数据集，每个类别仅包含少量图像 (实现父类的抽象方法)。

        参数:
            - dataset (list): 包含 MCQDatum 对象的列表。
            - num_shots (int): 每个类别采样的实例数量。| 默认 -1 即直接返回原始数据源
            - repeat (bool): 是否在需要时重复图像（默认：False）。
        
        返回:
            - fewshot_dataset (list): 包含少量图像的数据集。
        """
        if num_shots <= 0: # 不进行少样本采样
            super().generate_fewshot_dataset(dataset, num_shots, repeat)

        print(f"正在创建一个 {num_shots}-shot 数据集....")

        # 按照 correct_answer_type 分组
        grouped_data = defaultdict(list)
        for item in dataset:
            grouped_data[item.correct_answer_type].append(item)

        fewshot_dataset = []
        for correct_answer_type, items in grouped_data.items():
            if len(items) <= num_shots:
                if repeat:
                    # 如果样本不足且允许重复，则重复采样
                    fewshot_dataset.extend(
                        items * (num_shots // len(items)) +  # items 需要被完整重复的次数
                        items[:num_shots % len(items)]  # 不能整除的剩余部分
                    )
                else:
                    fewshot_dataset.extend(items)  # 不足时直接使用全部样本
            else:
                # 随机采样 num_shots 个样本
                fewshot_dataset.extend(random.sample(items, num_shots))

        print(f"少样本数据集创建完成，共包含 {len(fewshot_dataset)} 个样本。")
        return fewshot_dataset

    
    # -------------------具体数据集子类实现 - 抽象方法/接口-------------------

    def read_and_split_data(self):
        """
        读取 csv 文件的数据并分割为 train, val, test 数据集 (需要根据每个MCQ数据集的格式自定义)

        需要返回：
            - 训练、验证和测试数据 (MCQDatum 对象列表类型)。
        """
        raise NotImplementedError("子类需要实现 read_and_split_data 方法") 

        

# -------------------辅助类 和 函数-------------------


class MCQDatum:
    """MCQ数据实例类，定义了基本属性。

    参数:
        - impath (str): 图像路径。
        - num_choices (int): 答案选项数量。
        - choices (list<str>): 答案选项文本列表。
        - correct_answer (int): 正确答案索引。
        - correct_answer_template (str): 正确答案模板类型。(negation, hybrid, positive)
    """

    def __init__(self, impath, num_choices, choices, correct_answer, correct_answer_type):
        """初始化数据实例。"""
        # 确保图像路径是字符串类型&检查图像路径是否是有效文件
        assert isinstance(impath, str) and check_isfile(impath)
        self._impath = impath
        self._num_choices = num_choices
        self._choices = choices
        self._correct_answer = correct_answer
        self._correct_answer_type = correct_answer_type

    @property
    def impath(self):
        return self._impath
    
    @property
    def num_choices(self):
        return self._num_choices
    
    @property
    def choices(self):
        return self._choices
    
    @property
    def correct_answer(self):
        return self._correct_answer
    
    @property
    def correct_answer_type(self):
        return self._correct_answer_type
""" 定义了视频数据集类：CsvVideoCaptionDataset 和 CsvVideoMCQDataset """

import pandas as pd
from torch.utils.data import Dataset
import torch
import pandas as pd
import torch
from PIL import Image

class CsvMCQDataset(Dataset):
    """
    用于多选题 (MCQ) 任务的 CSV 数据集类
    """

    def __init__(self, csv_file, transforms, num_answers=4, path="image_path", tokenizer=None):
        """
        初始化 CsvMCQDataset 类

        __getitem__ 方法返回数据集中的 图像、答案选项文本、正确答案索引、模板类型

        Args:
            csv_file (str): CSV 文件路径
            transforms (callable): 用于图像的变换函数
            num_answers (int): 答案选项的数量
            path (str): 图像路径所在的列名
            tokenizer (callable): 用于文本的分词器（训练时需要传入）
        """
        self.df = pd.read_csv(csv_file, sep=',')  # 读取 CSV 文件
        self.transforms = transforms  # 图像变换函数
        self.num_answers = num_answers  # 答案选项数量
        self.path = path  # 图像路径列名
        self.tokenizer = tokenizer  # 文本分词器

    def __len__(self):
        # 返回数据集的大小
        return len(self.df)

    def __getitem__(self, idx):
        # 根据索引获取图像和多选题数据
        row = self.df.iloc[idx]  # 获取当前行数据

        # 图像
        image_path = row[self.path]  # 获取图像路径
        images = Image.open(image_path) # 加载图像
        images = self.transforms(images)  # 变换图像

        # 文本
        choices = [row[f"caption_{i}"] for i in range(self.num_answers)]  # 获取所有答案 (文本) 选项
        correct_answer = row["correct_answer"]  # 获取正确答案索引 (0, 1, 2, 3)
        correct_answer_template = row["correct_answer_template"]  # 获取正确答案模板类型 (negation, hybrid, positive)
        # 如果提供了分词器，对答案选项进行分词
        if self.tokenizer is not None: 
            choices = [self.tokenizer([str(caption)])[0] for caption in choices]
            choices = torch.stack(choices)  # 将分词后的答案选项堆叠为张量 (num_answers, max_seq_len)

        return images, choices, correct_answer, correct_answer_template  # 返回图像、答案选项文本、正确答案索引和模板类型

class CsvImageCaptionDataset(Dataset):
    """
    用于 图像描述或检索任务 的数据集类

    __getitem__ 方法返回数据集中的 图像 和 描述文本

    Args:
        csv_file (string): 包含图像路径和描述的 CSV 文件路径
        transforms (callable): 应用于样本的图像变换
        sep (string): CSV 文件的分隔符
        img_key (string): 图像文件路径所在的列名
        caption_key (string): 描述文本所在的列名
    """
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath', caption_key='captions'):
        # 读取 CSV 文件
        self.df = pd.read_csv(csv_file, sep=sep)
        # 图像变换函数
        self.transforms = transforms
        # 图像路径列名
        self.img_key = img_key
        # 描述文本列名
        self.caption_key = caption_key

    def __len__(self):
        # 返回数据集的大小
        return len(self.df)

    def __getitem__(self, idx):
        # 获取图像路径
        image_path = self.df.iloc[idx][self.img_key]
        # 加载并变换图像
        images = self.transforms(Image.open(image_path))
        # 获取描述文本并解析为列表
        captions = eval(self.df.iloc[idx][self.caption_key]) # eval() 将 字符串 解析为 Python 数据结构
        # 返回图像和描述文本
        return images, captions


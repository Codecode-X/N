from .build import DATASET_REGISTRY
from .base_class.DatasetMcqBase import DatasetMcqBase, MCQDatum
import pandas as pd

@DATASET_REGISTRY.register()
class CocoMcq(DatasetMcqBase):
    """
    CocoMCQ 数据集类。
    继承自 DatasetMCQBase 类。

    对外访问属性 (@property) :
        - 父类 DatasetClsBase 的属性：
            - 父类 DatasetBase 的属性：
                - train (list): 带标签的训练数据。
                - val (list): 验证数据（可选）。
                - test (list): 测试数据。
        - 当前读取：None

    对内访问属性：
        - 父类 DatasetClsBase 的属性：
            - 父类 DatasetBase 的属性：
                - num_shots (int): 少样本数量。
                - seed (int): 随机种子。
                - p_trn (float): 训练集比例。
                - p_val (float): 验证集比例。
                - p_tst (float): 测试集比例。
                - dataset_dir (str): 数据集目录。
            - csv_file (str): cfg.DATASET.CSV_FILE: 标注文件路径 | 例如：/root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
            - num_choices (int): cfg.DATASET.NUM_CHOICES: 答案选项数量。 例如：4
        - 当前读取：None

    实现 DatasetMCQBase 类的抽象方法/接口：
        - read_and_split_data: 读取数据并分割为 train, val, test 数据集 (需要根据每个MCQ数据集的格式自定义)。
    """

    def read_and_split_data(self):
        """
        读取数据并分割为 train, val, test 数据集 (需要根据每个MCQ数据集的格式自定义)

        csv 文件格式:
        image_path	correct_answer	caption_0	caption_1	caption_2	caption_3	correct_answer_template	
        data/coco/images/val2017/000000397133.jpg	0	This image features a knife, but no car is present.	A car is present in this image, but there is no knife.	This image features a car	This image does not feature a knife.	hybrid	
        data/coco/images/val2017/000000397133.jpg	0	A chair is not present in this image.	This image shows a chair, but no spoon is present.	A chair is present in this image.	A spoon is not included in this image.	negative	
        data/coco/images/val2017/000000397133.jpg	0	A person is present in this image, but there's no fork.	This image shows a fork, with no person in sight.	A fork is shown in this image.	No person is present in this image.	hybrid	

        需要返回：
            - 训练、验证和测试数据 (Datum 对象列表类型)。
        """
        df = pd.read_csv(self.csv_file, sep=',')  # 读取 CSV 文件

        # 将每一行转换为 MCQDatum 对象
        data = []
        for _, row in df.iterrows():
            impath = row['image_path']
            num_choices = self.num_choices  # 每行 4 个选项
            choices = [row[f'caption_{i}'] for i in range(num_choices)]
            correct_answer = row['correct_answer']
            correct_answer_type = row['correct_answer_template']
            datum = MCQDatum(impath, num_choices, choices, correct_answer, correct_answer_type)
            data.append(datum)

        # 按照比例分割数据集
        num_train = int(len(data) * self.p_trn)
        num_val = int(len(data) * self.p_val)
        train = data[:num_train]
        val = data[num_train:num_train + num_val]
        test = data[num_train + num_val:]

        return train, val, test
from .build import DATASET_REGISTRY
from .base_class.DatasetRetriBase import DatasetRetriBase, RetrievalDatum
import pandas as pd

@DATASET_REGISTRY.register()
class CocoRetrieval(DatasetRetriBase):
    """
    CocoRetrieval 数据集类。
    继承自 DatasetRetrievalBase 类。

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
            - csv_file (str): cfg.DATASET.CSV_FILE: 标注文件路径 | 例如：/root/autodl-tmp/NegBench/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
            - 当前读取：None

    实现 DatasetRetrievalBase 类的抽象方法/接口：
        - read_and_split_data: 读取数据并分割为 train, val, test 数据集 (需要根据每个Retrieval数据集的格式自定义)。
    """

    def read_and_split_data(self):
        """
        读取数据并分割为 train, val, test 数据集 (需要根据每个Retrieval数据集的格式自定义)

        csv 文件格式:
        positive_objects	negative_objects	filepath	image_id	captions
        ['person', 'bottle', 'cup', 'knife', 'spoon', 'bowl', 'broccoli', 'carrot', 'dining table', 'oven', 'sink']	['chair', 'fork', 'car']	data/coco/images/val2017/000000397133.jpg	397133	['A man in a kitchen is making pizzas, but there is no chair in sight.', 'A man in an apron stands in front of an oven with pans and bakeware nearby, without a chair in sight.', 'No fork is present, but a baker is busy working in the kitchen, rolling out dough.', 'A person stands by a stove in the kitchen, but a fork is noticeably absent.', "At the table, pies are being crafted, while a person stands by a wall adorned with pots and pans, and noticeably, there's no fork."]
        ['banana', 'orange', 'chair', 'potted plant', 'dining table', 'oven', 'sink', 'refrigerator']	['person', 'cup', 'bottle']	data/coco/images/val2017/000000037777.jpg	37777	['No cup can be seen; however, the dining table near the kitchen is set with a bowl of fruit.', 'No cup is visible in the image, but the small kitchen is equipped with various appliances and a table.', 'The kitchen is clean and empty, with no person in sight.', 'In this pristine white kitchen and dining area, no bottle is in sight.', 'A kitchen scene featuring a bowl of fruit on the table, but noticeably absent is a bottle.']
        ['person', 'traffic light', 'umbrella', 'handbag', 'cup']	['chair', 'car', 'dining table']	data/coco/images/val2017/000000252219.jpg	252219	['A person with a shopping cart is on a city street, with no car in sight.', 'No dining table is visible in the image, which shows city dwellers passing by a homeless man begging for cash.', 'On a city street, people walk past a homeless man begging, with no chair in sight.', 'On the street, a homeless man holds a cup and stands beside a shopping cart, with no sign of a car.', 'People walk by a homeless person standing on the street, with no cars in the surrounding area.']

        需要返回：
            - 训练、验证和测试数据 (Datum 对象列表类型)。
        """
        df = pd.read_csv(self.csv_file, sep=',')  # 读取 CSV 文件

        # 将每一行转换为 RetrievalDatum 对象
        data = []
        for _, row in df.iterrows():
            impath = row['image_path']
            captions = row['captions']
            datum = RetrievalDatum(impath, captions)
            data.append(datum)

        # 按照比例分割数据集
        num_train = int(len(data) * self.p_trn)
        num_val = int(len(data) * self.p_val)
        train = data[:num_train]
        val = data[num_train:num_train + num_val]
        test = data[num_train + num_val:]

        return train, val, test
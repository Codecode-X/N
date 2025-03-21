import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    """CIFAR10 for SSL. 半监督学习的 CIFAR10 数据集。

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar10"  # 数据集目录名称

    def __init__(self, cfg):
        # 获取训练数据目录和测试数据目录
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT)) # 获取数据集的根目录路径
        self.dataset_dir = osp.join(root, self.dataset_dir)  # 拼接完整数据集路径
        train_dir = osp.join(self.dataset_dir, "train")  # 训练数据目录
        test_dir = osp.join(self.dataset_dir, "test")  # 测试数据目录

        assert cfg.DATASET.NUM_LABELED > 0  # 确保有标注数据

        # 读取训练数据（有标签、无标签、验证集）
        train_x, train_u, val = self._read_data_train(
            train_dir, cfg.DATASET.NUM_LABELED, cfg.DATASET.VAL_PERCENT
        )
        # 读取测试数据
        test = self._read_data_test(test_dir)

        # 如果配置为 将所有数据 作为 无标签数据
        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x  # 将 有标签数据 加入 无标签数据

        # 如果验证集为空，则设置为 None
        if len(val) == 0:
            val = None

        # 调用父类构造函数，初始化数据集
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def _read_data_train(self, data_dir, num_labeled, val_percent):
        """读取训练数据。
        参数:
            data_dir (str): 数据目录路径。
            num_labeled (int): 有标签数据数量。
            val_percent (float): 验证集比例。
        返回:
            有标签数据列表、无标签数据列表、验证集数据列表。
        """
        # 获取数据目录中的类别名称列表
        class_names = listdir_nohidden(data_dir)
        class_names.sort()  # 按字母顺序排序
        num_labeled_per_class = num_labeled / len(class_names)  # 每类的标注数据数量
        items_x, items_u, items_v = [], [], []  # 初始化有标签、无标签和验证集列表

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)  # 每个类别的目录路径
            imnames = listdir_nohidden(class_dir)  # 获取类别目录中的图片名称列表

            # 按照 Oliver 等人的方法划分训练和验证集
            num_val = math.floor(len(imnames) * val_percent)  # 验证集数量
            imnames_train = imnames[num_val:]  # 训练集图片名称
            imnames_val = imnames[:num_val]  # 验证集图片名称

            random.shuffle(imnames_train)  # 在划分后对训练集图片随机打乱

            # 训练集图片，划分为有标签和无标签：有标签数量为 num_labeled_per_class，其余为无标签
            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)  # 图片路径
                item = Datum(impath=impath, label=label)  # 创建 Datum 对象

                if (i + 1) <= num_labeled_per_class:
                    items_x.append(item)  # 添加到有标签数据列表
                else:
                    items_u.append(item)  # 添加到无标签数据列表

            # 验证集图片，有标签
            for imname in imnames_val:
                impath = osp.join(class_dir, imname)  # 验证集图片路径
                item = Datum(impath=impath, label=label)  # 创建 Datum 对象
                items_v.append(item)  # 添加到验证集列表

        return items_x, items_u, items_v  # 返回有标签、无标签和验证集数据

    def _read_data_test(self, data_dir):
        """读取测试数据。
        参数:
            data_dir (str): 数据目录路径。
        返回：  
            测试数据列表。
        """
        class_names = listdir_nohidden(data_dir) # 获取测试数据目录中的类别名称列表
        class_names.sort()  # 按字母顺序排序
        items = []  # 初始化测试数据列表

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)  # 每个类别的目录路径
            imnames = listdir_nohidden(class_dir)  # 获取类别目录中的图片名称列表

            for imname in imnames:
                impath = osp.join(class_dir, imname)  # 图片路径
                item = Datum(impath=impath, label=label)  # 创建 Datum 对象
                items.append(item)  # 添加到测试数据列表

        return items  # 返回测试数据列表


@DATASET_REGISTRY.register()
class CIFAR100(CIFAR10):
    """CIFAR100 for SSL.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """

    dataset_dir = "cifar100"

    def __init__(self, cfg):
        super().__init__(cfg)

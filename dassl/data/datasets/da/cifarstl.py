import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class CIFARSTL(DatasetBase):
    """CIFAR-10 和 STL-10 数据集。

    CIFAR-10:
        - 包含 60,000 张 32x32 的彩色图像。
        - 分为 10 个类别，每个类别有 6,000 张图像。
        - 50,000 张训练图像和 10,000 张测试图像。
        - 数据集网址：https://www.cs.toronto.edu/~kriz/cifar.html。

    STL-10:
        - 包含 10 个类别：飞机、鸟、汽车、猫、鹿、狗、马、
          猴子、船、卡车。
        - 图像大小为 96x96 像素，彩色。
        - 每个类别有 500 张训练图像（10 个预定义折叠），
          800 张测试图像。
        - 数据集网址：https://cs.stanford.edu/~acoates/stl10/。

    参考文献:
        - Krizhevsky. Learning Multiple Layers of Features
          from Tiny Images. 技术报告。
        - Coates 等人。An Analysis of Single Layer Networks in
          Unsupervised Feature Learning. AISTATS 2011.
    """

    dataset_dir = "cifar_stl"  # 数据集的根目录名称
    domains = ["cifar", "stl"]  # 数据集的域名称

    def __init__(self, cfg):
        # 获取数据集的根目录路径
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # 检查输入的源域和目标域是否有效
        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        # 读取源域的训练数据
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        # 读取目标域的无标签训练数据
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="train")
        # 读取目标域的测试数据
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")

        # 调用父类构造函数，初始化数据集
        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, split="train"):
        """读取数据（训练数据或测试数据）。
        参数:
            input_domains (list): 输入的域名称列表。
            split (str): 数据子集名称（"train" 或 "test" 或 "val"）。
        返回:
            数据项列表。
        """
        # 初始化数据项列表 | 存放格式：[Datum, Datum, ...] | 不同域的数据项混合在一起
        items = []

        # 遍历输入的域
        for domain, dname in enumerate(input_domains):
            # 获取当前域的分割数据目录
            data_dir = osp.join(self.dataset_dir, dname, split)
            # 列出数据目录中的类别名称
            class_names = listdir_nohidden(data_dir)

            # 遍历每个类别
            for class_name in class_names:
                # 获取类别目录路径
                class_dir = osp.join(data_dir, class_name)
                # 列出类别目录中的所有图像文件
                imnames = listdir_nohidden(class_dir)
                # 从类别名称中提取标签（假设类别名称以数字开头）
                label = int(class_name.split("_")[0])

                # 遍历每张图像
                for imname in imnames:
                    # 获取图像路径
                    impath = osp.join(class_dir, imname)
                    # 创建一个数据项
                    item = Datum(impath=impath, label=label, domain=domain)
                    # 将数据项添加到列表中
                    items.append(item)

        # 返回所有数据项
        return items

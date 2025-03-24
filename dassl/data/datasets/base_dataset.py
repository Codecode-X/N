import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown

from dassl.utils import check_isfile


class DatasetBase:
    """统一的数据集类，用于以下任务：
    1) 域适应
    2) 域泛化
    3) 半监督学习

    参数:
        train_x (list): 带标签的训练数据。
        train_u (list): 无标签的训练数据（可选）。
        val (list): 验证数据（可选）。
        test (list): 测试数据。
    """
    dataset_dir = ""  # 数据集存储的目录
    domains = []  # 所有域的名称列表

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        """ 初始化数据集的基本属性：
        1) 带标签的训练数据
        2) 无标签的训练数据（可选）
        3) 验证数据（可选）
        4) 测试数据
        5) 类别数量
        6) 标签到类别名称的映射
        7) 类别名称列表
        """
        # 初始化带标签的训练数据
        self._train_x = train_x
        # 初始化无标签的训练数据（可选）
        self._train_u = train_u
        # 初始化验证数据（可选）
        self._val = val
        # 初始化测试数据
        self._test = test
        # 获取类别数量
        self._num_classes = self.get_num_classes(train_x)
        # 获取标签到类别名称的映射和类别名称列表
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        """返回带标签的训练数据"""
        return self._train_x

    @property
    def train_u(self):
        """返回无标签的训练数据"""
        return self._train_u

    @property
    def val(self):
        """返回验证数据"""
        return self._val

    @property
    def test(self):
        """返回测试数据"""
        return self._test

    @property
    def lab2cname(self):
        """返回标签到类别名称的映射"""
        return self._lab2cname

    @property
    def classnames(self):
        """返回类别名称列表"""
        return self._classnames

    @property
    def num_classes(self):
        """返回类别数量"""
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """统计类别数量。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        """
        # 使用集合存储唯一的标签
        label_set = set()
        for item in data_source:
            # 将每个数据实例的标签加入集合
            label_set.add(item.label)
        # 返回最大标签值加 1 作为类别数量
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """获取标签到类别名称的映射（字典）。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        返回:
            mapping (dict): 标签到类别名称的映射。
            classnames (list): 类别名称列表。
        """
        # 获取数据集中所有的 类别标签和类别名称 的映射关系 mapping
        container = set() # 去重 set
        for item in data_source: 
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        # 获取所有标签并排序
        labels = list(mapping.keys())
        labels.sort()
        # 根据排序后的标签生成类别名称列表
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        """检查输入的源域和目标域是否有效。
        参数:
            source_domains (list): 源域列表。
            target_domains (list): 目标域列表。
        返回:
            None | ValueError: 如果输入域无效，则引发异常。
        """
        # 确保源域列表不为空
        assert len(source_domains) > 0, "source_domains (list) is empty"
        # 确保目标域列表不为空
        assert len(target_domains) > 0, "target_domains (list) is empty"
        # 检查源域和目标域是否有效
        self.is_input_domain_valid(source_domains) # 验证所有源域是否在定义的域列表 (self.domains) 中。
        self.is_input_domain_valid(target_domains) # 验证所有目标域是否在定义的域列表 (self.domains) 中。

    def is_input_domain_valid(self, input_domains):
        """验证输入域是否在定义的域列表中。"""
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        """下载数据并解压。

        参数:
            url (str): 数据下载链接。
            dst (str): 下载文件的目标路径。
            from_gdrive (bool): 是否从 Google Drive 下载。
        返回:
            None
        """
        # 如果目标路径的父目录不存在，则创建
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            # 使用 gdown 下载文件
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        # 解压 zip 文件
        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        # 解压 tar 文件
        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        # 解压 tar.gz 文件
        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        """生成小样本数据集（通常用于训练集）。

        此函数用于在小样本学习设置中评估模型，
        每个类别仅包含少量图像。

        参数:
            data_sources: 数据源列表 | 每个数据源 data_source 是一个 Datum 对象列表。
            num_shots (int): 每个类别采样的实例数量。| 默认 -1 即直接返回原始数据源
            repeat (bool): 是否在需要时重复图像（默认：False）。
        返回:
            list | Datum: 如果只有一个数据源，则返回单个数据集。
        """
        # 如果 num_shots 小于 1，直接返回原始数据源
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        # 遍历每个数据源
        for data_source in data_sources: 
            # 按标签分割数据集
            tracker = self.split_dataset_by_label(data_source) # 将数据集（Datum 对象列表）按类别标签分组存储在字典中。
            dataset = [] 

            for label, items in tracker.items():
                # 如果样本数量足够，随机采样 num_shots 个样本
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots) # 每个 类别 随机采样 num_shots 个样本 (num_shots)
                else:
                    # 如果样本不足，根据 repeat 参数决定是否重复采样
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items # 如果不重复采样，直接使用所有样本
                # 将采样的样本加入数据集
                dataset.extend(sampled_items) # 包含 当前数据源 的 所有类别的 num_shots 个样本 (类别数*num_shots)

            output.append(dataset) # 包含 所有数据源 的 所有类别的 num_shots 个样本 (数据源数*类别数*num_shots)

        # 如果只有一个数据源，返回单个数据集
        if len(output) == 1:
            return output[0] # 该数据源 的 所有类别的 num_shots 个样本 (类别数*num_shots)

        return output

    def split_dataset_by_label(self, data_source):
        """按 类别标签 将数据集（Datum 对象列表）分组存储在字典中。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        返回:
            defaultdict: 每个类别标签对应的数据实例列表。
        """
        output = defaultdict(list)

        for item in data_source:
            # 根据标签将数据实例分组
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """按 域 将数据集（Datum 对象列表）分组存储在字典中。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        返回:
            defaultdict: 每个域对应的数据实例列表。
        """
        output = defaultdict(list)

        for item in data_source:
            # 根据域将数据实例分组
            output[item.domain].append(item)

        return output


class Datum:
    """数据实例类，定义了基本属性。

    参数:
        impath (str): 图像路径。
        label (int): 类别标签。
        domain (int): 域标签。
        classname (str): 类别名称。
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        """初始化数据实例。"""
        # 确保图像路径是字符串类型
        assert isinstance(impath, str)
        # 检查图像路径是否是有效文件
        assert check_isfile(impath)

        # 初始化图像路径
        self._impath = impath
        # 初始化类别标签
        self._label = label
        # 初始化域标签
        self._domain = domain
        # 初始化类别名称
        self._classname = classname

    @property
    def impath(self):
        """返回图像路径"""
        return self._impath

    @property
    def label(self):
        """返回类别标签"""
        return self._label

    @property
    def domain(self):
        """返回域标签"""
        return self._domain

    @property
    def classname(self):
        """返回类别名称"""
        return self._classname
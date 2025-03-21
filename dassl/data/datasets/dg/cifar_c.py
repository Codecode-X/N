import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

AVAI_C_TYPES = [ # 可用的 CIFAR-C 类型
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


@DATASET_REGISTRY.register()
class CIFAR10C(DatasetBase):
    """CIFAR-10 -> CIFAR-10-C.

    数据集链接：https://zenodo.org/record/2535967#.YFwtV2Qzb0o

    数据统计:
        - 2 个域：正常的 CIFAR-10 和被破坏的 CIFAR-10
        - 10 个类别

    参考文献:
        - Hendrycks 等人。基准测试神经网络对常见破坏和扰动的鲁棒性。ICLR 2019.
    """

    dataset_dir = ""  # 数据集目录
    domains = ["cifar10", "cifar10_c"]  # 定义两个域：正常 CIFAR-10 和破坏的 CIFAR-10

    def __init__(self, cfg):
        # 获取数据集的根目录
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = root

        # 检查输入的域是否正确
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]  # 源域
        target_domain = cfg.DATASET.TARGET_DOMAINS[0]  # 目标域
        assert source_domain == self.domains[0]  # 确保源域是 cifar10
        assert target_domain == self.domains[1]  # 确保目标域是 cifar10_c

        # 获取破坏类型和破坏等级
        c_type = cfg.DATASET.CIFAR_C_TYPE
        c_level = cfg.DATASET.CIFAR_C_LEVEL

        # 如果未指定破坏类型，抛出错误
        if not c_type: raise ValueError("请在配置文件中指定 DATASET.CIFAR_C_TYPE")
        # 确保破坏类型在可用类型列表中
        assert (c_type in AVAI_C_TYPES), f'C_TYPE 应属于 {AVAI_C_TYPES}, 但得到了 "{c_type}"'
        # 确保破坏等级在 1 到 5 之间
        assert 1 <= c_level <= 5

        # 定义训练集和测试集的目录
        train_dir = osp.join(self.dataset_dir, source_domain, "train")
        test_dir = osp.join(self.dataset_dir, target_domain, c_type, str(c_level))

        # 如果测试集目录不存在，抛出错误
        if not osp.exists(test_dir):
            raise ValueError

        # 读取训练集和测试集数据
        train = self._read_data(train_dir)
        test = self._read_data(test_dir)

        # 调用父类构造函数，初始化数据
        super().__init__(train_x=train, test=test)

    def _read_data(self, data_dir):
        """读取数据。
        参数：
            data_dir (str): 数据目录路径。
        返回：
            数据项列表。
        """
        class_names = listdir_nohidden(data_dir) # 获取类别名称
        class_names.sort()  # 对类别名称排序
        items = []  # 存储数据项

        # 遍历每个类别
        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)  # 类别目录
            imnames = listdir_nohidden(class_dir)  # 获取类别目录中的图片名称

            # 遍历每张图片
            for imname in imnames:
                impath = osp.join(class_dir, imname)  # 图片路径
                item = Datum(impath=impath, label=label, domain=0)  # 创建数据项
                items.append(item)  # 添加到列表中

        return items  # 返回数据项列表


@DATASET_REGISTRY.register()
class CIFAR100C(CIFAR10C):
    """CIFAR-100 -> CIFAR-100-C.

    数据集链接：https://zenodo.org/record/3555552#.YFxpQmQzb0o

    数据统计:
        - 2 个域：正常的 CIFAR-100 和被破坏的 CIFAR-100
        - 10 个类别

    参考文献:
        - Hendrycks 等人。基准测试神经网络对常见破坏和扰动的鲁棒性。ICLR 2019.
    """

    dataset_dir = ""  # 数据集目录
    domains = ["cifar100", "cifar100_c"]  # 定义两个域：正常 CIFAR-100 和破坏的 CIFAR-100

    def __init__(self, cfg):
        # 调用父类的构造函数
        super().__init__(cfg)

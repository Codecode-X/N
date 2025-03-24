import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


class DataManager:
    """数据管理器，用于加载数据集和构建数据加载器。
        参数：
        cfg (CfgNode): 配置。
        custom_tfm_train (list): 自定义训练数据增强。
        custom_tfm_test (list): 自定义测试数据增强。
        dataset_wrapper (DatasetWrapper): 数据集包装器。"""
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        """ 
        初始化数据管理器：构建 数据集 和 数据加载器。
        """
        # 构建数据集对象
        dataset = build_dataset(cfg)

        # 构建训练数据增强
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)  # 使用配置默认的训练数据增强
        else:
            print("* 使用自定义训练数据增强")
            tfm_train = custom_tfm_train  # 使用自定义的训练数据增强
        # 构建测试数据增强
        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)  # 使用配置默认的测试数据增强
        else:
            print("* 使用自定义测试数据增强")
            tfm_test = custom_tfm_test  # 使用自定义的测试数据增强

        # 根据配置信息，构建有标签训练数据加载器 train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,  # 采样器类型
            data_source=dataset.train_x,  # 数据源
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  # 批大小
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,  # 域数量
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,  # 每个类别的实例数量
            tfm=tfm_train,  # 训练数据增强
            is_train=True,  # 训练模式
            dataset_wrapper=dataset_wrapper  # 数据集包装器，用于对数据集进行增强
        )

        # 构建无标签训练数据加载器 train_loader_u
        train_loader_u = None
        if dataset.train_u:  # 如果存在无标签数据
            # 获取配置
            if cfg.DATALOADER.TRAIN_U.SAME_AS_X: # 如果配置要求与 train_x 使用 相同 的采样器和参数
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER # 采样器类型
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN # 域数量
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS # 每个类别的实例数量
            else: # 否则使用配置中 train_us 的参数
                sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER # 采样器类型
                batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN # 域数量
                n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS # 每个类别的实例数量
            # 构建无标签训练数据加载器
            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u, # 数据源
                batch_size=batch_size_,
                n_domain=n_domain_, # 域数量
                n_ins=n_ins_, # 每个类别的实例数量
                tfm=tfm_train, # 训练数据增强
                is_train=True, # 训练模式
                dataset_wrapper=dataset_wrapper  # 数据集包装器，用于对数据集进行增强
            )

        # 构建验证数据加载器 val_loader
        val_loader = None
        if dataset.val:  # 如果存在验证数据
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,  # 使用测试采样器
                data_source=dataset.val,  # 数据源
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,  # 批大小
                tfm=tfm_test,  # 测试数据增强
                is_train=False,  # 测试模式
                dataset_wrapper=dataset_wrapper # 数据集包装器，用于对数据集进行增强
            )

        # 构建测试数据加载器 test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,  # 使用测试采样器
            data_source=dataset.test,  # 数据源
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,  # 批大小
            tfm=tfm_test,  # 测试数据增强
            is_train=False,  # 测试模式
            dataset_wrapper=dataset_wrapper # 数据集包装器，用于对数据集进行增强
        )

        # 设置类属性
        self._num_classes = dataset.num_classes  # 类别数量
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)  # 源域数量
        self._lab2cname = dataset.lab2cname  # 类别到名称的映射

        # 数据集和数据加载器
        self.dataset = dataset # 数据集
        self.train_loader_x = train_loader_x # 有标签训练数据加载器
        self.train_loader_u = train_loader_u # 无标签训练数据加载器
        self.val_loader = val_loader # 验证数据加载器
        self.test_loader = test_loader # 测试数据加载器

        # 如果配置中启用了详细信息打印
        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        """返回类别数量。"""
        return self._num_classes

    @property
    def num_source_domains(self):
        """返回源域数量。"""
        return self._num_source_domains

    @property
    def lab2cname(self):
        """返回类别到名称的映射。"""
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        """打印数据集摘要信息。"""
        dataset_name = cfg.DATASET.NAME  # 数据集名称
        target_domains = cfg.DATASET.TARGET_DOMAINS  # 目标域

        # 构建摘要表格
        table = []
        table.append(["数据集", dataset_name])
        if target_domains:
            table.append(["目标域", target_domains])
        table.append(["类别数量", f"{self.num_classes:,}"])
        table.append(["有标签训练数据", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["无标签训练数据", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["验证数据", f"{len(self.dataset.val):,}"])
        table.append(["测试数据", f"{len(self.dataset.test):,}"])

        # 打印表格
        print(tabulate(table))


def build_data_loader(cfg, sampler_type="SequentialSampler", data_source=None, batch_size=64, 
                      n_domain=0, n_ins=2, tfm=None, is_train=True, dataset_wrapper=None):
    """构建数据加载器。
    参数：
        cfg (CfgNode): 配置。
        sampler_type (str): 采样器类型。
        data_source (list): 数据源。
        batch_size (int): 批大小。
        n_domain (int): 域数量。
        n_ins (int): 每个类别的实例数量。
        tfm (list): 数据增强。
        is_train (bool): 是否是训练模式。
        dataset_wrapper (DatasetWrapper): 数据集包装器。"""
    
    # 构建采样器
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper # 默认数据集包装器

    # 构建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train), # 经过数据集包装器处理的数据集
        batch_size=batch_size, 
        sampler=sampler, # 采样器
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=(is_train and len(data_source) >= batch_size), # 只有在 训练模式下 且 数据源的长度大于等于批大小时 才丢弃最后一个批次
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA) # 只有在 CUDA 可用且使用 CUDA 时才将数据存储在固定内存中
    )
    assert len(data_loader) > 0

    return data_loader



class DatasetWrapper(TorchDataset):
    """数据集包装器，用于对数据集进行增强。
    参数：
        cfg (CfgNode): 配置。
        data_source (list): 数据源。
        transform (list): 数据增强。
        is_train (bool): 是否是训练模式。"""

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        """ 初始化数据集包装器。"""
        self.cfg = cfg  # 保存配置
        self.data_source = data_source  # 数据源
        self.transform = transform  # 数据增强，接受列表或元组作为输入
        self.is_train = is_train  # 是否是训练模式
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1 # 在训练模式下，允许对图像进行 K 次增强
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0  # 是否返回未增强的原始图像

        # 如果需要对图像进行 K 次增强，但未提供 transform，则抛出异常
        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # 构建不应用任何数据增强的 self.to_tensor (步骤：调整大小、转换为张量、归一化)
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]  # 插值模式
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]  # 调整图像大小
        to_tensor += [T.ToTensor()]  # 转换为张量
        if "normalize" in cfg.INPUT.TRANSFORMS:  # 如果需要归一化
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD  # 使用均值和标准差
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)  # 将所有步骤组合成 transform

    def __len__(self):
        """返回数据源的长度。"""
        return len(self.data_source)

    def __getitem__(self, idx):
        """根据索引获取数据项。
        参数：
        - idx (int): 索引。
        返回：字典 output: 
            - label: 类别标签
            - domain: 域标签
            - impath: 图像路径
            - index: 索引
            - img: 增强后的图像 || - img1, img2, ...: 多次增强后的图像
            - img0: 未增强的原始图像
        """
        item = self.data_source[idx]  # 获取数据项

        # 初始化输出字典
        output = {
            "label": item.label,  # 类别标签
            "domain": item.domain,  # 域标签
            "impath": item.impath,  # 图像路径
            "index": idx,  # 索引
        }

        img0 = read_image(item.impath)  # 读取图像

        if self.transform is not None:  # 如果提供了 transform
            if isinstance(self.transform, (list, tuple)):  # 如果 transform 是列表或元组
                for i, tfm in enumerate(self.transform):  # 遍历每个 transform
                    img = self._transform_image(tfm, img0)  # 对图像进行增强
                    keyname = "img"  # 初始键名为 "img"
                    if (i + 1) > 1:  # 如果是多个 transform
                        keyname += str(i + 1)  # 键名后加编号
                    output[keyname] = img  # 保存增强后的图像：output["img1"], output["img2"], ...
            else:  # 如果 transform 不是列表或元组
                img = self._transform_image(self.transform, img0)  # 对图像进行增强
                output["img"] = img  # 保存增强后的图像
        else:  # 如果未提供 transform
            output["img"] = img0  # 保存原始图像

        if self.return_img0:  # 如果需要返回未增强的原始图像
            output["img0"] = self.to_tensor(img0)  # 应用基本 transform（无增强）

        return output  # 返回输出字典

    def _transform_image(self, tfm, img0):
        """对图像应用 transform 并返回结果。
        参数：
        - tfm (callable): transform 函数。
        - img0 (PIL.Image): 原始图像。
        返回：增强后的单个图像 (如果只有一个增强结果) imgs[0] || 增强后的图像列表 imgs。
        """
        img_list = []  # 初始化图像列表

        for k in range(self.k_tfm):  # 根据 k_tfm 的值重复增强
            img_list.append(tfm(img0))  # 对图像应用 transform 并添加到列表

        imgs = img_list  # 如果有多个增强结果，返回列表
        if len(imgs) == 1:  # 如果只有一个增强结果
            return imgs[0] # 返回单个图像
        else:  # 如果有多个增强结果
            return imgs  # 返回多次增强后的图像列表

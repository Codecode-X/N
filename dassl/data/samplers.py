import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


class RandomDomainSampler(Sampler):
    """随机采样 N 个域，每个域采样 K 张图片，
    组成一个大小为 N*K 的 minibatch。

    参数:
        data_source (list): Datums 的列表。| Datums 数据结构：{'img': 图像路径，'label': 类别标签，'domain': 域名，'classname': 类别名称}
        batch_size (int): 批量大小。
        n_domain (int): 在一个 minibatch 中采样的域数量。
    """

    def __init__(self, data_source, batch_size, n_domain):
        """初始化采样器。"""
        # 初始化数据源
        self.data_source = data_source

        # 使用 domain_dict(defaultdict) 记录每个域对应的图片索引列表
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        # 获取域名列表 domains
        self.domains = list(self.domain_dict.keys())

        if n_domain is None or n_domain <= 0: # 如果未指定 n_domain
            n_domain = len(self.domains) # 默认采样所有域
        
        # 确保 batch_size 可以被 n_domain 整除
        assert batch_size % n_domain == 0
        
        # 每个 batch 对 每个域中 采样的图片数量，确保每个域有相同数量的图片
        self.n_img_per_domain = batch_size // n_domain

        # 初始化 batch_size 和域数量
        self.batch_size = batch_size
        self.n_domain = n_domain

        # 估算一个 epoch 中的样本数量
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        """ 采样一个 minibatch 的索引。"""
        domain_dict = copy.deepcopy(self.domain_dict) # 深拷贝域字典，避免修改原始数据
        final_idxs = []  # 存储最终采样到的数据索引
        stop_sampling = False  # 停止采样标志

        while not stop_sampling:
            
            # 随机选择 n_domain 个域
            selected_domains = random.sample(self.domains, self.n_domain)
            for domain in selected_domains:
                # 获取当前域的数据索引列表
                idxs = domain_dict[domain]
                # 从当前域中随机采样 n_img_per_domain 个数据索引
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                # 将采样的数据索引添加到最终数据索引列表
                final_idxs.extend(selected_idxs)

                # 从域字典中移除已采样的索引
                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                # 检查当前域剩余的图片数量
                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    # 如果剩余图片不足以继续采样，停止采样
                    stop_sampling = True

        # 返回最终索引的迭代器
        return iter(final_idxs)

    def __len__(self):
        """返回样本数量"""
        return self.length


class SeqDomainSampler(Sampler):
    """顺序域采样器，从每个域中随机采样 K 张图片，
    组成一个 minibatch。

    参数:
        data_source (list): Datums 的列表。
        batch_size (int): 批量大小。
    """

    def __init__(self, data_source, batch_size):
        """初始化采样器。"""
        # 初始化数据源
        self.data_source = data_source

        # 使用 domain_dict(defaultdict) 记录每个域对应的图片索引列表
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        # 获取域名列表并排序
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        # 确保 batch_size 可以被 域数量 整除
        n_domain = len(self.domains)
        assert batch_size % n_domain == 0

        # 每个域中采样的图片数量
        self.n_img_per_domain = batch_size // n_domain

        # 初始化 batch_size 和域数量
        self.batch_size = batch_size
        self.n_domain = n_domain

        # 估算一个 epoch 中的样本数量
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        """ 采样一个 minibatch 的索引。"""
        # 深拷贝域字典，避免修改原始数据
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []  # 存储最终采样到的数据索引
        stop_sampling = False  # 停止采样标志

        while not stop_sampling:
            # 遍历每个域
            for domain in self.domains:
                # 获取当前域的数据索引列表
                idxs = domain_dict[domain]
                # 从当前域中随机采样 n_img_per_domain 个数据索引
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                # 将采样的数据索引添加到最终数据索引列表
                final_idxs.extend(selected_idxs)

                # 从域字典中移除已采样的索引
                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                # 检查当前域剩余的图片数量
                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    # 如果剩余图片不足以继续采样，停止采样
                    stop_sampling = True

        # 返回最终索引的迭代器
        return iter(final_idxs)

    def __len__(self):
        """返回样本数量"""
        return self.length


class RandomClassSampler(Sampler):
    """随机采样 N 个类别，每个类别采样 K 个实例，
    组成一个大小为 N*K 的 minibatch。

    修改自 https://github.com/KaiyangZhou/deep-person-reid。

    参数:
        data_source (list): Datums 的列表。
        batch_size (int): 批量大小。
        n_ins (int): 每个类别在一个 minibatch 中采样的实例数量。
    """

    def __init__(self, data_source, batch_size, n_ins):
        """初始化采样器。"""
        # 如果 batch_size 小于 n_ins，抛出异常
        if batch_size < n_ins:
            raise ValueError(
                "batch_size={} 必须不小于 "
                "n_ins={}".format(batch_size, n_ins)
            )

        # 初始化数据源、批量大小和每类实例数量
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_ins = n_ins

        # 每个 batch 中的类别数量
        self.ncls_per_batch = self.batch_size // self.n_ins

        # 使用 index_dic(defaultdict) 记录 每个类别 对应的 图片索引列表
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item.label].append(index)

        # 获取类别标签列表
        self.labels = list(self.index_dic.keys())

        # 确保类别数量不少于每个 batch 中的类别数量
        assert len(self.labels) >= self.ncls_per_batch

        # 估算一个 epoch 中的样本数量
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        """ 采样一个 minibatch 的索引。"""
        # 创建一个字典，用于存储每个类别的批次索引 batch_idxs_dict
        batch_idxs_dict = defaultdict(list)
        # 遍历每个类别，得到 每个类别 的 批次索引 batch_idxs_dict
        for label in self.labels:
            # 深拷贝当前类别的索引列表
            idxs = copy.deepcopy(self.index_dic[label])
            # 如果当前类别的实例数量不足 n_ins，则进行（重复）补充采样，确保每个类别至少有 n_ins 个实例
            if len(idxs) < self.n_ins:
                idxs = np.random.choice(idxs, size=self.n_ins, replace=True)
            # 随机打乱索引
            random.shuffle(idxs)
            # 将索引分成批次，每批次包含 n_ins 个索引
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.n_ins:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        # 深拷贝可用的类别标签列表 avai_labels
        avai_labels = copy.deepcopy(self.labels)
        
        # 最终采样到的数据索引列表 final_idxs
        final_idxs = []  
        
        # 当可用类别数量不少于每个 batch 的类别数量时，继续采样
        while len(avai_labels) >= self.ncls_per_batch:
            # 随机选择 ncls_per_batch 个类别
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)

            # 遍历选中的类别
            for label in selected_labels:
                # 从当前类别中取出一个批次的索引
                batch_idxs = batch_idxs_dict[label].pop(0)
                # 将索引添加到最终索引列表
                final_idxs.extend(batch_idxs)

                # 如果当前类别的批次索引已用完，从可用类别列表中移除
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        # 返回最终索引的迭代器
        return iter(final_idxs)

    def __len__(self):
        """返回样本数量"""
        return self.length


def build_sampler(sampler_type, cfg=None, data_source=None, batch_size=32, n_domain=0, n_ins=16):
    """根据采样器类型构建一个采样器。
    参数：
        sampler_type (str): 采样器类型。
        cfg (CfgNode): 配置参数。
        data_source (list): 数据源。
        batch_size (int): 批量大小。
        n_domain (int): 每个 minibatch 中的域数量。
        n_ins (int): 每个类别在一个 minibatch 中采样的实例数量。
    返回：
        采样器对象。
    """
    # 如果采样器类型是 "RandomSampler"
    if sampler_type == "RandomSampler":
        # 返回一个随机采样器
        return RandomSampler(data_source)

    # 如果采样器类型是 "SequentialSampler"
    elif sampler_type == "SequentialSampler":
        # 返回一个顺序采样器
        return SequentialSampler(data_source)

    # 如果采样器类型是 "RandomDomainSampler"
    elif sampler_type == "RandomDomainSampler":
        # 返回一个随机域采样器
        return RandomDomainSampler(data_source, batch_size, n_domain)

    # 如果采样器类型是 "SeqDomainSampler"
    elif sampler_type == "SeqDomainSampler":
        # 返回一个顺序域采样器
        return SeqDomainSampler(data_source, batch_size)

    # 如果采样器类型是 "RandomClassSampler"
    elif sampler_type == "RandomClassSampler":
        # 返回一个随机类别采样器
        return RandomClassSampler(data_source, batch_size, n_ins)

    # 如果采样器类型未知
    else:
        # 抛出异常，提示未知的采样器类型
        raise ValueError("Unknown sampler type: {}".format(sampler_type))

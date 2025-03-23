import torch.nn as nn


class ModelBase(nn.Module):
    """接口类 模型。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @property
    def fdim(self):
        """返回头部网络的输出特征维度"""
        return self._fdim
    
    def forward(self, x, return_feature=False):
        """
        前向传播。
        参数：
            x (torch.Tensor): 输入数据 [batch, ...]
            return_feature (bool): 是否返回特征
        返回：
            torch.Tensor: 输出结果 [batch, num_classes]
            (可选) torch.Tensor: 特征 [batch, ...]
        """
        raise NotImplementedError
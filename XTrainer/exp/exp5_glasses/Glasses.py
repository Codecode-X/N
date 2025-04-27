import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # 将输出重定向到log.txt文件
set_random_seed(3407)  # 设置随机种子
from Lens import CLIPGlassesLens
from Frame import CLIPGlassesFrame
import torch.nn as nn
import torch


class Glasses(nn.Module):
    def __init__(self, cfg):
        self.lens = CLIPGlassesLens.load_model(cfg['Lens'])
        self.frame = CLIPGlassesFrame.load_model(cfg['Frame'])
        
    def forward(self, I, h, level_h_list):
        """
        参数:
            - I: 图像特征 [B, D]
            - h: 最后一层特征 [B, D]
            - level_h_list: 各层特征列表 [B, L, D]
        返回:
            - scores_T2I: 文本->图像的分数 [N_caps, N_imgs=B]
            - scores_I2T: 图像->文本的分数 [N_imgs=B, N_caps]
        """
        h_neg = self.lens(h, level_h_list)
        scores_T2I = self.frame(I, h, h_neg)
        scores_I2T = scores_T2I.T
        return scores_T2I, scores_I2T
    
    @staticmethod
    def load_model(cfg, model_path):
        """
        加载模型
        参数:
            - cfg: 配置文件
            - model_path: 模型路径
        返回:
            - model: 加载的模型
        """
        model = Glasses(cfg)
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        return model
    
    
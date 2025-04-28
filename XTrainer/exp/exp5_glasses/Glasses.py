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
from McqDataset import McqDataset, evaluate_model_mcq
from RetrievalDataset import RetrievalDataset, evaluate_model_retrieval, retrieval_collate_fn
import torch.nn as nn
import torch


class Glasses(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.lens = CLIPGlassesLens.load_model(cfg['Lens'])
        self.frame = CLIPGlassesFrame.load_model(cfg['Frame'])
        
    def forward(self, I, h, level_h_list):
        """
        参数:
            - I: 图像特征 [N_imgs=B, D]
            - h: 最后一层特征 [N_caps=B*num_options, D]
            - level_h_list: 各层特征列表 [N_caps=B*num_options, L, D]
        返回:
            - scores_T2I: 文本->图像的分数 [N_caps, N_imgs=B]
            - scores_I2T: 图像->文本的分数 [N_imgs=B, N_caps]
        """
        h_neg = self.lens(h, level_h_list)
        assert I.size(0) == h_neg.size(0) == h.size(0), f"frame要求图片应该和文本一对一对应"
        scores_T2I = self.frame(I, h, h_neg)
        scores_I2T = scores_T2I.T
        return scores_T2I, scores_I2T
    
    @staticmethod
    def load_model(cfg):
        """
        加载模型
        参数:
            - cfg: 配置文件
            - model_path: 模型路径
        返回:
            - model: 加载的模型
        """
        model = Glasses(cfg)
        if cfg['model_path'] is not None:
            print(f"正在加载 Glasses 模型权重: {cfg['model_path']}, 将覆盖 Lens 和 Frame 的权重")
            model.load_state_dict(torch.load(cfg['model_path']))
        return model
   
    
    
if __name__ == "__main__":
    # Example usagerue
    cfg = {
        # -----模型参数-----
        'test_raw_clip': False, # 是否使用原始的CLIP模型进行测试
        'Lens': {
            'device': 'cuda',
            'dtype': torch.float32,
            'num_heads': 4,
            'dropout': 0.1,
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_lens_9832_0027.pth' # Lens的预训练权重
        },
        'Frame': {
            'device': 'cuda',
            'dtype': torch.float32,
            'lambda_0': 0.1, # 基础惩罚强度
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_Frame_mse_v1869.pth' # Frame的预训练权重
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/best_clip_Frame.pth' # Frame的预训练权重
        },
        
        'device': 'cuda',
        'dtype': torch.float32,
        'model_path': None, # Glasses(Lens+Frame)的预训练权重
        
        # -----数据参数-----
        'Mcq': {
            'batch_size': 64,
            'num_workers': 4,
            'num_options': 4,
            'dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
        },
        'Retrieval': {
            'batch_size': 64,
            'num_workers': 4,
            'dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv'
        },
    }
    print("==============配置项===============")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k1, v1 in v.items():
                print(f"  {k1}: {v1}")
        else:
            print(f"{k}: {v}")
    print("===================================")
    
    test_retrieval_dataset = RetrievalDataset(cfg['Retrieval']['dataset_path'])
    test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'], collate_fn=retrieval_collate_fn)

    if cfg['test_raw_clip'] is True:
        evaluate_model_retrieval(None, test_retrieval_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_retrieval(model, test_retrieval_dataloader, test_raw_clip=False)
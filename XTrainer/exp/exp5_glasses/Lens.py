import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model, extract_sentence_features
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # 将输出重定向到log.txt文件
set_random_seed(3407)  # 设置随机种子
import torch
import tqdm
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
from GlassesDataset import GlassesDataset
from torch.utils.data import DataLoader
import torch.optim as optim

class CLIPGlassesLens(nn.Module):
    """
    用于从CLIP文本编码器输出的最终文本特征(h)或中间层特征(level_h_list)中提取否定对象的文本特征(h_neg)。
    
    模型设计:
    前三层的特征用于语法流，分别乘以一个可学习的参数w1、w2、w3，生成Q1、Q2、Q3。
    最后一层的特征用于语义流，乘以一个可学习的参数w4，生成K， 乘以一个可学习的参数w5，w6,w7，生成V1、V2、V3。
    Q1、Q2、Q3 和 K 进行点积，得到注意力权重矩阵A1、A2、A3。
    A1、A2、A3 和 V1、V2、V3 进行加权求和，得到最终的否定对象文本特征c_neg。
    c_neg 经过一个ffn层，得到最终的否定对象文本特征 h_neg。
    
    模型训练:
    - 监督信号:
        - neg_obj： 否定对象文本特征 [batch_size, embed_dim]
        - neg_obj： 肯定对象文本特征 [batch_size, embed_dim]
    - 损失函数:
        - 1-neg_sim.mean() 目标是最大化预测的h_neg和真实的neg_obj之间的余弦相似度
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = 512
        
        # 残差连接门控机制
        self.res_gate = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()  # 直接内置Sigmoid
        )
        
        # 语法流参数 (前三层)
        self.syntax_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(3)
        ])
        
        # 语义流参数 (最后一层)
        self.semantic_k_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim)
        )
        self.semantic_v_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(3)
        ])
        
        # 注意力融合模块
        attention_scale = torch.tensor([1.0, 0.8, 0.6])
        self.attention_log_scale = nn.Parameter(torch.log(attention_scale))
        
        # FFN网络
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.embed_dim),
            nn.LayerNorm(2*self.embed_dim),  # 新增层归一化
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        # 残差门控层初始化
        nn.init.constant_(self.res_gate[0].bias, -2.0)  # 初始偏向原始特征
        nn.init.normal_(self.res_gate[0].weight, mean=0.0, std=0.02)
        
        # 语法流投影层初始化（GELU）
        for proj in self.syntax_proj:
            nn.init.kaiming_normal_(
                proj[0].weight, 
                mode='fan_in',
                nonlinearity='relu'
            )
        
        # 语义流投影层初始化（GELU）
        for proj in self.semantic_v_proj:
            nn.init.kaiming_normal_(
                proj[0].weight, 
                mode='fan_in',
                nonlinearity='relu'
            )
         
            
    @staticmethod        
    def load_model(cfg):
        model = CLIPGlassesLens(cfg)
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"正在加载 CLIPGlassesLens 模型权重: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=True))
        model = model.to(cfg['device'])
        model.eval()
        return model
        
    def forward(self, h, level_h_list):
        """
        参数:
            - h: 最后一层特征 [B, D]
            - level_h_list: 各层特征列表 [B, L, D]
        返回:
            - h_neg: 否定对象文本特征 [B, D]
        """
        # 语法流处理 (前三层)
        syntax_feats = []
        for i in range(3):
            layer_feat = level_h_list[:, i, :]  # 第i层特征
            proj_feat = self.syntax_proj[i](layer_feat)  # [B, D]
            syntax_feats.append(proj_feat)
        
        # 语义流处理 (最后一层)
        K = self.semantic_k_proj(h)  # [B, D] | K (key)
        Vs = [proj(h) for proj in self.semantic_v_proj]  # 3x[B, D] | V1,V2,V3 (value)
        
        # 分层注意力计算
        c_neg = 0
        scale_factor = torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)) # 注意力缩放因子
        for i in range(3):
            Q = syntax_feats[i] # [B, D] | Q1,Q2,Q3 (query)
            A = torch.bmm(Q.unsqueeze(1), K.unsqueeze(-1)).squeeze(-1) / scale_factor  # [B, 1]
            # A = F.softmax(A * self.attention_scale[i], dim=0)  # 归一化注意力权重
            A = F.softmax(A * self.attention_log_scale[i], dim=0)  # 归一化注意力权重
            c_neg += A * Vs[i]
        
        # 残差连接并添加门控机制
        gate = torch.sigmoid(self.res_gate(c_neg)) 
        c_neg = gate * c_neg + (1 - gate) * h  # 添加可学习的残差门控
        
        # FFN处理
        h_neg = self.ffn(c_neg)
        
        return h_neg

    def calc_losses(self, h_neg, neg_obj, I, h=None):
        """
        Args:
            h_neg: 预测的否定特征 [B,D]
            neg_obj: 真实否定对象特征 [B,D]
            I: 图像特征 [B,D]
            pos_obj: 真实肯定对象特征 [B,D]
            h_original: 原始特征 [B,D]
        """
        h_neg_n = F.normalize(h_neg, p=2, dim=-1)
        neg_obj_n = F.normalize(neg_obj, p=2, dim=-1)
        I_n = F.normalize(I, p=2, dim=-1)
        h_n = F.normalize(h, p=2, dim=-1)
        
        neg_sim = (h_neg_n * neg_obj_n).sum(dim=-1)  # [B] 越高越好
        neg_sim_loss = 1 - neg_sim.mean()  # 越低越好
        
        obj2i_sim = (neg_obj_n * I_n).sum(dim=-1)  # [B] 越高越好
        
        n2i_sim = (h_neg_n * I_n).sum(dim=-1)  # [B] 应该和obj2i_sim越接近越好 | 保证与图像特征的对齐
        n2i_sim_loss =  abs(obj2i_sim - n2i_sim).mean()  # 越低越好
        
        total_loss = neg_sim_loss + n2i_sim_loss  # 越低越好
        
        return total_loss, {
            'neg_sim_loss': neg_sim_loss.item(),
            'n2i_sim_loss': n2i_sim_loss.item(),
        }

def train(cfg, model:CLIPGlassesLens, device='cuda'):
    """Train the CLIPGlassesLens model"""
            
    if cfg:
        epochs = cfg['epochs']
        batch_size = cfg['batch_size']
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        train_size, val_size, test_size = cfg['split']
        num_workers = cfg['num_workers']
        early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
        
    dataset = GlassesDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # Training loop
    for epoch in range(epochs):
        model.train()
        best_sim_loss = float('inf')
        patience_counter = 0 # Early stopping counter
        total_loss = 0
        losses = {'neg_sim_loss': 0, 'n2i_sim_loss': 0}
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            I = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            h = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_h_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            neg_obj = batch['neg_obj'].to(device) # [batch_size, num_objs, embed_dim]
            
            # Forward pass
            h_neg = model(h, level_h_list)
            
            # Compute loss
            loss, loss_dict = model.calc_losses(h_neg, neg_obj, I, h)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['neg_sim_loss'] += loss_dict['neg_sim_loss']
            losses['n2i_sim_loss'] += loss_dict['n2i_sim_loss']
        
        scheduler.step()
        
        # Print epoch summary
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f}  \
                neg_sim_loss: {losses['neg_sim_loss']/batch_count:.4f} \
                n2i_sim_loss: {losses['n2i_sim_loss']/batch_count:.4f}")
        
        # Validation
        batch_sim_loss = evaluate(cfg, model, val_loader)
        
        # 早停 
        if batch_sim_loss < best_sim_loss:
            best_sim_loss = batch_sim_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
        else:
            print(f"💔loss improve from {best_sim_loss:.4f} to {batch_sim_loss:.4f}, cur patience_counter add to {patience_counter}")
            patience_counter += 1 # 增加耐心计数器
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
    return model


def evaluate(cfg, model, data_loader, device='cuda'):
    """
    Evaluate the CLIPGlassesLens model
    
    参数：
        - cfg: 配置参数
        - model: CLIPGlassesLens模型
        - data_loader: 数据加载器
        - device: 设备（'cuda'或'cpu'）
        
    返回：
        - None
    """
    model.eval()
    model = model.to(device)
    p_sim_total = 0
    h_sim_total = 0
    neg_sim_total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            neg_obj = batch['neg_obj'].to(device)
            
            # Forward pass
            h_neg = model(h, level_h_list)
            
            # Normalize features for cosine similarity
            h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            h_norm = h / h.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            neg_obj_norm = neg_obj / neg_obj.norm(dim=-1, keepdim=True) # [batch_size, num_objs, embed_dim]
            
            # 1. Similarity alignment loss
            p_sim = F.cosine_similarity(l_pos_norm, h_neg_norm, dim=-1) # [batch_size]
            h_sim = F.cosine_similarity(h_norm, h_neg_norm, dim=-1) # [batch_size]
            neg_sim = F.cosine_similarity(neg_obj_norm, h_neg_norm, dim=-1) # [batch_size, num_objs]
            
            p_sim = torch.mean(p_sim).item()
            h_sim = torch.mean(h_sim).item()
            neg_sim = torch.mean(neg_sim).item()
            
            p_sim_total += p_sim
            h_sim_total += h_sim
            neg_sim_total += neg_sim
    
    batch_count = len(data_loader)
    p_sim_total = p_sim_total / batch_count
    h_sim_total = h_sim_total / batch_count
    neg_sim_total = neg_sim_total / batch_count
    
    print(f"Evaluation results:")
    print(f"  p_sim: {p_sim_total:.4f}")
    print(f"  h_sim: {h_sim_total:.4f}")
    print(f"  neg_sim: {neg_sim_total:.4f}")
    
    sim_loss = (1-neg_sim_total)
    return sim_loss  # 越小越好


if __name__ == "__main__":
    # Paths to datasets
    negated_csv = "/root/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    original_csv = "/root/COCO_val_retrieval.csv"
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        'device': 'cuda',
        'num_heads': 4,
        
        'dropout': 0.1,
        'margin': 0.5,
        
        'model_path': os.path.join(current_dir, 'best_clip_lens.pth'), # Lens的预训练权重
        'save_path': os.path.join(current_dir, 'best_clip_lens.pth'), # Lens的训练权重保存路径
        
        # -----训练参数-----
        'epochs': 30,
        'batch_size': 32,
        'lr': 1e-3, # full:0.9831 val:0.9765
        # 'lr': 1e-2, # full:0.9857 val:0.9770
        # 'lr': 5e-2, # full:0.9750 val:0.9786
        'weight_decay': 0.05,
        'early_stop_patience': 5, # Early stopping patience
        'test_only': False, # 是否只测试
        
        # -----数据参数-----
        'pos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
        'negpos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
        'split': [0.9, 0.1, 0.0],  # train, val, test split
        'num_workers': 4,
        'test_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
    }
    
    test_dataset = GlassesDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=cfg['num_workers'])
    
    # if not cfg['test_only']:
    #     model = CLIPGlassesLens(cfg)
    #     # Train model
    #     trained_model = train(cfg, model)
    #     # 测试模型为训练的模型
    #     cfg['model_path'] = cfg['save_path']
    
    trained_model = CLIPGlassesLens.load_model(cfg)
    trained_model = trained_model.to('cuda')
    
    print("\nFinal evaluation on test set:")
    evaluate(cfg, trained_model, test_loader)
    
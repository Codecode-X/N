import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model, extract_sentence_features
from utils import Logger, set_random_seed
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
    def __init__(self, cfg):
        super().__init__()
        embed_dim = Clip_model.visual.output_dim
        self.num_layers = 12  # CLIP有12层
        hidden_dim = cfg['hidden_dim']
        num_heads = cfg['num_heads']
        dropout = cfg['dropout']
        self.syntax_level = cfg['syntax_level'] # 语法流使用前3层的特征
        
        # 层级特征选择器
        self.layer_selector = nn.Sequential(
            nn.Linear(embed_dim * self.num_layers, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_layers),
            nn.Softmax(dim=-1)
        )

        # 语法-语义双流处理
        self.syntax_flow = nn.ModuleDict({ # 语法流
            'transformer': nn.TransformerEncoderLayer( # transformer编码器提取语法特征
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            'neg_detector': nn.Sequential( # 对语法特征进行否定检测
                nn.Linear(embed_dim, hidden_dim//2),
                nn.GELU(),
                nn.Linear(hidden_dim//2, 1),
                nn.Sigmoid()
            )
        })

        self.semantic_flow = nn.ModuleDict({ # 语义流
            'attention': nn.MultiheadAttention(embed_dim, num_heads, dropout),
            'fusion': nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        })

        # 动态门控生成
        self.gate_generator = nn.Sequential(
            nn.Linear(embed_dim*3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )

        # 语义分离模块
        self.neg_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 语法检测器初始化为捕捉否定模式
        nn.init.uniform_(self.syntax_flow['neg_detector'][-2].weight, -0.1, 0.1)
        nn.init.constant_(self.syntax_flow['neg_detector'][-2].bias, 1.0)

    def _select_layers(self, level_h_list):
        """层级特征选择与融合"""
        batch_size = level_h_list.size(0)
        
        # 计算层级权重
        layer_weights = self.layer_selector(
            level_h_list.view(batch_size, -1))  # [B, num_layers]
        
        # 加权融合
        selected_feat = torch.einsum('bld,bl->bd', level_h_list, layer_weights)
        return selected_feat

    def _process_syntax(self, h, level_h_list):
        """语法流处理"""
        # 使用底层特征（假设前3层）检测否定结构
        syntax_feats = level_h_list[:, :self.syntax_level, :].mean(dim=1)  # [B, D]
        # 增强语法表示
        syntax_attn = self.syntax_flow['transformer'](h.unsqueeze(1)).squeeze(1)
        # 检测否定概率
        neg_prob = self.syntax_flow['neg_detector'](syntax_feats)  # [B, 1]
        return syntax_attn, neg_prob

    def _process_semantic(self, h):
        """语义流处理"""
        # 多头注意力增强
        semantic_attn, _ = self.semantic_flow['attention'](h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        semantic_attn = semantic_attn.squeeze(1)
        
        # 特征融合
        fused_semantic = self.semantic_flow['fusion'](
            torch.cat([h, semantic_attn], dim=-1))
        return fused_semantic

    def forward(self, h, level_h_list):
        # 层级特征选择
        layer_fused = self._select_layers(level_h_list)  # [B, D]
        # 语法流处理
        syntax_feat, neg_prob = self._process_syntax(h, level_h_list)
        # 语义流处理
        semantic_feat = self._process_semantic(h)
        
        # 动态门控生成
        gate_input = torch.cat([layer_fused, syntax_feat, semantic_feat], dim=-1)
        gate = self.gate_generator(gate_input)  # [B, D]
        
        # 语义分离
        h_neg = gate * self.neg_predictor(semantic_feat)
        h_pos = h - h_neg * neg_prob  # 用否定概率加权
        
        return h_pos, h_neg


def compute_losses(cfg, h_pos, h_neg, l_pos, h, neg_obj):
    """
    Compute all loss components for CLIPGlassesLens
    
    参数:
        - cfg: 配置参数
        - h_pos: 肯定内容文本特征 [batch_size, embed_dim] | This image shows a person.
        - h_neg: 否定内容文本特征 [batch_size, embed_dim]
        - l_pos: 监督信号-肯定内容文本特征 [batch_size, embed_dim]
        - h: 原始文本特征 [batch_size, embed_dim] | This image shows a person, but no motorbike is included.
        - neg_obj: 否定对象文本特征 [batch_size, embed_dim] | f('motorbike.')
    
    返回:
        - total_loss: 总损失
        - loss_dict: 各个分项损失的字典 | pos2pos_sim_loss, pos2negobj_sim_loss, sparse_loss
    """
    
    # Normalize features for cosine similarity
    h_pos_norm = h_pos / h_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    h_norm = h / h.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    neg_obj_norm = neg_obj / neg_obj.norm(dim=-1, keepdim=True) # [batch_size, num_objs, embed_dim]
    
    # 1. Similarity alignment loss
    pos2pos_sim = F.cosine_similarity(h_pos_norm, l_pos_norm, dim=-1) # [batch_size] 越大越好
    loss_pos2pos = 1.0 - pos2pos_sim # [batch_size] 越小越好
    
    # Total loss
    total_loss = loss_pos2pos
    
    return total_loss.mean(), {
        'pos2pos_sim_loss': loss_pos2pos.mean().item(),
    }


def train(cfg, model, device='cuda'):
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
        losses = {'pos2pos_sim_loss': 0, 'pos2negobj_sim_loss': 0}
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            h = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_h_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device)
            neg_obj = batch['neg_obj'].to(device) # [batch_size, num_objs, embed_dim]
            
            # Forward pass
            h_pos, h_neg = model(h, level_h_list)
            
            # Compute loss
            loss, loss_dict = compute_losses(cfg, h_pos, h_neg, l_pos, h, neg_obj)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['pos2pos_sim_loss'] += loss_dict['pos2pos_sim_loss']
            # losses['pos2negobj_sim_loss'] += loss_dict['pos2negobj_sim_loss']
        
        scheduler.step()
        
        # Print epoch summary
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f}  \
                pos2pos_sim_loss: {losses['pos2pos_sim_loss']/batch_count:.4f}\
                # pos2negobj_sim_loss: {losses['pos2negobj_sim_loss']/batch_count:.4f}"
                )
        
        # Validation
        batch_sim_loss = evaluate(cfg, model, val_loader)
        
        # 早停
        if batch_sim_loss < best_sim_loss:
            best_sim_loss = batch_sim_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_clip_lens.pth'))
        else:
            print(f"💔loss improve from {best_sim_loss:.4f} to {batch_sim_loss:.4f}, cur patience_counter add to {patience_counter}")
            patience_counter += 1 # 增加耐心计数器
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        if (epoch+1) % 10 == 0:
            print(f"visualize_examples")
            visualize_examples(model, top_k=5)
        
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
    pos_sim_total = 0
    neg_sim_total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            
            # Forward pass
            h_pos, h_neg = model(h, level_h_list)
            
            # Normalize features for cosine similarity
            h_pos_norm = h_pos / h_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            
            # 1. Similarity alignment loss
            pos2pos_sim = F.cosine_similarity(h_pos_norm, l_pos_norm, dim=-1) # [batch_size]
            pos2neg_sim = F.cosine_similarity(h_pos_norm, h_neg_norm, dim=-1) # [batch_size]
            pos2pos_sim = torch.mean(pos2pos_sim).item()
            pos2neg_sim = torch.mean(pos2neg_sim).item()
            
            pos_sim_total += pos2pos_sim
            neg_sim_total += pos2neg_sim
    
    batch_count = len(data_loader)
    pos_sim_total = pos_sim_total / batch_count
    neg_sim_total = neg_sim_total / batch_count
    
    print(f"Evaluation results:")
    print(f"  Pos Similarity: {pos_sim_total:.4f}")
    print(f"  Neg Similarity: {neg_sim_total:.4f}")
    
    sim_loss = (1-pos_sim_total)
    return sim_loss  # 越小越好


def visualize_examples(model, top_k=5):
    model = model.to('cuda')
    examples = [
        "This image shows a person, but no motorbike is included.", # p:['person']  n:['motorbike']
        "This image features a car, but no knife is present.", # p:['car']  n:['knife']
        "A bus is included, while a car is absent.", # p:['bus']  n:['car']
        "A bird is present in this image, with no person in sight." # p:['bird']  n:['person']
    ]
    candidates = ['bench', 'camera', 'gloves', 'woman', 'screwdriver', 'egg', 'knife', 'plate', 'car', 'motorbike', 'bus', 'bird', 'person']

    """可视化模型预测效果"""
    print("="*50)
    print("CLIP-Lens Prediction Examples")
    print("="*50)
        
    # 提取候选对象文本特征
    candidate_features = []
    for obj in candidates:
        text = f"This image shows a {obj}."
        feature, _ = extract_sentence_features(text)
        candidate_features.append(feature)
    candidate_features = torch.tensor(candidate_features, dtype=cfg['dtype']).to('cuda') # [num_candidates, embed_dim]
    
    for sentence in examples:
        print(f"\nInput: {sentence}")
        
        with torch.no_grad():
            features, level_h_list = extract_sentence_features(sentence)
            h = torch.tensor(features, dtype=cfg['dtype']).unsqueeze(0).to('cuda') # [1, embed_dim]
            level_h_list = torch.stack([torch.tensor(l, dtype=cfg['dtype']) for l in level_h_list]).unsqueeze(0).to('cuda') # [1, num_layers, embed_dim]
            h_pos, h_neg = model(h, level_h_list)
        
        # 计算相似度
        pos_sim = F.cosine_similarity(h_pos, candidate_features, dim=-1) # [batch_size, num_candidates]
        # neg_sim = F.cosine_similarity(h_neg, candidate_features, dim=-1) # [batch_size, num_candidates] 
        pos_similarities = pos_sim.cpu().numpy()
        # neg_similarities = neg_sim.cpu().numpy()   
             
        # 获取前K个最相似的对象
        top_pos_indices = np.argsort(-pos_similarities)[:top_k]
        print("Top positive objects:")
        for idx in top_pos_indices:
            print(f"  - {candidates[idx]}: {pos_similarities[idx]:.4f}")
            
        # 计算h和候选对象的相似度
        h_sim = F.cosine_similarity(h, candidate_features, dim=-1) # [batch_size, num_candidates]
        h_similarities = h_sim.cpu().numpy()
        # 获取前K个最相似的对象
        top_h_indices = np.argsort(-h_similarities)[:top_k]
        print("Top raw clip objects:")
        for idx in top_h_indices:
            print(f"  - {candidates[idx]}: {h_similarities[idx]:.4f}")
        
        # top_neg_indices = np.argsort(-neg_similarities)[:top_k]
        # print("Top negative objects:")
        # for idx in top_neg_indices:
        #     print(f"  - {candidates[idx]}: {neg_similarities[idx]:.4f}")
        
        print("-"*50)
        
        
def visualize(cfg):
    examples = [
        "This image shows a person, but no motorbike is included.", # p:['person']  n:['motorbike']
        "This image features a car, but no knife is present.", # p:['car']  n:['knife']
        "A bus is included, while a car is absent.", # p:['bus']  n:['car']
        "A bird is present in this image, with no person in sight." # p:['bird']  n:['person']
    ]
    labels = [
        "This image shows a person",
        "This image features a car",
        "A bus is included",
        "A bird is present in this image"
    ]
    candidates = ['bench', 'camera', 'gloves', 'woman', 'screwdriver', 'egg', 'knife', 'plate', 'car', 'motorbike', 'bus', 'bird', 'person']

    """可视化模型预测效果"""
    print("="*50)
    print("CLIP-Lens Prediction Examples")
    print("="*50)
        
    # 提取候选对象文本特征
    candidate_features = []
    for obj in candidates:
        text = f"This image shows a {obj}."
        feature, _ = extract_sentence_features(text)
        candidate_features.append(feature)
    candidate_features = torch.tensor(candidate_features, dtype=cfg['dtype']).to('cuda') # [num_candidates, embed_dim]
    
    for sentence, label in zip(examples, labels):
        print(f"\nInput: {sentence}")
        with torch.no_grad():
            s_features, _ = extract_sentence_features(sentence)
            s_features_tensor = torch.tensor(s_features, dtype=cfg['dtype']).unsqueeze(0).to('cuda')
            l_features, _ = extract_sentence_features(label)
            l_features_tensor = torch.tensor(l_features, dtype=cfg['dtype']).unsqueeze(0).to('cuda')
            # 计算 s_features_tensor 和 l_features_tensor 的相似度
            sim2label = F.cosine_similarity(s_features_tensor, l_features_tensor, dim=-1)
            print(f"  Similarity with label: {sim2label.item():.4f}")
            # 计算 s_features_tensor 和 l_features_tensor 的mse
            mse = F.mse_loss(s_features_tensor, l_features_tensor)
            print(f"  MSE with label: {mse.item():.4f}")
            print("-"*50)
            # s_features_tensor 和 候选对象 的余弦相似度
            s2objs = F.cosine_similarity(s_features_tensor, candidate_features, dim=-1).cpu().numpy() # [batch_size, num_candidates]
            # l_features_tensor 和 候选对象 的余弦相似度
            l2objs = F.cosine_similarity(l_features_tensor, candidate_features, dim=-1).cpu().numpy()
            # 获取前K个最相似的对象
            top_s_indices = np.argsort(-s2objs)[:5]
            top_l_indices = np.argsort(-l2objs)[:5]
            print("Top positive objects:")
            print('----top_s_indices----')
            for idx in top_s_indices:
                print(f"  - {candidates[idx]}: {s2objs[idx]:.4f}") 
            print('----top_l_indices----')   
            for idx in top_l_indices:
                print(f"  - {candidates[idx]}: {l2objs[idx]:.4f}")
            

if __name__ == "__main__":
    # Paths to datasets
    negated_csv = "/root/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    original_csv = "/root/COCO_val_retrieval.csv"
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        'hidden_dim': Clip_model.visual.output_dim * 2,
        'num_heads': 4,
        'dropout': 0.1,
        'syntax_level': 3, # 语法流使用前3层的特征
        
        # -----训练参数-----
        'epochs': 60,
        'batch_size': 32,
        'lr': 1e-3,
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
    
    sys.stdout = Logger(os.path.join(current_dir, "log.txt"))  # 将输出重定向到log.txt文件
    set_random_seed(3407)  # 设置随机种子为42
    
    test_dataset = GlassesDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=cfg['num_workers'])
    
    if not cfg['test_only']:
        model = CLIPGlassesLens(cfg)
        visualize_examples(model, top_k=10)
        # Train model
        trained_model = train(cfg, model)
    
    # Final evaluation
    trained_model = CLIPGlassesLens(cfg)
    trained_model.load_state_dict(torch.load(os.path.join(current_dir, 'best_clip_lens.pth')))
    trained_model.eval()
    trained_model = trained_model.to('cuda')
    print("\nFinal evaluation on test set:")
    evaluate(cfg, trained_model, test_loader)
    
    # after trainning visualize examples
    visualize_examples(trained_model, top_k=5)
    
    # visualize(cfg)
    
    
    """
    'syntax_level': 3
    This image shows a person, but no motorbike is included.
    Top positive objects:
    - person: 0.9318
    - motorbike: 0.9247
    - woman: 0.9118
    - car: 0.8789
    - camera: 0.8580
    
    This image features a car, but no knife is present.
    Top positive objects:
    - car: 0.9502
    - bus: 0.8745
    - knife: 0.8726
    - person: 0.8650
    - camera: 0.8564
    
    'syntax_level': 12
    Input: This image shows a person, but no motorbike is included.
    Top positive objects:
    - person: 0.9314
    - motorbike: 0.9309
    - woman: 0.9134
    - car: 0.8935
    - camera: 0.8766
    
    Input: This image features a car, but no knife is present.
    Top positive objects:
    - car: 0.9364
    - knife: 0.8736
    - bus: 0.8703
    - person: 0.8516
    - camera: 0.8454
    """